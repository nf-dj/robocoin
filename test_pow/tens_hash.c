#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sodium.h>

#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64

typedef enum {
    IMPL_INT8 = 0,
    IMPL_FP32 = 1,
    IMPL_FP16 = 2
} ImplType;

typedef struct {
    int8_t **middle_mats[ROUNDS];
    ImplType impl_type;
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;
} HashBuffers;

static void matrix_multiply_mod2_int8(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    //printf("sums: ");
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            int8_t w=A[i][j];
            //int8_t w=rand()%3-1;
            sum += w * in[j];
        }
        sum += noise[i];
        //out[i] = sum & 1;
        //printf("%d ",sum);
        out[i] = sum > 0 ? 1 : 0;
    }
    //printf("\n");
}

static void matrix_multiply_mod2_fp32(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (float)A[i][j] * (float)in[j];
        }
        sum += (float)noise[i];
        out[i] = ((int)sum) & 1;
    }
}

static void matrix_multiply_mod2_fp16(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        _Float16 sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (_Float16)A[i][j] * (_Float16)in[j];
        }
        sum += (_Float16)noise[i];
        out[i] = ((int)sum) & 1;
    }
}

static void matrix_multiply_mod2(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, ImplType impl_type) {
    switch(impl_type) {
        case IMPL_FP32:
            matrix_multiply_mod2_fp32(A, in, out, noise, rows, cols);
            break;
        case IMPL_FP16:
            matrix_multiply_mod2_fp16(A, in, out, noise, rows, cols);
            break;
        default:
            matrix_multiply_mod2_int8(A, in, out, noise, rows, cols);
    }
}

static void generate_matrices(int8_t **middle_mats[ROUNDS], uint8_t seed[32]) {
    size_t total_size = ROUNDS * HIDDEN * HIDDEN;
    uint8_t *data = malloc(total_size);
    if (!data) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    unsigned char nonce[crypto_stream_chacha20_NONCEBYTES] = {0};
    crypto_stream_chacha20(data, total_size, nonce, seed);

    uint8_t *pos = data;
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < HIDDEN; j++) {
                uint8_t val = pos[i * HIDDEN + j] % 3;
                middle_mats[r][i][j] = val - 1;
            }
        }
        pos += HIDDEN * HIDDEN;
    }
    free(data);
}

PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type) {
    PrecomputedMatrices* matrices = malloc(sizeof(PrecomputedMatrices));
    if (!matrices) return NULL;
    
    matrices->impl_type = impl_type;

    for (int r = 0; r < ROUNDS; r++) {
        matrices->middle_mats[r] = malloc(HIDDEN * sizeof(int8_t*));
        if (!matrices->middle_mats[r]) {
            for (int j = 0; j < r; j++) {
                free(matrices->middle_mats[j][0]);
                free(matrices->middle_mats[j]);
            }
            free(matrices);
            return NULL;
        }
        matrices->middle_mats[r][0] = malloc(HIDDEN * HIDDEN * sizeof(int8_t));
        if (!matrices->middle_mats[r][0]) {
            for (int j = 0; j < r; j++) {
                free(matrices->middle_mats[j][0]);
                free(matrices->middle_mats[j]);
            }
            free(matrices->middle_mats[r]);
            free(matrices);
            return NULL;
        }
        for (int i = 1; i < HIDDEN; i++) {
            matrices->middle_mats[r][i] = matrices->middle_mats[r][0] + (i * HIDDEN);
        }
    }

    generate_matrices(matrices->middle_mats, seed);
    return matrices;
}

HashBuffers* init_hash_buffers(void) {
    HashBuffers* buffers = malloc(sizeof(HashBuffers));
    if (!buffers) return NULL;
    
    buffers->state = calloc(HIDDEN, sizeof(uint8_t));
    buffers->next_state = calloc(HIDDEN, sizeof(uint8_t));
    buffers->noise = malloc(ROUNDS * HIDDEN * sizeof(int8_t));
    
    if (!buffers->state || !buffers->next_state || !buffers->noise) {
        if (buffers->state) free(buffers->state);
        if (buffers->next_state) free(buffers->next_state);
        if (buffers->noise) free(buffers->noise);
        free(buffers);
        return NULL;
    }
    
    return buffers;
}

void free_matrices(PrecomputedMatrices* matrices) {
    if (matrices) {
        for (int r = 0; r < ROUNDS; r++) {
            if (matrices->middle_mats[r]) {
                free(matrices->middle_mats[r][0]);
                free(matrices->middle_mats[r]);
            }
        }
        free(matrices);
    }
}

void free_hash_buffers(HashBuffers* buffers) {
    if (buffers) {
        free(buffers->state);
        free(buffers->next_state);
        free(buffers->noise);
        free(buffers);
    }
}

void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices, 
                          HashBuffers* buffers, uint8_t output[IN_SIZE]) {
    // Generate noise from input
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256(digest, input, IN_SIZE);
    for (size_t i = 0; i < ROUNDS * HIDDEN; i++) {
        //buffers->noise[i] = (digest[i % 32] >> (i % 8)) & 1;
        buffers->noise[i] = rand() % 3 - 1; // FIXME
    }

    // Convert byte input to bits
    for (int i = 0; i < IN_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            buffers->state[i*8 + j] = (input[i] >> j) & 1;
        }
    }

    uint32_t round;
    for (round = 0; round < ROUNDS; round++) {
        /*printf("round %d: ",round);
        for (int i = 0; i < HIDDEN; i++)
            printf("%02x", buffers->state[HIDDEN-1-i]);
        printf("\n");*/
        matrix_multiply_mod2(matrices->middle_mats[round], buffers->state, 
                           buffers->next_state, buffers->noise + (round * HIDDEN), 
                           HIDDEN, HIDDEN, matrices->impl_type);
        uint8_t *temp = buffers->state;
        buffers->state = buffers->next_state;
        buffers->next_state = temp;
    }
    /*printf("round %d: ",round);
    for (int i = 0; i < HIDDEN; i++)
        printf("%02x", buffers->state[HIDDEN-1-i]);
    printf("\n");*/

    // Convert bits back to bytes for output
    memset(output, 0, IN_SIZE);
    for (int i = 0; i < IN_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            output[i] |= buffers->state[i*8 + j] << j;
        }
    }
}

void tens_hash(uint8_t input[IN_SIZE], uint8_t seed[32], uint8_t output[IN_SIZE], ImplType impl_type) {
    PrecomputedMatrices* matrices = precompute_matrices(seed, impl_type);
    if (!matrices) {
        fprintf(stderr, "Failed to precompute matrices\n");
        exit(1);
    }
    HashBuffers* buffers = init_hash_buffers();
    if (!buffers) {
        free_matrices(matrices);
        fprintf(stderr, "Failed to initialize hash buffers\n");
        exit(1);
    }
    tens_hash_precomputed(input, matrices, buffers, output);
    free_matrices(matrices);
    free_hash_buffers(buffers);
}

int hexchar_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int parse_hex(const char *hex, size_t hex_len, uint8_t *out, size_t out_len) {
    if (hex_len != out_len * 2) return -1;
    for (size_t i = 0; i < out_len; i++) {
        int hi = hexchar_to_int(hex[2 * (out_len - 1 - i)]);
        int lo = hexchar_to_int(hex[2 * (out_len - 1 - i) + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (hi << 4) | lo;
    }
    return 0;
}

#ifdef HASH_MAIN
int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (sodium_init() < 0) {
        fprintf(stderr, "Error: libsodium initialization failed\n");
        return 1;
    }
    
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <seed_hex> <input_hex> [impl_type]\n", argv[0]);
        fprintf(stderr, "impl_type: 0 for int8 (default), 1 for fp32, 2 for fp16\n");
        return 1;
    }
    
    ImplType impl_type = IMPL_INT8;  // Default
    if (argc == 4) {
        int type = atoi(argv[3]);
        if (type >= 0 && type <= 2) {
            impl_type = (ImplType)type;
        } else {
            fprintf(stderr, "Invalid impl_type. Must be 0 (int8), 1 (fp32), or 2 (fp16)\n");
            return 1;
        }
    }

    if (strlen(argv[1]) != 64) {
        fprintf(stderr, "Error: seed must be 64 hex characters\n");
        return 1;
    }
    uint8_t seed[32];
    if (parse_hex(argv[1], strlen(argv[1]), seed, sizeof(seed)) != 0) {
        fprintf(stderr, "Error: invalid seed hex format\n");
        return 1;
    }
    
    if (strlen(argv[2]) != 64) {
        fprintf(stderr, "Error: input must be 64 hex characters\n");
        return 1;
    }
    uint8_t input[IN_SIZE];
    if (parse_hex(argv[2], strlen(argv[2]), input, sizeof(input)) != 0) {
        fprintf(stderr, "Error: invalid input hex format\n");
        return 1;
    }
    
    uint8_t output[IN_SIZE];
    tens_hash(input, seed, output, impl_type);
    
    for (int i = 0; i < IN_SIZE; i++)
        printf("%02x", output[IN_SIZE-1-i]);
    printf("\n");
    
    return 0;
}
#endif
