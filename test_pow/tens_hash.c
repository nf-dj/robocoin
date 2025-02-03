#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sodium.h>
#include <ctype.h>  // for isxdigit()

// -----------------------------------------------------------------------------
// Definitions
// -----------------------------------------------------------------------------
#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64

// Add implementation type enum
typedef enum {
    IMPL_INT8 = 0,
    IMPL_FP32 = 1
} ImplType;

// Update structures to include implementation type
typedef struct {
    uint8_t **expand_mat;         // HIDDEN x IN_SIZE
    uint8_t **middle_mats[ROUNDS]; // ROUNDS of HIDDEN x HIDDEN
    uint8_t **compress_mat;       // IN_SIZE x HIDDEN
    ImplType impl_type;           // Selected implementation type
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;  // noise buffer of size: HIDDEN + (ROUNDS * HIDDEN) + IN_SIZE
} HashBuffers;

// -----------------------------------------------------------------------------
// Function Prototypes - Added all necessary declarations
// -----------------------------------------------------------------------------
PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type);
void free_matrices(PrecomputedMatrices* matrices);
HashBuffers* init_hash_buffers(void);
void free_hash_buffers(HashBuffers* buffers);
void tens_hash_precomputed(uint8_t input[IN_SIZE],
                          PrecomputedMatrices* matrices,
                          HashBuffers* buffers,
                          uint8_t output[IN_SIZE]);
void tens_hash(uint8_t input[IN_SIZE],
               uint8_t seed[32],
               uint8_t output[IN_SIZE],
               ImplType impl_type);
int parse_hex(const char *hex, size_t hex_len, uint8_t *out, size_t out_len);
static void generate_matrices(uint8_t **expand_mat,
                            uint8_t **middle_mats[ROUNDS],
                            uint8_t **compress_mat,
                            uint8_t seed[32]);

// -----------------------------------------------------------------------------
// Matrix multiplication implementations
// -----------------------------------------------------------------------------
static void matrix_multiply_int8(int8_t **A, int8_t *in, int8_t *out, int8_t *e, int rows, int cols) {
	//printf("mm_int8\n");
    for (int i = 0; i < rows; i++) {
        //int32_t sum = 0;
        int8_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += A[i][j] * in[j];
        }
        sum += e[i];
        out[i] = sum;
    }
    //for (int i = 0; i < rows; i++)
    //    printf("%02x", (uint8_t)out[i]);
    //printf(" int8\n");
}

static void matrix_multiply_fp32(int8_t **A, int8_t *in, int8_t *out, int8_t *e, int rows, int cols) {
	//printf("mm_fp32\n");
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (float)A[i][j] * (float)in[j];
        }
        sum += (float)e[i];
        out[i] = (int8_t)sum;
    }
    //for (int i = 0; i < rows; i++)
    //    printf("%02x", (uint8_t)out[i]);
    //printf(" fp32\n");
}

// Updated matrix multiply dispatcher
static void matrix_multiply(uint8_t **A, uint8_t *in, uint8_t *out, int8_t *e, int rows, int cols, ImplType impl_type) {
    if (impl_type == IMPL_FP32) {
        matrix_multiply_fp32(A, in, out, e, rows, cols);
    } else {
        matrix_multiply_int8(A, in, out, e, rows, cols);
    }
}

// -----------------------------------------------------------------------------
// Matrix Generation (using the original generate_matrices implementation)
// -----------------------------------------------------------------------------
static void generate_matrices(uint8_t **expand_mat,
                            uint8_t **middle_mats[ROUNDS],
                            uint8_t **compress_mat,
                            uint8_t seed[32]) {
    printf("Using seed: ");
    for (int i = 0; i < 32; i++) printf("%02x", seed[i]);    
    printf("\n");
    size_t total_size = (HIDDEN * IN_SIZE) + (ROUNDS * HIDDEN * HIDDEN) + (IN_SIZE * HIDDEN);
    uint8_t *data = malloc(total_size);
    if (!data) {
        fprintf(stderr, "Memory allocation error in generate_matrices\n");
        exit(1);
    }
    unsigned char nonce[crypto_stream_chacha20_NONCEBYTES] = {0}; // fixed nonce
    crypto_stream_chacha20(data, total_size, nonce, seed);

    uint8_t *pos = data;
    memcpy(expand_mat[0], pos, HIDDEN * IN_SIZE);
    printf("Expand matrix (first 8 values):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%u ", (unsigned int)pos[i * IN_SIZE + j]);
        }
    }
    printf("\n");
    pos += HIDDEN * IN_SIZE;

    for (int r = 0; r < ROUNDS; r++) {
        memcpy(middle_mats[r][0], pos, HIDDEN * HIDDEN);
        if (r == 0) {
            printf("First middle matrix (first 8 values):\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 2; j++) {
                    printf("%u ", (unsigned int)pos[i * HIDDEN + j]);
                }
            }
            printf("\n");
        }
        pos += HIDDEN * HIDDEN;
    }

    memcpy(compress_mat[0], pos, IN_SIZE * HIDDEN);
    printf("Compress matrix (first 8 values):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%u ", (unsigned int)pos[i * HIDDEN + j]);
        }
    }
    printf("\n");
    free(data);
}

// -----------------------------------------------------------------------------
// Precompute Matrices Implementation - Fixed to use seed parameter
// -----------------------------------------------------------------------------
PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type) {
    PrecomputedMatrices* matrices = malloc(sizeof(PrecomputedMatrices));
    if (!matrices) return NULL;
    
    matrices->impl_type = impl_type;

    // Allocate expansion matrix
    matrices->expand_mat = malloc(HIDDEN * sizeof(uint8_t*));
    if (!matrices->expand_mat) { free(matrices); return NULL; }
    matrices->expand_mat[0] = malloc(HIDDEN * IN_SIZE);
    if (!matrices->expand_mat[0]) { free(matrices->expand_mat); free(matrices); return NULL; }
    for (int i = 1; i < HIDDEN; i++) {
        matrices->expand_mat[i] = matrices->expand_mat[0] + (i * IN_SIZE);
    }

    // Allocate middle matrices
    for (int r = 0; r < ROUNDS; r++) {
        matrices->middle_mats[r] = malloc(HIDDEN * sizeof(uint8_t*));
        if (!matrices->middle_mats[r]) {
            for (int j = 0; j < r; j++) {
                free(matrices->middle_mats[j][0]);
                free(matrices->middle_mats[j]);
            }
            free(matrices->expand_mat[0]);
            free(matrices->expand_mat);
            free(matrices);
            return NULL;
        }
        matrices->middle_mats[r][0] = malloc(HIDDEN * HIDDEN);
        if (!matrices->middle_mats[r][0]) {
            for (int j = 0; j < r; j++) {
                free(matrices->middle_mats[j][0]);
                free(matrices->middle_mats[j]);
            }
            free(matrices->middle_mats[r]);
            free(matrices->expand_mat[0]);
            free(matrices->expand_mat);
            free(matrices);
            return NULL;
        }
        for (int i = 1; i < HIDDEN; i++) {
            matrices->middle_mats[r][i] = matrices->middle_mats[r][0] + (i * HIDDEN);
        }
    }

    // Allocate compression matrix
    matrices->compress_mat = malloc(IN_SIZE * sizeof(uint8_t*));
    if (!matrices->compress_mat) {
        for (int r = 0; r < ROUNDS; r++) {
            free(matrices->middle_mats[r][0]);
            free(matrices->middle_mats[r]);
        }
        free(matrices->expand_mat[0]);
        free(matrices->expand_mat);
        free(matrices);
        return NULL;
    }
    matrices->compress_mat[0] = malloc(IN_SIZE * HIDDEN);
    if (!matrices->compress_mat[0]) {
        for (int r = 0; r < ROUNDS; r++) {
            free(matrices->middle_mats[r][0]);
            free(matrices->middle_mats[r]);
        }
        free(matrices->compress_mat);
        free(matrices->expand_mat[0]);
        free(matrices->expand_mat);
        free(matrices);
        return NULL;
    }
    for (int i = 1; i < IN_SIZE; i++) {
        matrices->compress_mat[i] = matrices->compress_mat[0] + (i * HIDDEN);
    }

    // Generate matrices using the seed
    generate_matrices(matrices->expand_mat, matrices->middle_mats, matrices->compress_mat, seed);
    
    return matrices;
}

// -----------------------------------------------------------------------------
// Memory Management Functions
// -----------------------------------------------------------------------------
void free_matrices(PrecomputedMatrices* matrices) {
    if (matrices) {
        if (matrices->expand_mat) {
            free(matrices->expand_mat[0]);
            free(matrices->expand_mat);
        }
        for (int r = 0; r < ROUNDS; r++) {
            if (matrices->middle_mats[r]) {
                free(matrices->middle_mats[r][0]);
                free(matrices->middle_mats[r]);
            }
        }
        if (matrices->compress_mat) {
            free(matrices->compress_mat[0]);
            free(matrices->compress_mat);
        }
        free(matrices);
    }
}

HashBuffers* init_hash_buffers(void) {
    HashBuffers* buffers = malloc(sizeof(HashBuffers));
    if (!buffers) return NULL;
    
    buffers->state = calloc(HIDDEN, sizeof(uint8_t));
    buffers->next_state = calloc(HIDDEN, sizeof(uint8_t));
    int total_noise = HIDDEN + (ROUNDS * HIDDEN) + IN_SIZE;
    buffers->noise = malloc(total_noise * sizeof(int8_t));
    
    if (!buffers->state || !buffers->next_state || !buffers->noise) {
        if (buffers->state) free(buffers->state);
        if (buffers->next_state) free(buffers->next_state);
        if (buffers->noise) free(buffers->noise);
        free(buffers);
        return NULL;
    }
    
    return buffers;
}

void free_hash_buffers(HashBuffers* buffers) {
    if (buffers) {
        free(buffers->state);
        free(buffers->next_state);
        free(buffers->noise);
        free(buffers);
    }
}

// -----------------------------------------------------------------------------
// Hash Implementation
// -----------------------------------------------------------------------------
void tens_hash_precomputed(uint8_t input[IN_SIZE],
                          PrecomputedMatrices* matrices,
                          HashBuffers* buffers,
                          uint8_t output[IN_SIZE]) {
    int total_noise = HIDDEN + (ROUNDS * HIDDEN) + IN_SIZE;
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256(digest, input, IN_SIZE);
    for (int i = 0; i < total_noise; i++) {
        buffers->noise[i] = digest[i % crypto_hash_sha256_BYTES];
    }

    int8_t *expand_noise = buffers->noise;
    int8_t *middle_noise = buffers->noise + HIDDEN;
    int8_t *compress_noise = buffers->noise + HIDDEN + (ROUNDS * HIDDEN);

    matrix_multiply(matrices->expand_mat, input, buffers->state, expand_noise, 
                   HIDDEN, IN_SIZE, matrices->impl_type);

    for (uint32_t round = 0; round < ROUNDS; round++) {
        matrix_multiply(matrices->middle_mats[round], buffers->state, buffers->next_state, 
                       middle_noise + (round * HIDDEN), HIDDEN, HIDDEN, matrices->impl_type);
        uint8_t *temp = buffers->state;
        buffers->state = buffers->next_state;
        buffers->next_state = temp;
    }

    matrix_multiply(matrices->compress_mat, buffers->state, output, compress_noise, 
                   IN_SIZE, HIDDEN, matrices->impl_type);
}

void tens_hash(uint8_t input[IN_SIZE],
              uint8_t seed[32],
              uint8_t output[IN_SIZE],
              ImplType impl_type) {
    PrecomputedMatrices* matrices = precompute_matrices(seed, impl_type);
    if (!matrices) {
        fprintf(stderr, "Failed to precompute matrices.\n");
        exit(1);
    }
    HashBuffers* buffers = init_hash_buffers();
    if (!buffers) {
        free_matrices(matrices);
        fprintf(stderr, "Failed to initialize hash buffers.\n");
        exit(1);
    }
    tens_hash_precomputed(input, matrices, buffers, output);
    free_matrices(matrices);
    free_hash_buffers(buffers);
}

// -----------------------------------------------------------------------------
// Hex Parsing Utilities
// -----------------------------------------------------------------------------
int hexchar_to_int(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

int parse_hex(const char *hex, size_t hex_len, uint8_t *out, size_t out_len) {
    if (hex_len != out_len * 2)
        return -1;
    for (size_t i = 0; i < out_len; i++) {
        int hi = hexchar_to_int(hex[2 * i]);
        int lo = hexchar_to_int(hex[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return -1;
        out[i] = (hi << 4) | lo;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
#ifdef HASH_MAIN
int main(int argc, char *argv[]) {
    if (sodium_init() < 0) {
        fprintf(stderr, "Error: libsodium initialization failed\n");
        return 1;
    }
    
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <seed_hex> <input_hex> [impl_type]\n", argv[0]);
        fprintf(stderr, "impl_type: 0 for integer, 1 for floating-point (default: 0)\n");
        return 1;
    }
    
    ImplType impl_type = IMPL_INT8;  // Default to integer implementation
    if (argc >= 4) {
        impl_type = atoi(argv[3]) ? IMPL_FP32 : IMPL_INT8;
    }
    
    if (strlen(argv[1]) != 64) {
        fprintf(stderr, "Error: seed must be 64 hex characters\n");
        return 1;
    }
    uint8_t seed[32];
    if (parse_hex(argv[1], strlen(argv[1]), seed, sizeof(seed)) != 0) {
        fprintf(stderr, "Error: invalid seed hex format");
        return 1;
    }
    
    if (strlen(argv[2]) != 64) {
        fprintf(stderr, "Error: input must be 64 hex characters");
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
        printf("%02x", output[i]);
    printf("\n");
    
    return 0;
}
#endif

