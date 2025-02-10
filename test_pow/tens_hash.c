#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sodium.h>
#include <fcntl.h>
#include <unistd.h>

#define IN_SIZE 32
#define HIDDEN 256
//#define ROUNDS 64
#define ROUNDS 1

typedef enum {
    IMPL_INT8 = 0,
    IMPL_FP32 = 1,
    IMPL_FP16 = 2
} ImplType;

typedef struct {
    int8_t **matrices[ROUNDS];
    int8_t *biases[ROUNDS];
    ImplType impl_type;
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;
} HashBuffers;

static void matrix_multiply_relu_int8(int8_t **weights, int8_t *biases, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += weights[j][i] * in[j];
        }
        sum *= 2;
        sum += biases[i];
        //sum = 0; // XXX
        sum += noise[i]*8-4;
        //fprintf(stderr,"%d ",sum);
        out[i] = (sum > 0) ? 1 : 0;
    }
    //fprintf(stderr,"\n");
}

static void matrix_multiply_relu_fp32(int8_t **weights, int8_t *biases, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += (float)weights[j][i] * (float)in[j];
        }
		sum *= 2.0;
        sum += (float)biases[i];
        sum += (float)noise[i];
        out[i] = (sum > 0) ? 1.0 : 0.0;
    }
}

static void matrix_multiply_relu_fp16(int8_t **weights, int8_t *biases, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        _Float16 sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += (_Float16)weights[j][i] * (_Float16)in[j];
        }
		sum *= 2.0;
        sum += (_Float16)biases[i];
        sum += (_Float16)noise[i];
        out[i] = (sum > 0) ? 1.0 : 0.0;
    }
}

static void matrix_multiply_relu(int8_t **weights, int8_t *biases, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, ImplType impl_type) {
    switch(impl_type) {
        case IMPL_FP32:
            matrix_multiply_relu_fp32(weights, biases, in, out, noise, rows, cols);
            break;
        case IMPL_FP16:
            matrix_multiply_relu_fp16(weights, biases, in, out, noise, rows, cols);
            break;
        default:
            matrix_multiply_relu_int8(weights, biases, in, out, noise, rows, cols);
    }
}

int fill_random_bytes(uint8_t *buffer, size_t len) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        perror("Failed to open /dev/urandom");
        return -1;
    }

    size_t total = 0;
    while (total < len) {
        ssize_t nread = read(fd, buffer + total, len - total);
        if (nread <= 0) {
            perror("Failed to read /dev/urandom");
            close(fd);
            return -1;
        }
        total += nread;
    }
    close(fd);
    return 0;
}

typedef struct { 
    uint32_t val; 
    int idx; 
} sort_pair_t;

static int compare_pairs(const void *a, const void *b) {
    const sort_pair_t *pa = (const sort_pair_t *)a;
    const sort_pair_t *pb = (const sort_pair_t *)b;
    if (pa->val < pb->val) return -1;
    if (pa->val > pb->val) return 1;
    return 0;
}

static void verify_matrix(int8_t **matrix, int8_t *biases, int round) {
    // Verify row counts
    for (int i = 0; i < HIDDEN; i++) {
        int pos_count = 0;
        int neg_count = 0;
        for (int j = 0; j < HIDDEN; j++) {
            if (matrix[i][j] == 1) pos_count++;
            if (matrix[i][j] == -1) neg_count++;
        }
        /*if (pos_count != 32 || neg_count != 32) {
            fprintf(stderr, "Error in round %d, row %d: found %d +1s and %d -1s (expected 32 each)\n",
                    round, i, pos_count, neg_count);
            exit(1);
        }*/
    }

    // Verify column sums match negative biases
    for (int j = 0; j < HIDDEN; j++) {
        int32_t col_sum = 0;
        for (int i = 0; i < HIDDEN; i++) {
            col_sum += matrix[i][j];
        }
        if (col_sum != -biases[j]) {
            fprintf(stderr, "Error in round %d, col %d: sum=%d but bias=%d\n",
                    round, j, col_sum, -biases[j]);
            exit(1);
        }
    }
    
    //fprintf(stderr, "Round %d matrix verification passed\n", round);
}

void print_weights_and_bias(int8_t **matrix, int8_t *bias) {
    fprintf(stderr, "weights: [[");
    // Print first three rows fully
    for (int i = 0; i < 3; i++) {
        if (i > 0) fprintf(stderr, "\n [");
        for (int j = 0; j < HIDDEN; j++) {
            if (j < 3 || j >= HIDDEN-3) {
                fprintf(stderr, "%2d", matrix[i][j]);
                if (j < HIDDEN-1) fprintf(stderr, " ");
            }
            else if (j == 3) {
                fprintf(stderr, "...");
            }
        }
        fprintf(stderr, "]");
    }
    
    // Print ellipsis for middle rows
    fprintf(stderr, "\n ...");
    
    // Print last three rows
    for (int i = HIDDEN-3; i < HIDDEN; i++) {
        fprintf(stderr, "\n [");
        for (int j = 0; j < HIDDEN; j++) {
            if (j < 3 || j >= HIDDEN-3) {
                fprintf(stderr, "%2d", matrix[i][j]);
                if (j < HIDDEN-1) fprintf(stderr, " ");
            }
            else if (j == 3) {
                fprintf(stderr, "...");
            }
        }
        fprintf(stderr, "]");
    }
    fprintf(stderr, "]\n");
    
    fprintf(stderr, "bias: [");
    for (int i = 0; i < HIDDEN; i++) {
        fprintf(stderr, "%3d", bias[i]);
        if (i < HIDDEN-1) fprintf(stderr, " ");
    }
    fprintf(stderr, "]\n");
}

static void generate_matrices(int8_t **matrices[ROUNDS], int8_t *biases[ROUNDS], uint8_t seed[32]) {
    const int pos_count = 8;  // Number of +1s per row
    const int neg_count = 8;  // Number of -1s per row
    const int total_nonzero = pos_count + neg_count;
    uint8_t nonce[8];  // 4 bytes for round, 4 bytes zeros
    
    // Generate random values per round
    const size_t round_rand_vals = HIDDEN * HIDDEN;
    uint32_t *rand_vals = malloc(round_rand_vals * sizeof(uint32_t));

    // Pre-generate the sign array
    int8_t base_signs[total_nonzero];
    for (int j = 0; j < pos_count; j++) base_signs[j] = 1;
    for (int j = pos_count; j < total_nonzero; j++) base_signs[j] = -1;

    typedef struct { uint32_t val; int idx; } sort_pair_t;
    sort_pair_t pairs[HIDDEN];

    for (int r = 0; r < ROUNDS; r++) {
        // Initialize column sums to zero
        // Create nonce for this round
    memset(nonce, 0, sizeof(nonce));
    nonce[3] = r & 0xFF;  // Little-endian round number
    nonce[2] = (r >> 8) & 0xFF;
    nonce[1] = (r >> 16) & 0xFF;
    nonce[0] = (r >> 24) & 0xFF;

    // Generate random values for this round
    crypto_stream_chacha20((uint8_t*)rand_vals, round_rand_vals * sizeof(uint32_t), nonce, seed);

    int32_t col_sums[HIDDEN] = {0};
        
        for (int i = 0; i < HIDDEN; i++) {
            // Clear the row
            memset(matrices[r][i], 0, HIDDEN * sizeof(int8_t));
            
            // Point to the random values for this row
            uint32_t *row_rand_vals = rand_vals + (i * HIDDEN);
            
            // Setup pairs for sorting
            for (int j = 0; j < HIDDEN; j++) {
                pairs[j].val = row_rand_vals[j];
                pairs[j].idx = j;
            }
            
            // Sort pairs by value
            qsort(pairs, HIDDEN, sizeof(sort_pair_t), compare_pairs);
            
            // Place signs at the sorted positions and update column sums
            for (int j = 0; j < total_nonzero; j++) {
                int8_t sign = base_signs[j];
                int col = pairs[j].idx;
                matrices[r][i][col] = sign;
                col_sums[col] += sign;
            }
        }
        
        // Set biases as negative column sums
        for (int i = 0; i < HIDDEN; i++) {
            biases[r][i] = -col_sums[i];
        }

        // Verify the matrix and biases for this round
        verify_matrix(matrices[r], biases[r], r);
    }

    print_weights_and_bias(matrices[0], biases[0]);
    //print_weights_and_bias(matrices[63], biases[63]);
    
    free(rand_vals);
}

PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type) {
    PrecomputedMatrices* matrices = malloc(sizeof(PrecomputedMatrices));
    if (!matrices) return NULL;

    matrices->impl_type = impl_type;

    for (int r = 0; r < ROUNDS; r++) {
        matrices->matrices[r] = malloc(HIDDEN * sizeof(int8_t*));
        if (!matrices->matrices[r]) {
            exit(1);
        }
        matrices->matrices[r][0] = malloc(HIDDEN * HIDDEN * sizeof(int8_t));
        if (!matrices->matrices[r][0]) {
            exit(1);
        }
        for (int i = 1; i < HIDDEN; i++) {
            matrices->matrices[r][i] = matrices->matrices[r][0] + (i * HIDDEN);
        }
        matrices->biases[r] = malloc(HIDDEN * sizeof(int8_t*));
        if (!matrices->biases[r]) {
            exit(1);
        }
    }

    generate_matrices(matrices->matrices, matrices->biases, seed);
    return matrices;
}

HashBuffers* init_hash_buffers(void) {
    HashBuffers* buffers = malloc(sizeof(HashBuffers));
    if (!buffers) return NULL;

    buffers->state = calloc(HIDDEN, sizeof(uint8_t));
    buffers->next_state = calloc(HIDDEN, sizeof(uint8_t));
    buffers->noise = malloc(HIDDEN * sizeof(int8_t));

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
    for (int r = 0; r < ROUNDS; r++) {
        free(matrices->matrices[r][0]);
        free(matrices->matrices[r]);
        free(matrices->biases[r]);
    }
    free(matrices);
}

void free_hash_buffers(HashBuffers* buffers) {
    if (buffers) {
        free(buffers->state);
        free(buffers->next_state);
        free(buffers->noise);
        free(buffers);
    }
}

void compute_binary_and_noise_vectors(const uint8_t *input, uint8_t *binary_out, int8_t *noise_out) {
    unsigned char first_hash[crypto_hash_sha256_BYTES];
    unsigned char second_hash[crypto_hash_sha256_BYTES];
    
    // First SHA256 for binary vector
    crypto_hash_sha256(first_hash, input, IN_SIZE);
    
    // Convert first hash to binary vector
    for (int i = 0; i < HIDDEN; i++) {
        binary_out[i] = (uint8_t)((first_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
    
    // Second SHA256 for noise
    crypto_hash_sha256(second_hash, first_hash, IN_SIZE);
    
    // Convert second hash to noise vector - now using same method as binary vector
    for (int i = 0; i < HIDDEN; i++) {
        noise_out[i] = (int8_t)((second_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void bits_to_bytes_msb(const uint8_t *bits, uint8_t *bytes, size_t bit_len) {
    memset(bytes, 0, (bit_len + 7) / 8);
    for (size_t i = 0; i < bit_len; i++) {
        if (bits[i]) {
            bytes[i / 8] |= 0x80 >> (i % 8);  // MSB first
        }
    }
}

void print_bit_array(const char* label, const uint8_t* bits, size_t len) {
    fprintf(stderr, "%s: [", label);
    for (size_t i = 0; i < len; i++) {
        fprintf(stderr, "%d", bits[i]);
        if (i < len-1) fprintf(stderr, " ");
    }
    fprintf(stderr, "]\n");
}

void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices,
                          HashBuffers* buffers, uint8_t output[IN_SIZE]) {

    fprintf(stderr,"input: ");
    for (int i = 0; i < IN_SIZE; i++)
        fprintf(stderr,"%02x", input[i]);
    fprintf(stderr,"\n");

	compute_binary_and_noise_vectors(input, buffers->state, buffers->noise);
    print_bit_array("input", buffers->state, HIDDEN);
    print_bit_array("noise", buffers->noise, HIDDEN);

    uint32_t round;
    for (round = 0; round < ROUNDS; round++) {
        matrix_multiply_relu(matrices->matrices[round], matrices->biases[round], buffers->state,
                           buffers->next_state, buffers->noise,
                           HIDDEN, HIDDEN, matrices->impl_type);
        uint8_t *temp = buffers->state;
        buffers->state = buffers->next_state;
        buffers->next_state = temp;
    }

    print_bit_array("output", buffers->state, HIDDEN);

    bits_to_bytes_msb(buffers->state, output, HIDDEN);
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
        int hi = hexchar_to_int(hex[2 * i]);
        int lo = hexchar_to_int(hex[2 * i + 1]);
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
        printf("%02x", output[i]);
    printf("\n");

    return 0;
}
#endif
