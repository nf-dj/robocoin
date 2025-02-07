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
    int8_t **middle_mats[ROUNDS];
    ImplType impl_type;
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;
} HashBuffers;

static void matrix_multiply_mod2_int8(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, int round) {
    fprintf(stderr,"sums: ");
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            int8_t w=A[i][j];
            //int8_t w=rand()%3-1;
            sum += w * in[j];
        }
        //sum += noise[i];
        //sum = noise[i];
        fprintf(stderr,"%d ",sum);
        if (sum>0) {
            out[i]=1;
        } else if (sum<0) {
            out[i]=0;
        } else {
            out[i]=noise[i];
        }
        //out[i] = sum & 1;
        //out[i] = (sum > 0) ? 1 : 0;
    }
    fprintf(stderr,"\n");
}

static void matrix_multiply_mod2_fp32(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, int round) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (float)A[i][j] * (float)in[j];
        }
        sum += (float)noise[i];
        out[i] = ((int)sum) & 1;
    }
}

static void matrix_multiply_mod2_fp16(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, int round) {
    for (int i = 0; i < rows; i++) {
        _Float16 sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (_Float16)A[i][j] * (_Float16)in[j];
        }
        sum += (_Float16)noise[i];
        out[i] = ((int)sum) & 1;
    }
}

static void matrix_multiply_mod2(int8_t **A, uint8_t *in, uint8_t *out, int8_t *noise, int rows, int cols, int round, ImplType impl_type) {
    switch(impl_type) {
        case IMPL_FP32:
            matrix_multiply_mod2_fp32(A, in, out, noise, rows, cols, round);
            break;
        case IMPL_FP16:
            matrix_multiply_mod2_fp16(A, in, out, noise, rows, cols, round);
            break;
        default:
            matrix_multiply_mod2_int8(A, in, out, noise, rows, cols, round);
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

// -----------------------------------------------------------------------------
// Helper: get_random_trits()
// Fills the given int8_t buffer with 'count' trits (values in {-1, 0, 1})
// using random bytes from /dev/urandom with rejection sampling.
static void get_random_trits(int8_t *buffer, size_t count) {
    size_t filled = 0;
    // Use a reasonable block size for batch reading.
    const size_t BLOCK_SIZE = 1024;
    uint8_t temp[BLOCK_SIZE];

    while (filled < count) {
        // Read up to BLOCK_SIZE bytes or whatever remains.
        size_t to_read = (count - filled < BLOCK_SIZE) ? (count - filled) : BLOCK_SIZE;
        if (fill_random_bytes(temp, to_read) != 0) {
            perror("Error reading random bytes");
            exit(EXIT_FAILURE);
        }
        // Process each byte in the temporary buffer.
        for (size_t i = 0; i < to_read && filled < count; i++) {
            // Reject the byte if it is 255 to avoid modulo bias.
            if (temp[i] == 255)
                continue;
            // Map the byte to a value in {0,1,2} then shift to {-1,0,1}
            buffer[filled++] = (temp[i] % 3) - 1;
        }
    }
}

static void get_random_trits_from_seed(int8_t *buffer, size_t count, uint8_t seed[32]) {
    size_t filled = 0;
    const size_t BLOCK_SIZE = 1024;
    uint8_t temp[BLOCK_SIZE];

    // Set up a base nonce of 32 bytes.
    // The first 4 bytes serve as a domain tag ("TRIT"), and the rest will be used for the counter.
    uint8_t base_nonce[32];
    memset(base_nonce, 0, sizeof(base_nonce));
    memcpy(base_nonce, "TRIT", 4);  // Domain separation for trits

    uint64_t block_counter = 0;

    while (filled < count) {
        uint8_t nonce[32];
        memcpy(nonce, base_nonce, sizeof(nonce));
        // Write the block_counter into nonce starting at offset 4.
        for (size_t i = 0; i < sizeof(block_counter) && (4 + i) < sizeof(nonce); i++) {
            nonce[4 + i] = (block_counter >> (8 * i)) & 0xff;
        }
        block_counter++;

        if (crypto_stream_chacha20(temp, BLOCK_SIZE, nonce, seed) != 0) {
            fprintf(stderr, "Error generating pseudorandom bytes from seed\n");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < BLOCK_SIZE && filled < count; i++) {
            if (temp[i] == 255)
                continue;
            buffer[filled++] = (temp[i] % 3) - 1;
        }
    }
}

//--------------------------------------------------------------------------
// (For illustration, here is an example of a similar function for bits that uses a different domain tag.)
// This function would generate random bits (0 or 1) from a seed.
// (You can use this if you need an independent bit stream.)
static void get_random_bits_from_seed(uint8_t *buffer, size_t count, uint8_t seed[32]) {
    size_t filled = 0;
    const size_t BLOCK_SIZE = 1024;
    uint8_t temp[BLOCK_SIZE];

    // Domain-separated nonce for bits: use "BIT_" as the tag.
    uint8_t base_nonce[32];
    memset(base_nonce, 0, sizeof(base_nonce));
    memcpy(base_nonce, "BIT_", 4);  // Domain separation for bits

    uint64_t block_counter = 0;

    while (filled < count) {
        uint8_t nonce[32];
        memcpy(nonce, base_nonce, sizeof(nonce));
        for (size_t i = 0; i < sizeof(block_counter) && (4 + i) < sizeof(nonce); i++) {
            nonce[4 + i] = (block_counter >> (8 * i)) & 0xff;
        }
        block_counter++;

        if (crypto_stream_chacha20(temp, BLOCK_SIZE, nonce, seed) != 0) {
            fprintf(stderr, "Error generating pseudorandom bytes for bits\n");
            exit(EXIT_FAILURE);
        }
        // Use each byte's LSB as a random bit.
        for (size_t i = 0; i < BLOCK_SIZE && filled < count; i++) {
            buffer[filled++] = temp[i] & 1;
        }
    }
}

void get_random_bits_from_trits_from_seed(uint8_t *bit_buffer, size_t count, uint8_t seed[32]) {
    // Allocate a temporary buffer for trits.
    int8_t *trit_buffer = malloc(count * sizeof(int8_t));
    if (trit_buffer == NULL) {
        perror("Memory allocation error for trits");
        exit(EXIT_FAILURE);
    }

    // Generate trits from the seed.
    get_random_trits_from_seed(trit_buffer, count, seed);

    // Apply the transformation to obtain bits.
    for (size_t i = 0; i < count; i++) {
        if ((i & 1) == 0) {  // Even index: only +1 gives 1.
            bit_buffer[i] = (trit_buffer[i] > 0) ? 1 : 0;
        } else {             // Odd index: 0 and +1 yield 1.
            bit_buffer[i] = (trit_buffer[i] >= 0) ? 1 : 0;
        }
    }

    free(trit_buffer);
}

// -----------------------------------------------------------------------------
// generate_matrices()
// Fills the provided matrices with random values from the set {-1, 0, 1}.
// The randomness comes from /dev/urandom. The 'seed' parameter is not used.
static void generate_matrices(int8_t **middle_mats[ROUNDS], uint8_t seed[32]) {
    (void)seed;  // The seed is unused since we draw from /dev/urandom.

    // Calculate total number of elements in all matrices.
    size_t total_elements = ROUNDS * HIDDEN * HIDDEN;
    // Allocate a temporary buffer to hold all the random trits.
    int8_t *rand_trits = malloc(total_elements * sizeof(int8_t));
    if (!rand_trits) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Fill the buffer with random trits in batch.
    get_random_trits_from_seed(rand_trits, total_elements, seed);
    //get_random_bits_from_seed(rand_trits, total_elements, seed);

    // Now fill the matrices with the random trits.
    size_t index = 0;
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < HIDDEN; j++) {
                middle_mats[r][i][j] = rand_trits[index++];
            }
        }
    }

    free(rand_trits);
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

    size_t noise_len = ROUNDS * HIDDEN;
    get_random_bits_from_seed((int8_t *)buffers->noise, noise_len, input);
    //get_random_trits_from_seed((int8_t *)buffers->noise, noise_len, input);
    //int8_t *noise2 = malloc(noise_len * sizeof(int8_t));
    //get_random_bits_from_seed((uint8_t *)noise2, noise_len, input);
    //for (int i = 0; i < noise_len; i++) {
    //    buffers->noise[i] += noise2[i];
   // }

    uint8_t seed[32];
    crypto_hash_sha256(seed, input, 32);

    // Convert byte input to bits
    for (int i = 0; i < IN_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            buffers->state[i*8 + j] = (seed[i] >> j) & 1;
        }
    }

    uint32_t round;
    for (round = 0; round < ROUNDS; round++) {
        fprintf(stderr,"round %d: ",round);
        for (int i = 0; i < HIDDEN; i++)
            fprintf(stderr,"%d", buffers->state[HIDDEN-1-i]);
        fprintf(stderr,"\n");
        matrix_multiply_mod2(matrices->middle_mats[round], buffers->state,
                           buffers->next_state, buffers->noise + (round * HIDDEN),
                           HIDDEN, HIDDEN, matrices->impl_type, round);
        uint8_t *temp = buffers->state;
        buffers->state = buffers->next_state;
        buffers->next_state = temp;
    }

    //get_random_bits_from_seed(buffers->state, HIDDEN, input); // XXX
    //get_random_bits_from_trits_from_seed(buffers->state, HIDDEN, input); // XXX

    fprintf(stderr,"round %d: ",round);
    for (int i = 0; i < HIDDEN; i++)
        fprintf(stderr,"%d", buffers->state[HIDDEN-1-i]);
    fprintf(stderr,"\n");

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
