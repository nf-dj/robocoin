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
#define ROUNDS 1

typedef enum {
    IMPL_INT8 = 0,
    IMPL_FP32 = 1,
    IMPL_FP16 = 2
} ImplType;

typedef struct {
    int8_t **matrices[ROUNDS];
    int8_t *biases[ROUNDS];  // (Not used in the new transform)
    ImplType impl_type;
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;
} HashBuffers;

/* ---------------------------------------------------------------------
   Binary Hadamard transform (binary vector → binary vector)
   This function implements the transform as follows:
     1. Convert the input binary vector to a "signed" vector: 0 → -1, 1 → +1.
     2. For each row i, compute the dot product:
             dot = sum_{j=0}^{HIDDEN-1} H[i][j] * (in[j] ? 1 : -1)
     3. Then set:
             if (dot > 0)      out[i] = 1;
             if (dot < 0)      out[i] = 0;
             if (dot == 0)     out[i] = noise[i];   // use noise parameter
   --------------------------------------------------------------------- */
static void binary_hadamard_transform(int8_t **H, uint8_t *in, uint8_t *out, int n, int8_t *noise) {
    for (int i = 0; i < n; i++) {
        int32_t dot = 0;
        for (int j = 0; j < n; j++) {
            int val = in[j] ? 1 : -1;
            dot += H[i][j] * val;
        }
        /*
        if (dot > 0)
            out[i] = 1;
        else if (dot < 0)
            out[i] = 0;
        else
            out[i] = noise[i];  // Use noise vector for tie resolution.
        */
        dot+=noise[i];
        out[i] = dot > 0;
    }
}

/* ---------------------------------------------------------------------
   The following functions remain unchanged:
   - fill_random_bytes
   - generate_sylvester_hadamard (recursive Sylvester construction)
   - generate_permuted_hadamard_matrix (using ChaCha20 from libsodium)
   --------------------------------------------------------------------- */

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

/* Recursive Sylvester construction: H(1) = [1], H(2n) = [ H(n)  H(n); H(n) -H(n) ] */
static void generate_sylvester_hadamard(int8_t *H, int n) {
    if (n == 1) {
        H[0] = 1;
        return;
    }
    int half = n / 2;
    generate_sylvester_hadamard(H, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            int base = i * half + j;
            int idx_tl = i * n + j;
            int idx_tr = i * n + (j + half);
            int idx_bl = (i + half) * n + j;
            int idx_br = (i + half) * n + (j + half);
            H[idx_tl] = H[base];
            H[idx_tr] = H[base];
            H[idx_bl] = H[base];
            H[idx_br] = -H[base];
        }
    }
}

/* Helper structure and comparator for generating permutations */
typedef struct {
    uint64_t val;
    int idx;
} sort_pair_t;

static int compare_pairs(const void *a, const void *b) {
    const sort_pair_t *pa = (const sort_pair_t *)a;
    const sort_pair_t *pb = (const sort_pair_t *)b;
    if (pa->val < pb->val) return -1;
    if (pa->val > pb->val) return 1;
    return 0;
}

/* Generate a permuted Hadamard matrix using ChaCha20.
   The Sylvester Hadamard matrix of size n x n is generated and then
   its rows and columns are permuted according to random values generated
   by ChaCha20 (with a fixed 8-byte nonce and a 32-byte seed).
*/
static int8_t *generate_permuted_hadamard_matrix(uint8_t seed[32], int n) {
    int size = n * n;
    int8_t *H = malloc(size * sizeof(int8_t));
    if (!H) {
        fprintf(stderr, "Failed to allocate Hadamard matrix\n");
        exit(1);
    }
    generate_sylvester_hadamard(H, n);

    uint8_t nonce[8] = {0};  // Fixed nonce: 8 bytes of 0
    sort_pair_t *row_pairs = malloc(n * sizeof(sort_pair_t));
    sort_pair_t *col_pairs = malloc(n * sizeof(sort_pair_t));
    if (!row_pairs || !col_pairs) {
        fprintf(stderr, "Failed to allocate permutation pairs\n");
        exit(1);
    }
    uint64_t *rand_vals = malloc(n * sizeof(uint64_t));
    if (!rand_vals) {
        fprintf(stderr, "Failed to allocate random values\n");
        exit(1);
    }

    // Generate row permutation
    crypto_stream_chacha20((uint8_t*)rand_vals, n * sizeof(uint64_t), nonce, seed);
    for (int i = 0; i < n; i++) {
        row_pairs[i].val = rand_vals[i];
        row_pairs[i].idx = i;
    }
    qsort(row_pairs, n, sizeof(sort_pair_t), compare_pairs);

    // Generate column permutation (reuse rand_vals with a second call)
    crypto_stream_chacha20((uint8_t*)rand_vals, n * sizeof(uint64_t), nonce, seed);
    for (int i = 0; i < n; i++) {
        col_pairs[i].val = rand_vals[i];
        col_pairs[i].idx = i;
    }
    qsort(col_pairs, n, sizeof(sort_pair_t), compare_pairs);

    int8_t *P = malloc(size * sizeof(int8_t));
    if (!P) {
        fprintf(stderr, "Failed to allocate permuted matrix\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        int orig_row = row_pairs[i].idx;
        for (int j = 0; j < n; j++) {
            int orig_col = col_pairs[j].idx;
            P[i * n + j] = H[orig_row * n + orig_col];
        }
    }

    free(H);
    free(rand_vals);
    free(row_pairs);
    free(col_pairs);
    return P;
}

/* Updated precomputation: generate one permuted Hadamard matrix and copy it into all rounds.
   The biases are allocated (but not used in the new binary transform).
*/
PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type) {
    PrecomputedMatrices* matrices = malloc(sizeof(PrecomputedMatrices));
    if (!matrices) return NULL;
    matrices->impl_type = impl_type;

    int8_t *perm_hadamard = generate_permuted_hadamard_matrix(seed, HIDDEN);

    for (int r = 0; r < ROUNDS; r++) {
        matrices->matrices[r] = malloc(HIDDEN * sizeof(int8_t*));
        if (!matrices->matrices[r]) {
            exit(1);
        }
        matrices->matrices[r][0] = malloc(HIDDEN * HIDDEN * sizeof(int8_t));
        if (!matrices->matrices[r][0]) {
            exit(1);
        }
        memcpy(matrices->matrices[r][0], perm_hadamard, HIDDEN * HIDDEN * sizeof(int8_t));
        for (int i = 1; i < HIDDEN; i++) {
            matrices->matrices[r][i] = matrices->matrices[r][0] + (i * HIDDEN);
        }
        matrices->biases[r] = malloc(HIDDEN * sizeof(int8_t));
        if (!matrices->biases[r]) {
            exit(1);
        }
        memset(matrices->biases[r], 0, HIDDEN * sizeof(int8_t));
    }
    free(perm_hadamard);
    return matrices;
}

/* ---------------------------------------------------------------------
   The following functions remain unchanged:
      - verify_matrix
      - print_weights_and_bias
      - init_hash_buffers
      - free_matrices / free_hash_buffers
      - compute_binary_and_noise_vectors
      - bits_to_bytes_msb
      - print_bit_array
   --------------------------------------------------------------------- */

void verify_matrix(int8_t **matrix, int8_t *biases, int round) {
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
}

void print_weights_and_bias(int8_t **matrix, int8_t *bias) {
    fprintf(stderr, "weights: [[");
    for (int i = 0; i < 3; i++) {
        if (i > 0) fprintf(stderr, "\n [");
        for (int j = 0; j < HIDDEN; j++) {
            if (j < 3 || j >= HIDDEN-3) {
                fprintf(stderr, "%2d", matrix[i][j]);
                if (j < HIDDEN-1) fprintf(stderr, " ");
            } else if (j == 3) {
                fprintf(stderr, "...");
            }
        }
        fprintf(stderr, "]");
    }
    fprintf(stderr, "\n ...");
    for (int i = HIDDEN-3; i < HIDDEN; i++) {
        fprintf(stderr, "\n [");
        for (int j = 0; j < HIDDEN; j++) {
            if (j < 3 || j >= HIDDEN-3) {
                fprintf(stderr, "%2d", matrix[i][j]);
                if (j < HIDDEN-1) fprintf(stderr, " ");
            } else if (j == 3) {
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
    crypto_hash_sha256(first_hash, input, IN_SIZE);
    for (int i = 0; i < HIDDEN; i++) {
        binary_out[i] = (uint8_t)((first_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
    crypto_hash_sha256(second_hash, first_hash, IN_SIZE);
    for (int i = 0; i < HIDDEN; i++) {
        noise_out[i] = (int8_t)((second_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void bits_to_bytes_msb(const uint8_t *bits, uint8_t *bytes, size_t bit_len) {
    memset(bytes, 0, (bit_len + 7) / 8);
    for (size_t i = 0; i < bit_len; i++) {
        if (bits[i]) {
            bytes[i / 8] |= 0x80 >> (i % 8);
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

/* ---------------------------------------------------------------------
   Updated tens_hash_precomputed:
   The multi-round transform now calls binary_hadamard_transform with the noise vector.
   The same precomputed (permuted Hadamard) matrix is used in every round.
   --------------------------------------------------------------------- */
void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices,
                          HashBuffers* buffers, uint8_t output[IN_SIZE]) {

    fprintf(stderr,"input: ");
    for (int i = 0; i < IN_SIZE; i++)
        fprintf(stderr,"%02x", input[i]);
    fprintf(stderr,"\n");

    compute_binary_and_noise_vectors(input, buffers->state, buffers->noise);
    print_bit_array("input", buffers->state, HIDDEN);
    print_bit_array("noise", buffers->noise, HIDDEN);

    for (uint32_t round = 0; round < ROUNDS; round++) {
        binary_hadamard_transform(matrices->matrices[round], buffers->state,
                                    buffers->next_state, HIDDEN, buffers->noise);
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

    ImplType impl_type = IMPL_INT8;
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

