#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/sha.h>
#include <sodium.h>

// Constants
#define N 256                   // Matrix dimensions: 256 x 256
#define BLOCK_SIZE 32           // SHA256 produces 32 bytes (256 bits)
#define MAX_ATTEMPTS 1000
#define DOT_THRESHOLD 2
#define ROUNDS 16               // Fixed number of rounds

// Global array for row biases (for debugging)
static double row_biases[N];

// Forward declaration of dot_product (so it can be used in calculate_row_biases)
int32_t dot_product(int8_t *row1, int8_t *row2, int len);

// Calculate row biases after matrix generation
void calculate_row_biases(int8_t **M) {
    for (int i = 0; i < N; i++) {
        int32_t total_dot = 0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                total_dot += dot_product(M[i], M[j], N);
            }
        }
        row_biases[i] = (double)total_dot / (N - 1);
        if (i < 5)
            printf("Row %d bias: %.3f\n", i, row_biases[i]);
    }
}

static void generate_random_row(int8_t *row, const unsigned char *key, const unsigned char *nonce, uint64_t counter) {
    unsigned char rand_buf[N];
    memset(rand_buf, 0, N);
    crypto_stream_chacha20_xor_ic(rand_buf, rand_buf, N, nonce, counter, key);
    for (int j = 0; j < N; j++) {
        uint8_t rand_val = rand_buf[j] & 0x1F;
        if (rand_val == 0)
            row[j] = 1;
        else if (rand_val == 1)
            row[j] = -1;
        else
            row[j] = 0;
    }
}

bool generate_ternary_matrix(int8_t **M, const unsigned char *key, uint64_t round) {
    unsigned char nonce[8] = {0};
    memcpy(nonce, &round, sizeof(round));

    generate_random_row(M[0], key, nonce, 0);
    
    for (int i = 1; i < N; i++) {
        bool found_valid_row = false;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            uint64_t counter = (uint64_t)i * MAX_ATTEMPTS + attempt;
            generate_random_row(M[i], key, nonce, counter);
            bool valid = true;
            for (int j = 0; j < i; j++) {

                int32_t dot = dot_product(M[i], M[j], N);
                if (dot > DOT_THRESHOLD) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                found_valid_row = true;
                break;
            }
        }
        if (!found_valid_row) {
            printf("Failed to generate valid row %d after %d attempts\n", i, MAX_ATTEMPTS);
            return false;
        }
    }
    return true;
}

int32_t dot_product(int8_t *row1, int8_t *row2, int len) {
    int32_t dot = 0;
    for (int i = 0; i < len; i++) {
        dot += row1[i] * row2[i];
    }
    return abs(dot);
}

void bytes_to_binary(uint8_t *bytes, uint8_t *binary, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i * 8 + j] = (bytes[i] >> (7 - j)) & 1;
        }
    }
}

int count_leading_zeros(uint8_t *binary) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (binary[i] == 0)
            count++;
        else
            break;
    }
    return count;
}

static void ternary_transform(int8_t **M, uint8_t *in, uint8_t *out, int n, uint8_t *noise) {
    for (int i = 0; i < n; i++) {
        int32_t dot = 0;
        for (int j = 0; j < n; j++) {
            int val = in[j] ? 1 : -1;
            dot += M[i][j] * val;
        }
        if (dot > 0)
            out[i] = 1;
        else if (dot < 0)
            out[i] = 0;
        else
            out[i] = noise[i];
    }
}

void print_hex(uint8_t *data, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void binary_to_bytes(uint8_t *binary, uint8_t *bytes, int nbits) {
    memset(bytes, 0, (nbits + 7) / 8);
    for (int i = 0; i < nbits; i++) {
        if (binary[i])
            bytes[i / 8] |= (1 << (7 - (i % 8)));
    }
}

// Convert hex string to bytes
bool hex_to_bytes(const char *hex, unsigned char *bytes, size_t expected_len) {
    size_t hex_len = strlen(hex);
    if (hex_len != expected_len * 2) {
        return false;
    }
    
    for (size_t i = 0; i < expected_len; i++) {
        char hex_byte[3] = {hex[i * 2], hex[i * 2 + 1], 0};
        char *endptr;
        bytes[i] = (unsigned char)strtol(hex_byte, &endptr, 16);
        if (*endptr != '\0') {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <32-byte-hex-seed> <difficulty>\n", argv[0]);
        return 1;
    }

    // Parse seed (32 bytes in hex)
    unsigned char key[32];
    if (!hex_to_bytes(argv[1], key, 32)) {
        fprintf(stderr, "Error: Seed must be exactly 64 hex characters (32 bytes)\n");
        return 1;
    }

    // Parse difficulty
    int difficulty = atoi(argv[2]);
    if (difficulty <= 0 || difficulty > N) {
        fprintf(stderr, "Error: Difficulty must be between 1 and %d\n", N);
        return 1;
    }

    printf("Using seed: ");
    print_hex(key, 32);
    printf("Target difficulty: %d leading zeros\n", difficulty);

    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }

    int8_t ***matrices = malloc(ROUNDS * sizeof(int8_t **));
    if (!matrices) {
        perror("malloc");
        return 1;
    }
    for (int r = 0; r < ROUNDS; r++) {
        matrices[r] = malloc(N * sizeof(int8_t *));
        if (!matrices[r]) {
            perror("malloc");
            return 1;
        }
        for (int i = 0; i < N; i++) {
            matrices[r][i] = malloc(N * sizeof(int8_t));
            if (!matrices[r][i]) {
                perror("malloc");
                return 1;
            }
        }
        printf("Generating ternary matrix for round %d...\n", r);
        if (!generate_ternary_matrix(matrices[r], key, r)) {
            printf("Failed to generate matrix for round %d. Exiting.\n", r);
            return 1;
        }
        if (r == 0) {
            calculate_row_biases(matrices[r]);
        }
    }
    
    uint8_t hash[BLOCK_SIZE];
    uint8_t input[N];
    uint8_t noise_bits[N];
    uint8_t output[N];
    uint8_t output_bytes[BLOCK_SIZE];
    
    uint8_t *current_bits = malloc(N * sizeof(uint8_t));
    uint8_t *next_bits = malloc(N * sizeof(uint8_t));
    if (!current_bits || !next_bits) {
        perror("malloc");
        return 1;
    }
    
    SHA256_CTX sha256;
    uint64_t nonce = 0;
    time_t last_report = time(NULL);
    uint64_t hashes = 0;
    uint64_t total_hashes = 0;
    int best_zeros = 0;
    
    printf("Starting search for %d leading zeros...\n", difficulty);
    
    while (1) {
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, input, BLOCK_SIZE);
        memcpy(current_bits, input, N * sizeof(uint8_t));
        
        uint64_t next_nonce = nonce + 1;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &next_nonce, sizeof(next_nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, noise_bits, BLOCK_SIZE);
        
        for (int r = 0; r < ROUNDS; r++) {
            ternary_transform(matrices[r], current_bits, next_bits, N, noise_bits);
            uint8_t *temp = current_bits;
            current_bits = next_bits;
            next_bits = temp;
        }
        memcpy(output, current_bits, N * sizeof(uint8_t));
        
        int zeros = count_leading_zeros(output);
        if (zeros > best_zeros) {
            best_zeros = zeros;
            printf("\nNew best! Found %d leading zeros\n", zeros);
            printf("Nonce (hex): 0x%016lx\n", nonce);
            printf("Output (hex): ");
            binary_to_bytes(output, output_bytes, N);
            print_hex(output_bytes, BLOCK_SIZE);
        }
        
        if (zeros >= difficulty) {
            printf("\nSuccess! Found solution with %d leading zeros\n", zeros);
            printf("Nonce (hex): 0x%016lx\n", nonce);
            printf("Output (hex): ");
            binary_to_bytes(output, output_bytes, N);
            print_hex(output_bytes, BLOCK_SIZE);
            break;
        }
        
        time_t current = time(NULL);
        if (current > last_report) {
            printf("Progress: %lu hashes/s (%lu total), best leading zeros so far: %d\n",
                   hashes, total_hashes + hashes, best_zeros);
            last_report = current;
            total_hashes += hashes;
            hashes = 0;
        }
        
        nonce++;
        hashes++;
    }
    
    // Cleanup
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < N; i++) {
            free(matrices[r][i]);
        }
        free(matrices[r]);
    }
    free(matrices);
    free(current_bits);
    free(next_bits);
    
    return 0;
}
