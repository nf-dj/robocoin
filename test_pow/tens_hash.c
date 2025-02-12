#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sodium.h>
#include <openssl/sha.h>

// Constants from original implementation
#define N 256
#define DOT_THRESHOLD 5
#define ROUNDS 16

// Global debug flag; set to true if user passes "--debug"
static bool debug_mode = false;

// Forward declarations
static void print_row(int8_t *row, int row_num, int round, int attempts, int max_dot);

// Print a row for debugging (prints only if debug_mode is enabled)
static void print_row(int8_t *row, int row_num, int round, int attempts, int max_dot) {
    if (!debug_mode)
        return;
    fprintf(stderr, "R%d Row %d (attempt %d, max_dot %d): ", round, row_num, attempts, max_dot);
    int nonzero = 0;
    for (int i = 0; i < N; i++) {
        if (row[i] != 0) {
            fprintf(stderr, "%d@%d ", row[i], i);
            nonzero++;
        }
    }
    fprintf(stderr, "(%d nonzero)\n", nonzero);
}

// Helper to convert hex string to bytes
static bool hex_to_bytes(const char *hex, uint8_t *bytes, size_t len) {
    if (strlen(hex) != len * 2) return false;
    for (size_t i = 0; i < len; i++) {
        char byte_str[3] = {hex[i*2], hex[i*2+1], '\0'};
        char *endptr;
        bytes[i] = (uint8_t)strtol(byte_str, &endptr, 16);
        if (*endptr != '\0') return false;
    }
    return true;
}

// Generate a random ternary row using ChaCha20
static void generate_random_row(int8_t *row, const uint8_t *key, const uint8_t *nonce, uint64_t counter) {
    uint8_t rand_buf[N];
    memset(rand_buf, 0, N);  // Zero out the buffer so we get just the keystream
    crypto_stream_chacha20_xor_ic(rand_buf, rand_buf, N, nonce, counter, key);

    for (int j = 0; j < N; j++) {
        //uint8_t rand_val = rand_buf[j] & 0x1F;  // Take lower 5 bits
        uint8_t rand_val = rand_buf[j] & 0xF;  // Take lower 4 bits
        if (rand_val == 0)
            row[j] = 1;
        else if (rand_val == 1)
            row[j] = -1;
        else
            row[j] = 0;
    }
}

// Calculate dot product
static int32_t dot_product(int8_t *row1, int8_t *row2) {
    int32_t dot = 0;
    for (int i = 0; i < N; i++) {
        dot += row1[i] * row2[i];
    }
    return abs(dot);
}

// Generate matrix using seed for randomness
static bool generate_ternary_matrix(int8_t **M, const uint8_t *seed, uint64_t round) {
    uint8_t nonce[8] = {0};  // Use round number in nonce
    memcpy(nonce, &round, sizeof(round));
    
    // First row is random
    generate_random_row(M[0], seed, nonce, 0);
    print_row(M[0], 0, round, 1, 0);  // First row always succeeds in 1 attempt, max_dot 0
    
    // Generate subsequent rows
    for (int i = 1; i < N; i++) {
        int max_attempts = 1000;
        bool found_valid = false;
        int attempt = 0;
        int max_dot = 0;
        
        while (max_attempts-- > 0) {
            generate_random_row(M[i], seed, nonce, i * 1000 + attempt);
            bool valid = true;
            max_dot = 0;
            
            for (int j = 0; j < i; j++) {
                int32_t dp = dot_product(M[i], M[j]);
                if (dp > max_dot) max_dot = dp;
                if (dp > DOT_THRESHOLD) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                found_valid = true;
                print_row(M[i], i, round, attempt + 1, max_dot);
                break;
            }
            attempt++;
        }
        
        if (!found_valid) {
            fprintf(stderr, "Failed to generate row %d after 1000 attempts (max_dot %d)\n", i, max_dot);
            return false;
        }
    }
    return true;
}

static void bytes_to_binary(const uint8_t *bytes, uint8_t *binary) {
    for (int i = 0; i < N/8; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i*8 + j] = (bytes[i] >> (7-j)) & 1;
        }
    }
}

// Convert binary array to bytes
static void binary_to_bytes(const uint8_t *binary, uint8_t *bytes) {
    memset(bytes, 0, N/8);
    for (int i = 0; i < N; i++) {
        bytes[i/8] |= binary[i] << (7-(i%8));
    }
}

// Ternary transform function
static void ternary_transform(int8_t **M, uint8_t *in, uint8_t *out, uint8_t *noise) {
    for (int i = 0; i < N; i++) {
        int32_t dot = 0;
        for (int j = 0; j < N; j++) {
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

int main(int argc, char *argv[]) {
    // Allow usage: program <seed> <nonce> [--debug]
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <seed> <nonce> [--debug]\n", argv[0]);
        fprintf(stderr, "seed and nonce should be 32-byte hex strings\n");
        return 1;
    }
    
    // Set debug mode if the extra argument is provided
    if (argc == 4) {
        if (strcmp(argv[3], "--debug") == 0) {
            debug_mode = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[3]);
            return 1;
        }
    }
    
    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }
    
    // Convert hex inputs to bytes
    uint8_t seed[32], nonce[32];
    if (!hex_to_bytes(argv[1], seed, 32) || !hex_to_bytes(argv[2], nonce, 32)) {
        fprintf(stderr, "Invalid hex input\n");
        return 1;
    }
    
    // Initialize matrices
    int8_t ***matrices = malloc(ROUNDS * sizeof(int8_t **));
    for (int r = 0; r < ROUNDS; r++) {
        matrices[r] = malloc(N * sizeof(int8_t *));
        for (int i = 0; i < N; i++) {
            matrices[r][i] = malloc(N * sizeof(int8_t));
        }
    }
    
    // Generate matrices
    for (int r = 0; r < ROUNDS; r++) {
        if (!generate_ternary_matrix(matrices[r], seed, r)) {
            fprintf(stderr, "Failed to generate matrix for round %d\n", r);
            return 1;
        }
    }
    
    // Allocate working buffers
    uint8_t input[N], noise[N], *curr = malloc(N), *next = malloc(N);
    
    // Get input bits from SHA256(nonce)
    uint8_t input_hash[32];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, nonce, 32);
    SHA256_Final(input_hash, &sha256);
    bytes_to_binary(input_hash, input);
    
    // Get noise bits from SHA256(input_hash)
    uint8_t noise_hash[32];
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input_hash, 32);
    SHA256_Final(noise_hash, &sha256);
    bytes_to_binary(noise_hash, noise);
    
    // Apply rounds
    memcpy(curr, input, N);
    for (int r = 0; r < ROUNDS; r++) {
        ternary_transform(matrices[r], curr, next, noise);
        uint8_t *temp = curr;
        curr = next;
        next = temp;
    }
    
    // Convert final output to bytes and print
    uint8_t output[32];
    binary_to_bytes(curr, output);
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
    
    // Cleanup
    free(next);
    free(curr);
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < N; i++) {
            free(matrices[r][i]);
        }
        free(matrices[r]);
    }
    free(matrices);
    
    return 0;
}
