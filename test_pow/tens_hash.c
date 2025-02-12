#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sodium.h>
#include <openssl/sha.h>

#define N 256
#define DOT_THRESHOLD 5
#define ROUNDS 16

// Global debug flag; set to true if user passes "--debug"
static bool debug_mode = false;

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

// Convert a hex string into bytes.
static bool hex_to_bytes(const char *hex, uint8_t *bytes, size_t len) {
    if (strlen(hex) != len * 2)
        return false;
    for (size_t i = 0; i < len; i++) {
        char byte_str[3] = {hex[i*2], hex[i*2+1], '\0'};
        char *endptr;
        bytes[i] = (uint8_t)strtol(byte_str, &endptr, 16);
        if (*endptr != '\0')
            return false;
    }
    return true;
}

// Print a byte array as hex.
void print_hex(const uint8_t *buffer, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%02X", buffer[i]);
    }
    printf("\n");
}

/*
 * Generate a random ternary row using ChaCha20.
 * - key: the 32-byte seed.
 * - nonce: an 8-byte nonce (constructed from the round number).
 * - offset: pointer to a 64-bit counter (in blocks, not bytes).
 *
 * For each call, we request nbytes = N/2 (i.e. 128) bytes from the ChaCha20 stream.
 * Since each ChaCha20 block is 64 bytes, we must increment *offset by (nbytes / 64)
 * so that the counter advances correctly.
 */
static void generate_random_row(int8_t *row, const uint8_t *key, const uint8_t *nonce, uint64_t *offset) {
    const size_t nbytes = N / 2;  // 128 bytes to get 256 nibbles
    uint8_t rand_buf[nbytes];
    memset(rand_buf, 0, nbytes);  // prepare a zeroed buffer
    crypto_stream_chacha20_xor_ic(rand_buf, rand_buf, nbytes, nonce, *offset, key);
    //print_hex(rand_buf, sizeof(rand_buf));
    // FIXED: increment the counter by the number of 64-byte blocks consumed
    *offset += nbytes / 64;  
    for (int j = 0; j < nbytes; j++) {
        uint8_t byte = rand_buf[j];
        uint8_t high = byte >> 4;        // high nibble
        uint8_t low  = byte & 0x0F;        // low nibble
        row[2*j]     = (high == 0) ? 1 : (high == 1 ? -1 : 0);
        row[2*j + 1] = (low  == 0) ? 1 : (low  == 1 ? -1 : 0);
    }
}

// Allocate a 2D matrix of int8_t with N rows and N columns.
static int8_t **allocate_matrix() {
    int8_t **mat = malloc(N * sizeof(int8_t *));
    if (!mat) { perror("malloc"); exit(1); }
    for (int i = 0; i < N; i++) {
        mat[i] = malloc(N * sizeof(int8_t));
        if (!mat[i]) { perror("malloc"); exit(1); }
    }
    return mat;
}

// Free a 2D matrix allocated by allocate_matrix().
static void free_matrix(int8_t **mat) {
    for (int i = 0; i < N; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Calculate the absolute dot product of two rows.
static int32_t dot_product(int8_t *row1, int8_t *row2) {
    int32_t dot = 0;
    for (int i = 0; i < N; i++) {
        dot += row1[i] * row2[i];
    }
    return dot < 0 ? -dot : dot;
}

// Generate a 256x256 ternary matrix using the provided seed as the key
// and the round number (converted to an 8-byte big-endian nonce).
static bool generate_ternary_matrix(int8_t **M, const uint8_t *seed, uint64_t round) {
    uint8_t nonce[8];
    // Convert round number to an 8-byte big-endian nonce.
    for (int i = 0; i < 8; i++) {
        nonce[i] = (uint8_t)((round >> (56 - 8*i)) & 0xFF);
    }
    
    // Initialize offset (in 64-byte blocks) to 0.
    uint64_t offset = 0;
    
    // Generate the first row.
    generate_random_row(M[0], seed, nonce, &offset);
    print_row(M[0], 0, round, 1, 0);
    
    // Generate subsequent rows.
    for (int i = 1; i < N; i++) {
        int max_attempts = 1000;
        bool found_valid = false;
        int attempt = 0;
        int max_dot = 0;
        while (max_attempts-- > 0) {
            generate_random_row(M[i], seed, nonce, &offset);
            bool valid = true;
            max_dot = 0;
            for (int j = 0; j < i; j++) {
                int32_t dp = dot_product(M[i], M[j]);
                if (dp > max_dot)
                    max_dot = dp;
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

// Convert a byte array into a binary array (each bit becomes one byte 0 or 1).
static void bytes_to_binary(const uint8_t *bytes, uint8_t *binary) {
    for (int i = 0; i < N/8; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i*8 + j] = (bytes[i] >> (7 - j)) & 1;
        }
    }
}

// Convert a binary array (of length N) to bytes (N/8 bytes).
static void binary_to_bytes(const uint8_t *binary, uint8_t *bytes) {
    memset(bytes, 0, N/8);
    for (int i = 0; i < N; i++) {
        bytes[i/8] |= binary[i] << (7 - (i % 8));
    }
}

// Ternary transform function.
static void ternary_transform(int8_t **M, uint8_t *in, uint8_t *out, uint8_t *noise) {
    for (int i = 0; i < N; i++) {
        int32_t sum = 0;
        for (int j = 0; j < N; j++) {
            int val = in[j] ? 1 : -1;
            sum += M[i][j] * val;
        }
        /*if (sum>4) {
            out[i] = 1;
        } else if (sum<=-4) {
            out[i] = 0;
        } else {
            out[i] = noise[i];
        }*/
        //sum += noise[i] ? 4 : -4;
        sum+=noise[i];
        out[i] = sum > 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <seed> <nonce> [--debug]\n", argv[0]);
        fprintf(stderr, "seed and nonce should be 32-byte hex strings\n");
        return 1;
    }
    
    if (argc == 4) {
        if (strcmp(argv[3], "--debug") == 0)
            debug_mode = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[3]);
            return 1;
        }
    }
    
    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }
    
    uint8_t seed[32], nonce_input[32];
    if (!hex_to_bytes(argv[1], seed, 32) || !hex_to_bytes(argv[2], nonce_input, 32)) {
        fprintf(stderr, "Invalid hex input\n");
        return 1;
    }
    
    // Generate one matrix per round.
    int8_t ***matrices = malloc(ROUNDS * sizeof(int8_t **));
    if (!matrices) { perror("malloc"); exit(1); }
    for (int r = 0; r < ROUNDS; r++) {
        matrices[r] = allocate_matrix();
    }
    
    // Generate matrices for each round using the seed.
    for (uint64_t r = 0; r < ROUNDS; r++) {
        if (!generate_ternary_matrix(matrices[r], seed, r)) {
            fprintf(stderr, "Failed to generate matrix for round %llu\n", r);
            return 1;
        }
    }
    
    // Allocate buffers for transformation.
    uint8_t input[N], noise[N];
    uint8_t *curr = malloc(N), *next = malloc(N);
    if (!curr || !next) { perror("malloc"); exit(1); }
    
    // Derive input bits from SHA256(nonce_input).
    uint8_t input_hash[32];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, nonce_input, 32);
    SHA256_Final(input_hash, &sha256);
    //bytes_to_binary(input_hash, input);
    bytes_to_binary(nonce_input, input); // XXX
    
    // Derive noise bits from SHA256(input_hash).
    uint8_t noise_hash[32];
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input_hash, 32);
    SHA256_Final(noise_hash, &sha256);
    bytes_to_binary(noise_hash, noise);
    
    // Apply ROUNDS rounds of ternary transform.
    memcpy(curr, input, N);
    for (int r = 0; r < ROUNDS; r++) {
        ternary_transform(matrices[r], curr, next, noise);
        uint8_t *temp = curr;
        curr = next;
        next = temp;
    }
    
    // Convert final binary output to bytes and print as hex.
    uint8_t output[32];
    binary_to_bytes(curr, output);
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
    
    // Cleanup.
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

