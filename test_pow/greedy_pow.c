#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/sha.h>

// Constants
#define N 256                   // Matrix dimensions: 256 x 256
#define BLOCK_SIZE 32           // SHA256 produces 32 bytes (256 bits)
#define MAX_ATTEMPTS 1000
#define DOT_THRESHOLD 5

// Global statistics variables
static uint64_t total_ones = 0;
static uint64_t total_bits = 0;
static uint64_t max_run_zeros = 0;
static uint64_t max_run_ones = 0;
static uint64_t curr_run_zeros = 0;
static uint64_t curr_run_ones = 0;
static uint64_t transitions[4] = {0};
static uint8_t last_bit = 0;
#define MAX_RUN_TRACK 32
static uint64_t zero_runs[MAX_RUN_TRACK] = {0};
static uint64_t one_runs[MAX_RUN_TRACK] = {0};

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
            printf("Row %d bias: %.3f\n", i, row_biases[i]); // Print first few for debugging
    }
}

// Generate a random ternary row with a sparse distribution (mostly zeros)
void generate_random_row(int8_t *row) {
    for (int j = 0; j < N; j++) {
        // Generate a random number from 0 to 31.
        //int r = rand() % 32;
        int r = rand() % 16;
        if (r == 0)
            row[j] = 1;
        else if (r == 1)
            row[j] = -1;
        else
            row[j] = 0;
    }
}

// Calculate the absolute dot product of two rows.
int32_t dot_product(int8_t *row1, int8_t *row2, int len) {
    int32_t dot = 0;
    for (int i = 0; i < len; i++) {
        dot += row1[i] * row2[i];
    }
    return abs(dot);
}

// Generate an "almost orthogonal" ternary matrix M (size N x N).
// Each new row is accepted only if its dot product (absolute value)
// with every previous row is ≤ DOT_THRESHOLD.
bool generate_ternary_matrix(int8_t **M) {
    // First row is random.
    generate_random_row(M[0]);
    
    // Generate subsequent rows.
    for (int i = 1; i < N; i++) {
        bool found_valid_row = false;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            generate_random_row(M[i]);
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

// Convert a byte array to a binary array (0 or 1) of length nbytes*8 (which equals N bits)
void bytes_to_binary(uint8_t *bytes, uint8_t *binary, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i * 8 + j] = (bytes[i] >> (7 - j)) & 1;
        }
    }
}

// Count leading zeros in a binary array (length N)
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

// Ternary transform function.
// For each row i of matrix M, compute the dot product of the row (using the input bits,
// interpreting 0 as -1 and 1 as +1). A small bias based on the previous output is added
// (if any) to help counter correlation. Then output 1 if dot > 0, 0 if dot < 0, and if
// dot == 0, use the corresponding noise bit.
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

//
// Main program:
//   - Accepts an optional command-line argument specifying the number of rounds (default 16).
//   - Generates that many different ternary matrices (each of size 256x256) and then uses
//     them in sequence for each nonce. The same matrices are used for all nonces.
int main(int argc, char *argv[]) {
    int rounds = 16; // Default number of rounds.
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0)
            rounds = 16;
    }
    printf("Using %d rounds of the ternary transform.\n", rounds);
    
    // Allocate an array to hold pointers to each round's matrix.
    int8_t ***matrices = malloc(rounds * sizeof(int8_t **));
    if (!matrices) {
        perror("malloc");
        exit(1);
    }
    for (int r = 0; r < rounds; r++) {
        matrices[r] = malloc(N * sizeof(int8_t *));
        if (!matrices[r]) {
            perror("malloc");
            exit(1);
        }
        for (int i = 0; i < N; i++) {
            matrices[r][i] = malloc(N * sizeof(int8_t));
            if (!matrices[r][i]) {
                perror("malloc");
                exit(1);
            }
        }
        printf("Generating ternary matrix for round %d...\n", r);
        if (!generate_ternary_matrix(matrices[r])) {
            printf("Failed to generate matrix for round %d. Exiting.\n", r);
            exit(1);
        }
        // Optionally, calculate and print row biases for the first matrix.
        if (r == 0) {
            calculate_row_biases(matrices[r]);
            printf("Row biases for round 0 calculated.\n");
        }
    }
    
    // Allocate buffers for SHA256, binary conversion, etc.
    uint8_t hash[BLOCK_SIZE];
    uint8_t input[N];         // 256 bits from SHA256 hash
    uint8_t noise_bits[N];    // 256 noise bits from SHA256 of (nonce+1)
    uint8_t output[N];        // Final output bits after all rounds
    
    // Allocate two temporary buffers to chain rounds.
    uint8_t *current_bits = malloc(N * sizeof(uint8_t));
    uint8_t *next_bits = malloc(N * sizeof(uint8_t));
    if (!current_bits || !next_bits) {
        perror("malloc");
        exit(1);
    }
    
    SHA256_CTX sha256;
    uint64_t nonce = 0;
    time_t last_report = time(NULL);
    time_t last_stats = time(NULL);
    int max_zeros = 0;
    uint64_t hashes = 0;
    uint64_t total_hashes = 0;
    
    printf("Starting search for leading zeros...\n");
    
    while (1) {
        // Generate input bits from SHA256(nonce).
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, input, BLOCK_SIZE);
        // Copy the input bits into our working buffer.
        memcpy(current_bits, input, N * sizeof(uint8_t));
        
        // Generate noise bits from SHA256(nonce+1).
        nonce++;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, noise_bits, BLOCK_SIZE);
        nonce--;  // Restore nonce for the main loop.
        
        // Apply each round of the ternary transform sequentially.
        // The output of one round becomes the input for the next.
        for (int r = 0; r < rounds; r++) {
            ternary_transform(matrices[r], current_bits, next_bits, N, noise_bits);
            // Swap the pointers so that next_bits becomes current_bits for the next round.
            uint8_t *temp = current_bits;
            current_bits = next_bits;
            next_bits = temp;
        }
        // After all rounds, current_bits holds the final output.
        memcpy(output, current_bits, N * sizeof(uint8_t));
        
        // Update statistics based on the final output.
        for (int i = 0; i < N; i++) {
            total_bits++;
            
            // Track bit distribution and runs.
            if (output[i] == 1) {
                total_ones++;
                curr_run_ones++;
                if (curr_run_ones > max_run_ones)
                    max_run_ones = curr_run_ones;
                if (curr_run_zeros > 0) {
                    if (curr_run_zeros < MAX_RUN_TRACK)
                        zero_runs[curr_run_zeros - 1]++;
                    curr_run_zeros = 0;
                }
            } else {
                curr_run_zeros++;
                if (curr_run_zeros > max_run_zeros)
                    max_run_zeros = curr_run_zeros;
                if (curr_run_ones > 0) {
                    if (curr_run_ones < MAX_RUN_TRACK)
                        one_runs[curr_run_ones - 1]++;
                    curr_run_ones = 0;
                }
            }
            uint8_t curr_bit = output[i];
            transitions[(last_bit << 1) | curr_bit]++;
            last_bit = curr_bit;
        }
        
        // Count leading zeros in the final output.
        int zeros = count_leading_zeros(output);
        if (zeros > max_zeros) {
            max_zeros = zeros;
            printf("New record: %d leading zeros found with nonce %lu\n", zeros, nonce);
        }
        
        // Report progress every second.
        time_t current = time(NULL);
        if (current > last_report) {
            printf("Progress: %lu hashes/s (%lu total), max zeros found: %d\n",
                   hashes, total_hashes + hashes, max_zeros);
            last_report = current;
            total_hashes += hashes;
            hashes = 0;
        }
        
        // Report detailed statistics every 15 seconds.
        if (current >= last_stats + 15) {
            double ones_ratio = (double)total_ones / total_bits;
            printf("\nRandomness statistics after %lu hashes:\n", total_hashes);
            printf("1. Bit distribution: %.4f%% ones (ideal: 50%%) %s\n",
                   ones_ratio * 100,
                   (fabs(ones_ratio * 100 - 50.0) < 0.1) ? "[PASS]" : "[FAIL]");
            printf("2. Longest runs: %lu zeros, %lu ones %s\n",
                   max_run_zeros, max_run_ones,
                   (max_run_zeros <= 32 && max_run_ones <= 32) ? "[PASS]" : "[FAIL]");
            
            double expected_trans = total_bits / 4.0;
            double trans_chi = 0;
            for (int i = 0; i < 4; i++) {
                double diff = transitions[i] - expected_trans;
                trans_chi += (diff * diff) / expected_trans;
            }
            bool serial_pass = trans_chi < 7.815;
            printf("3. Serial test chi-square: %.4f %s\n", trans_chi, serial_pass ? "[PASS]" : "[FAIL]");
            
            double entropy = 0;
            uint64_t total_runs = 0;
            for (int i = 0; i < MAX_RUN_TRACK; i++) {
                total_runs += zero_runs[i] + one_runs[i];
            }
            if (total_runs > 0) {
                for (int i = 0; i < MAX_RUN_TRACK; i++) {
                    double p_zero = (double)zero_runs[i] / total_runs;
                    double p_one = (double)one_runs[i] / total_runs;
                    if (p_zero > 0) entropy -= p_zero * log2(p_zero);
                    if (p_one > 0) entropy -= p_one * log2(p_one);
                }
            }
            printf("4. Run length entropy: %.4f bits %s (ideal: 3.0)\n",
                   entropy, (fabs(entropy - 3.0) < 0.1) ? "[PASS]" : "[FAIL]");
            printf("5. Chi-square deviation from 50/50: %.4f%% %s\n\n",
                   fabs(ones_ratio - 0.5) * 200,
                   (fabs(ones_ratio - 0.5) * 200 < 0.01) ? "[PASS]" : "[FAIL]");
            
            last_stats = current;
        }
        
        nonce++;
        hashes++;
    }
    
    // Cleanup (unreachable due to infinite loop, but provided for completeness)
    for (int r = 0; r < rounds; r++) {
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

