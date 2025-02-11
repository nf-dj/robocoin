// Calculate row biases after matrix generation
static double row_biases[N];

void calculate_row_biases(int8_t **M) {
    for (int i = 0; i < N; i++) {
        int32_t total_dot = 0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                total_dot += dot_product(M[i], M[j], N);
            }
        }
        row_biases[i] = (double)total_dot / (N - 1);
        if (i < 5) printf("Row %d bias: %.3f\n", i, row_biases[i]); // Print first few for debugging
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/sha.h>

static double row_biases[N];

#define N 256
#define BLOCK_SIZE 32  // SHA256 produces 32 bytes
#define MAX_ATTEMPTS 1000
#define DOT_THRESHOLD 2

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

// Generate a random ternary row with sparse distribution
void generate_random_row(int8_t *row) {
    for (int j = 0; j < N; j++) {
        // Generate random number 0-31
        int r = rand() % 32;
        if (r == 0) {  // 1/32 chance of +1
            row[j] = 1;
        } else if (r == 1) {  // 1/32 chance of -1
            row[j] = -1;
        } else {  // 15/16 chance of 0
            row[j] = 0;
        }
    }
}

// Calculate dot product of two rows
int32_t dot_product(int8_t *row1, int8_t *row2, int len) {
    int32_t dot = 0;
    for (int i = 0; i < len; i++) {
        dot += row1[i] * row2[i];
    }
    return abs(dot);  // We care about absolute value for threshold
}

// Generate almost orthogonal ternary matrix
bool generate_ternary_matrix(int8_t **M) {
    // First row can be random
    generate_random_row(M[0]);
    
    // Generate subsequent rows
    for (int i = 1; i < N; i++) {
        bool found_valid_row = false;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            generate_random_row(M[i]);
            
            // Check dot product with all previous rows
            bool valid = true;
            for (int j = 0; j < i; j++) {
                if (dot_product(M[i], M[j], N) > DOT_THRESHOLD) {
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

// Convert byte array to binary array
void bytes_to_binary(uint8_t *bytes, uint8_t *binary, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i*8 + j] = (bytes[i] >> (7-j)) & 1;
        }
    }
}

// Count leading zeros in binary array
int count_leading_zeros(uint8_t *binary) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (binary[i] == 0) {
            count++;
        } else {
            break;
        }
    }
    return count;
}

// Ternary transform function
static void ternary_transform(int8_t **M, uint8_t *in, uint8_t *out, int n, uint8_t *noise) {
    for (int i = 0; i < n; i++) {
        int32_t dot = 0;
        for (int j = 0; j < n; j++) {
            int val = in[j] ? 1 : -1;
            dot += M[i][j] * val;
        }
        // Add bias based on previous output to counter correlation
        if (i > 0) {
            dot += (out[i-1] ? -1 : 1);  // Small push against previous bit
        }
        if (dot > 0)
            out[i] = 1;
        else if (dot < 0)
            out[i] = 0;
        else
            out[i] = noise[i];
    }
}

int main() {
    // Allocate matrix
    int8_t **M = malloc(N * sizeof(int8_t*));
    for (int i = 0; i < N; i++) {
        M[i] = malloc(N * sizeof(int8_t));
    }
    
    // Generate almost orthogonal ternary matrix
    srand(time(NULL));
    printf("Generating almost orthogonal ternary matrix...\n");
    if (!generate_ternary_matrix(M)) {
        printf("Failed to generate matrix. Exiting.\n");
        return 1;
    }
    printf("Matrix generated successfully.\n");
    
    // Calculate biases in matrix rows
    calculate_row_biases(M);
    printf("Row biases calculated.\n");
    
    // Buffers for SHA256 and binary conversion
    uint8_t hash[BLOCK_SIZE];
    uint8_t input[N];
    uint8_t noise[N];
    uint8_t output[N];
    SHA256_CTX sha256;
    uint64_t nonce = 0;
    time_t last_report = time(NULL);
    time_t last_stats = time(NULL);
    int max_zeros = 0;
    uint64_t hashes = 0;
    uint64_t total_hashes = 0;
    
    printf("Starting search for leading zeros...\n");
    
    while (1) {
        // Hash nonce to get input
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, input, BLOCK_SIZE);
        
        // Hash nonce+1 to get noise
        nonce++;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, noise, BLOCK_SIZE);
        nonce--;
        
        // Apply transform
        ternary_transform(M, input, output, N, noise);
        
        // Update statistics
        for (int i = 0; i < N; i++) {
            total_bits++;
            
            // Track bit distribution
            if (output[i] == 1) {
                total_ones++;
                curr_run_ones++;
                if (curr_run_ones > max_run_ones) max_run_ones = curr_run_ones;
                if (curr_run_zeros > 0) {
                    if (curr_run_zeros < MAX_RUN_TRACK) {
                        zero_runs[curr_run_zeros-1]++;
                    }
                    curr_run_zeros = 0;
                }
            } else {
                curr_run_zeros++;
                if (curr_run_zeros > max_run_zeros) max_run_zeros = curr_run_zeros;
                if (curr_run_ones > 0) {
                    if (curr_run_ones < MAX_RUN_TRACK) {
                        one_runs[curr_run_ones-1]++;
                    }
                    curr_run_ones = 0;
                }
            }
            
            uint8_t curr_bit = output[i];
            transitions[(last_bit << 1) | curr_bit]++;
            last_bit = curr_bit;
        }
        
        // Count leading zeros
        int zeros = count_leading_zeros(output);
        if (zeros > max_zeros) {
            max_zeros = zeros;
            printf("New record: %d leading zeros found with nonce %lu\n", zeros, nonce);
        }
        
        // Report progress every second
        time_t current = time(NULL);
        if (current > last_report) {
            printf("Progress: %lu hashes/s (%lu total), max zeros found: %d\n", 
                   hashes, total_hashes + hashes, max_zeros);
            last_report = current;
            total_hashes += hashes;
            hashes = 0;
        }
        
        // Report statistics every 15 seconds
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
            for(int i = 0; i < 4; i++) {
                double diff = transitions[i] - expected_trans;
                trans_chi += (diff * diff) / expected_trans;
            }
            bool serial_pass = trans_chi < 7.815;
            printf("3. Serial test chi-square: %.4f %s\n", 
                   trans_chi,
                   serial_pass ? "[PASS]" : "[FAIL]");

            if (!serial_pass) {
                printf("   Serial test details:\n");
                printf("   Transition  Count      Expected   Deviation\n");
                printf("   ----------  ---------  ---------  ---------\n");
                const char* trans_names[] = {"0->0", "0->1", "1->0", "1->1"};
                for(int i = 0; i < 4; i++) {
                    double diff = transitions[i] - expected_trans;
                    double dev_percent = (diff / expected_trans) * 100;
                    printf("   %-10s  %9lu  %9.0f  %+8.2f%%\n", 
                           trans_names[i], transitions[i], expected_trans, dev_percent);
                }
                printf("\n");
            }
            
            double entropy = 0;
            uint64_t total_runs = 0;
            for(int i = 0; i < MAX_RUN_TRACK; i++) {
                total_runs += zero_runs[i] + one_runs[i];
            }
            
            if(total_runs > 0) {
                for(int i = 0; i < MAX_RUN_TRACK; i++) {
                    double p_zero = (double)zero_runs[i] / total_runs;
                    double p_one = (double)one_runs[i] / total_runs;
                    if(p_zero > 0) entropy -= p_zero * log2(p_zero);
                    if(p_one > 0) entropy -= p_one * log2(p_one);
                }
            }
            printf("4. Run length entropy: %.4f bits %s (ideal: 3.0)\n", 
                   entropy,
                   (fabs(entropy - 3.0) < 0.1) ? "[PASS]" : "[FAIL]");
            
            printf("5. Chi-square deviation from 50/50: %.4f%% %s\n\n", 
                   fabs(ones_ratio - 0.5) * 200,
                   (fabs(ones_ratio - 0.5) * 200 < 0.01) ? "[PASS]" : "[FAIL]");
            
            last_stats = current;
        }
        
        nonce++;
        hashes++;
    }
    
    // Cleanup
    for (int i = 0; i < N; i++) {
        free(M[i]);
    }
    free(M);
    
    return 0;
}
