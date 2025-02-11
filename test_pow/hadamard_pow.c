#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/sha.h>

#define N 256
#define BLOCK_SIZE 32  // SHA256 produces 32 bytes

// Global statistics variables
static uint64_t total_ones = 0;
static uint64_t total_bits = 0;
static uint64_t max_run_zeros = 0;
static uint64_t max_run_ones = 0;
static uint64_t curr_run_zeros = 0;
static uint64_t curr_run_ones = 0;

// Transition counts for serial test (00,01,10,11)
static uint64_t transitions[4] = {0};
static uint8_t last_bit = 0;

// Run length distribution
#define MAX_RUN_TRACK 32
static uint64_t zero_runs[MAX_RUN_TRACK] = {0};
static uint64_t one_runs[MAX_RUN_TRACK] = {0};

// Generate randomized Hadamard matrix and verify orthogonality
void generate_hadamard_matrix(int8_t **H) {
    // First generate standard Hadamard matrix using Sylvester construction
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int bit = 1;
            int x = i, y = j;
            while (x > 0 && y > 0) {
                bit ^= (x & 1) & (y & 1);
                x >>= 1;
                y >>= 1;
            }
            H[i][j] = bit ? 1 : -1;
        }
    }
    
    // Random permutation of rows
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i + 1);
        if (i != j) {
            for (int k = 0; k < N; k++) {
                int8_t temp = H[i][k];
                H[i][k] = H[j][k];
                H[j][k] = temp;
            }
        }
    }
    
    // Random permutation of columns
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i + 1);
        if (i != j) {
            for (int k = 0; k < N; k++) {
                int8_t temp = H[k][i];
                H[k][i] = H[k][j];
                H[k][j] = temp;
            }
        }
    }
    
    // Verify orthogonality
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            int32_t dot = 0;
            for (int k = 0; k < N; k++) {
                dot += H[i][k] * H[j][k];
            }
            if (dot != 0) {
                fprintf(stderr, "Error: Matrix not orthogonal!\n");
                exit(1);
            }
        }
    }
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

// Binary Hadamard transform function (provided in original code)
static void binary_hadamard_transform(int8_t **H, uint8_t *in, uint8_t *out, int n, uint8_t *noise) {
    for (int i = 0; i < n; i++) {
        int32_t dot = 0;
        for (int j = 0; j < n; j++) {
            int val = in[j] ? 1 : -1;
            dot += H[i][j] * val;
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
    int8_t **H = malloc(N * sizeof(int8_t*));
    for (int i = 0; i < N; i++) {
        H[i] = malloc(N * sizeof(int8_t));
    }
    
    // Generate randomized Hadamard matrix
    srand(time(NULL));
    generate_hadamard_matrix(H);
    
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
        binary_hadamard_transform(H, input, output, N, noise);
        
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
            
            // Track transitions (serial test)
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
            
            // Basic distribution tests
            double ones_percent = ones_ratio * 100;
            double chi_square = fabs(ones_ratio - 0.5) * 200;
            int max_acceptable_run = 32;
            
            printf("1. Bit distribution: %.4f%% ones (ideal: 50%%) %s\n", 
                   ones_percent,
                   (fabs(ones_percent - 50.0) < 0.1) ? "[PASS]" : "[FAIL]");
            
            printf("2. Longest runs: %lu zeros, %lu ones %s\n", 
                   max_run_zeros, max_run_ones,
                   (max_run_zeros <= max_acceptable_run && max_run_ones <= max_acceptable_run) ? "[PASS]" : "[FAIL]");
            
            // Serial test (transition probabilities)
            double expected_trans = total_bits / 4.0;
            double trans_chi = 0;
            for(int i = 0; i < 4; i++) {
                double diff = transitions[i] - expected_trans;
                trans_chi += (diff * diff) / expected_trans;
            }
            printf("3. Serial test chi-square: %.4f %s\n", 
                   trans_chi,
                   (trans_chi < 7.815) ? "[PASS]" : "[FAIL]"); // 95% confidence for 3 DOF
            
            // Run length distribution analysis
            double entropy = 0;
            uint64_t total_runs = 0;
            uint64_t bits_in_runs = 0;
            for(int i = 0; i < MAX_RUN_TRACK; i++) {
                total_runs += zero_runs[i] + one_runs[i];
                bits_in_runs += (i+1) * (zero_runs[i] + one_runs[i]);
            }
            
            if(total_runs > 0) {
                for(int i = 0; i < MAX_RUN_TRACK; i++) {
                    double p_zero = (double)zero_runs[i] / total_runs;
                    double p_one = (double)one_runs[i] / total_runs;
                    if(p_zero > 0) entropy -= p_zero * log2(p_zero);
                    if(p_one > 0) entropy -= p_one * log2(p_one);
                }
            }
            
            bool entropy_pass = fabs(entropy - 3.0) < 0.1;
            printf("4. Run length entropy: %.4f bits %s (ideal: 3.0)\n", 
                   entropy,
                   entropy_pass ? "[PASS]" : "[FAIL]");
            
            // Only show distribution details if test failed
            if (!entropy_pass) {
                printf("\nRun length distribution:\n");
                printf("Length  |  Zero runs  |  One runs  |  Total  |  Probability\n");
                printf("--------|-------------|------------|----------|-------------\n");
                for(int i = 0; i < 10; i++) {
                    double prob = (double)(zero_runs[i] + one_runs[i]) / total_runs;
                    printf("%7d | %11lu | %10lu | %8lu | %11.4f%%\n", 
                           i+1, zero_runs[i], one_runs[i], 
                           zero_runs[i] + one_runs[i],
                           prob * 100);
                }
                printf("\nTotal bits processed: %lu\n", total_bits);
                printf("Total bits in runs: %lu\n", bits_in_runs);
                printf("Total runs counted: %lu\n\n", total_runs);
            }
            
            printf("5. Chi-square deviation from 50/50: %.4f%% %s\n", 
                   chi_square,
                   (chi_square < 0.01) ? "[PASS]" : "[FAIL]");
            
            printf("\n");
            last_stats = current;
        }
        
        nonce++;
        hashes++;
    }
    
    // Cleanup
    for (int i = 0; i < N; i++) {
        free(H[i]);
    }
    free(H);
    
    return 0;
}
