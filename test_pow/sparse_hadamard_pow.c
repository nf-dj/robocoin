#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/sha.h>

#define N 256                 // Final weighing matrix size: 256 x 256
#define HALF_N (N/2)          // 128
#define BLOCK_SIZE 32         // SHA256 produces 32 bytes

// Global randomness statistics (unchanged)
static uint64_t total_ones = 0;
static uint64_t total_bits = 0;
static uint64_t max_run_zeros = 0;
static uint64_t max_run_ones = 0;
static uint64_t curr_run_zeros = 0;
static uint64_t curr_run_ones = 0;

// Transition counts for serial test (00, 01, 10, 11)
static uint64_t transitions[4] = {0};
static uint8_t last_bit = 0;

#define MAX_RUN_TRACK 32
static uint64_t zero_runs[MAX_RUN_TRACK] = {0};
static uint64_t one_runs[MAX_RUN_TRACK] = {0};

// --------------------------------------------------------------------------
// Fisher–Yates shuffle for an integer array.
void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// --------------------------------------------------------------------------
// Generate a Hadamard matrix of size n (n a power of two) using a Sylvester‐like method.
void generate_hadamard_matrix_n(int8_t **H, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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
}

// --------------------------------------------------------------------------
// Build a 256x256 weighing matrix using two 128x128 Hadamard matrices.
// First, we create a block-diagonal matrix W where the top‐left block is H1
// and the bottom–right block is H2 (all other entries zero). Then we randomly
// permute the rows and columns (using the same pair of permutation matrices)
// to “spread” the nonzero entries. (Note: since a permutation is orthogonal,
// the final W satisfies W Wᵀ = 128·I.)
void generate_weighing_matrix(int8_t **W) {
    // Allocate two temporary 128x128 matrices.
    int8_t **H1 = malloc(HALF_N * sizeof(int8_t *));
    int8_t **H2 = malloc(HALF_N * sizeof(int8_t *));
    for (int i = 0; i < HALF_N; i++) {
        H1[i] = malloc(HALF_N * sizeof(int8_t));
        H2[i] = malloc(HALF_N * sizeof(int8_t));
    }
    // Generate the two Hadamard matrices.
    generate_hadamard_matrix_n(H1, HALF_N);
    generate_hadamard_matrix_n(H2, HALF_N);

    // Build the 256x256 block-diagonal matrix:
    // Rows 0..127: copy H1 into columns 0..127; columns 128..255 remain 0.
    // Rows 128..255: copy H2 into columns 128..255; columns 0..127 remain 0.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W[i][j] = 0;
        }
    }
    for (int i = 0; i < HALF_N; i++) {
        for (int j = 0; j < HALF_N; j++) {
            W[i][j] = H1[i][j];
        }
    }
    for (int i = 0; i < HALF_N; i++) {
        for (int j = 0; j < HALF_N; j++) {
            W[i + HALF_N][j + HALF_N] = H2[i][j];
        }
    }
    
    // Free the temporary Hadamard matrices.
    for (int i = 0; i < HALF_N; i++) {
        free(H1[i]);
        free(H2[i]);
    }
    free(H1);
    free(H2);

    // Now, randomly permute the rows and columns of W.
    int *row_perm = malloc(N * sizeof(int));
    int *col_perm = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        row_perm[i] = i;
        col_perm[i] = i;
    }
    shuffle(row_perm, N);
    shuffle(col_perm, N);

    // Create a temporary matrix to hold the permuted version.
    int8_t **W_temp = malloc(N * sizeof(int8_t *));
    for (int i = 0; i < N; i++) {
        W_temp[i] = malloc(N * sizeof(int8_t));
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W_temp[i][j] = W[row_perm[i]][col_perm[j]];
        }
    }
    // Copy W_temp back into W.
    for (int i = 0; i < N; i++) {
        memcpy(W[i], W_temp[i], N * sizeof(int8_t));
    }
    // Clean up temporary arrays.
    for (int i = 0; i < N; i++) {
        free(W_temp[i]);
    }
    free(W_temp);
    free(row_perm);
    free(col_perm);

    // Verify orthogonality.
    // Each row should have a self dot–product of HALF_N (i.e. 128), and
    // the dot product between any two distinct rows should be 0.
    for (int i = 0; i < N; i++) {
        int dot_self = 0;
        for (int j = 0; j < N; j++) {
            dot_self += W[i][j] * W[i][j];
        }
        if (dot_self != HALF_N) {
            fprintf(stderr, "Error: Row %d self dot product is %d, expected %d\n",
                    i, dot_self, HALF_N);
            exit(1);
        }
        for (int k = i + 1; k < N; k++) {
            int dot = 0;
            for (int j = 0; j < N; j++) {
                dot += W[i][j] * W[k][j];
            }
            if (dot != 0) {
                fprintf(stderr, "Error: Rows %d and %d are not orthogonal (dot = %d)\n",
                        i, k, dot);
                exit(1);
            }
        }
    }
    printf("Weighing matrix generated: size %dx%d, weight per row = %d (%.2f%% nonzero).\n",
           N, N, HALF_N, (float)(N * HALF_N * 100) / (N * N));
}

// --------------------------------------------------------------------------
// Convert a byte array into a binary (0/1) array.
void bytes_to_binary(uint8_t *bytes, uint8_t *binary, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        for (int j = 0; j < 8; j++) {
            binary[i*8 + j] = (bytes[i] >> (7 - j)) & 1;
        }
    }
}

// Count leading zeros in a binary array.
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

// --------------------------------------------------------------------------
// Apply the weighing transform: for each row i, compute the dot product
// of row i of W with the input vector (interpreting bits as ±1).
// If the dot is positive, output bit 1; if negative, output 0; if zero,
// use the noise bit as a tiebreaker.
static void binary_weighing_transform(int8_t **W, uint8_t *in, uint8_t *out, int n, uint8_t *noise) {
    for (int i = 0; i < n; i++) {
        int32_t dot = 0;
        for (int j = 0; j < n; j++) {
            int val = in[j] ? 1 : -1;
            dot += W[i][j] * val;
        }
        if (dot > 0)
            out[i] = 1;
        else if (dot < 0)
            out[i] = 0;
        else
            out[i] = noise[i];
    }
}

// --------------------------------------------------------------------------
// Main function. This is largely the same as your original code except
// that we now generate a weighing matrix (using two 128x128 Hadamard matrices)
// and use it in the transform.
int main() {
    // Allocate the 256x256 weighing matrix W.
    int8_t **W = malloc(N * sizeof(int8_t *));
    for (int i = 0; i < N; i++) {
        W[i] = malloc(N * sizeof(int8_t));
    }
    
    srand(time(NULL));
    generate_weighing_matrix(W);
    
    // Buffers for SHA256, binary conversion, etc.
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
        // Hash nonce to get input bits.
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, input, BLOCK_SIZE);
        
        // Hash nonce+1 to get noise bits.
        nonce++;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, &nonce, sizeof(nonce));
        SHA256_Final(hash, &sha256);
        bytes_to_binary(hash, noise, BLOCK_SIZE);
        nonce--;
        
        // Apply the weighing transform.
        binary_weighing_transform(W, input, output, N, noise);
        
        // Update randomness statistics.
        for (int i = 0; i < N; i++) {
            total_bits++;
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
        
        // Report detailed randomness statistics every 15 seconds.
        if (current >= last_stats + 15) {
            double ones_ratio = (double)total_ones / total_bits;
            printf("\nRandomness statistics after %lu hashes:\n", total_hashes);
            
            double ones_percent = ones_ratio * 100;
            double chi_square = fabs(ones_ratio - 0.5) * 200;
            int max_acceptable_run = 32;
            
            printf("1. Bit distribution: %.4f%% ones (ideal: 50%%) %s\n", 
                   ones_percent,
                   (fabs(ones_percent - 50.0) < 0.1) ? "[PASS]" : "[FAIL]");
            
            printf("2. Longest runs: %lu zeros, %lu ones %s\n", 
                   max_run_zeros, max_run_ones,
                   (max_run_zeros <= max_acceptable_run && max_run_ones <= max_acceptable_run) ? "[PASS]" : "[FAIL]");
            
            double expected_trans = total_bits / 4.0;
            double trans_chi = 0;
            for (int i = 0; i < 4; i++) {
                double diff = transitions[i] - expected_trans;
                trans_chi += (diff * diff) / expected_trans;
            }
            printf("3. Serial test chi-square: %.4f %s\n", 
                   trans_chi,
                   (trans_chi < 7.815) ? "[PASS]" : "[FAIL]");
            
            double entropy = 0;
            uint64_t total_runs = 0;
            uint64_t bits_in_runs = 0;
            for (int i = 0; i < MAX_RUN_TRACK; i++) {
                total_runs += zero_runs[i] + one_runs[i];
                bits_in_runs += (i + 1) * (zero_runs[i] + one_runs[i]);
            }
            
            if (total_runs > 0) {
                for (int i = 0; i < MAX_RUN_TRACK; i++) {
                    double p_zero = (double)zero_runs[i] / total_runs;
                    double p_one = (double)one_runs[i] / total_runs;
                    if (p_zero > 0)
                        entropy -= p_zero * log2(p_zero);
                    if (p_one > 0)
                        entropy -= p_one * log2(p_one);
                }
            }
            
            bool entropy_pass = fabs(entropy - 3.0) < 0.1;
            printf("4. Run length entropy: %.4f bits %s (ideal: 3.0)\n", 
                   entropy,
                   entropy_pass ? "[PASS]" : "[FAIL]");
            
            if (!entropy_pass) {
                printf("\nRun length distribution:\n");
                printf("Length  |  Zero runs  |  One runs  |  Total  |  Probability\n");
                printf("--------|-------------|------------|---------|-------------\n");
                for (int i = 0; i < 10; i++) {
                    double prob = (double)(zero_runs[i] + one_runs[i]) / total_runs;
                    printf("%7d | %11lu | %10lu | %8lu | %11.4f%%\n", 
                           i + 1, zero_runs[i], one_runs[i], zero_runs[i] + one_runs[i],
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
    
    // Cleanup.
    for (int i = 0; i < N; i++) {
        free(W[i]);
    }
    free(W);
    
    return 0;
}

