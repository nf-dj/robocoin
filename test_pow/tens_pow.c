#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sodium.h>
#include <inttypes.h>

#define IN_SIZE 32
#define OPS_PER_HASH (256*256*2*64)

typedef enum {
    IMPL_INT8 = 0,
    IMPL_FP32 = 1,
    IMPL_FP16 = 2
} ImplType;

typedef struct {
    int8_t **middle_mats[64];
    ImplType impl_type;
} PrecomputedMatrices;

typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t  *noise;
} HashBuffers;

extern PrecomputedMatrices* precompute_matrices(uint8_t seed[32], ImplType impl_type);
extern void free_matrices(PrecomputedMatrices* matrices);
extern HashBuffers* init_hash_buffers(void);
extern void free_hash_buffers(HashBuffers* buffers);
extern void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices, HashBuffers* buffers, uint8_t output[IN_SIZE]);

void hex_to_bytes(const char *hex, uint8_t *bytes, size_t len) {
    for (size_t i = 0; i < len; i++) {
        sscanf(hex + 2 * (len - 1 - i), "%2hhx", &bytes[i]);
    }
}

void print_hex(uint8_t *bytes, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", bytes[len - 1 - i]);
    }
    printf("\n");
}

int count_leading_zero_bits(uint8_t *hash) {
    int count = 0;
    for (int i = IN_SIZE - 1; i >= 0; i--) {
        uint8_t byte = hash[i];
        if (byte == 0) {
            count += 8;
        } else {
            for (int bit = 7; bit >= 0; bit--) {
                if ((byte >> bit) & 1)
                    return count;
                count++;
            }
            break;
        }
    }
    return count;
}

int check_difficulty(uint8_t *hash, int difficulty) {
    return count_leading_zero_bits(hash) >= difficulty;
}

void set_nonce_from_counter(uint8_t *nonce, uint64_t counter) {
    memset(nonce, 0, IN_SIZE);
    for (int i = 0; i < 8; i++) {
        nonce[i] = (counter >> (8 * i)) & 0xFF;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <seed_hex> <difficulty> <impl_type>\n", argv[0]);
        fprintf(stderr, "  seed_hex: 64 hex characters\n");
        fprintf(stderr, "  difficulty: number of leading 0 bits required (1-256)\n");
        fprintf(stderr, "  impl_type: 0 for int8, 1 for fp32, 2 for fp16\n");
        return 1;
    }

    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }

    uint8_t seed[32];
    hex_to_bytes(argv[1], seed, 32);
    
    int difficulty = atoi(argv[2]);
    if (difficulty < 1 || difficulty > 256) {
        fprintf(stderr, "Difficulty must be between 1 and 256\n");
        return 1;
    }

    ImplType impl_type = atoi(argv[3]);
    if (impl_type < 0 || impl_type > 2) {
        fprintf(stderr, "Implementation type must be 0, 1, or 2\n");
        return 1;
    }

    PrecomputedMatrices* matrices = precompute_matrices(seed, impl_type);
    HashBuffers* buffers = init_hash_buffers();
    if (!matrices || !buffers) {
        fprintf(stderr, "Failed to allocate memory\n");
        free_matrices(matrices);
        free_hash_buffers(buffers);
        return 1;
    }

    uint8_t nonce[IN_SIZE];
    uint8_t hash[IN_SIZE];
    uint64_t attempts = 0;
    time_t start_time = time(NULL);
    time_t last_report = start_time;
    uint64_t last_attempts = 0;
    int best_zero_bits = 0;
    
    printf("Mining with %s implementation:\n", 
           impl_type == IMPL_FP32 ? "fp32" : 
           impl_type == IMPL_FP16 ? "fp16" : "int8");
    printf("  Seed: ");
    print_hex(seed, 32);
    printf("  Difficulty: %d leading 0 bits\n", difficulty);
    printf("\nProgress:\n");
    printf("  Time    Hash Rate      TOPS         Total Hashes    Best Bits\n");
    printf("  ----    ---------    --------      ------------    ----------\n");

    while (1) {
        set_nonce_from_counter(nonce, attempts);
        attempts++;

        tens_hash_precomputed(nonce, matrices, buffers, hash);
        
        int zeros = count_leading_zero_bits(hash);
        if (zeros > best_zero_bits) {
            best_zero_bits = zeros;
        }

        if (check_difficulty(hash, difficulty)) {
            time_t end_time = time(NULL);
            double duration = difftime(end_time, start_time);
            double avg_tops = (attempts * OPS_PER_HASH) / (duration * 1e12);
            
            printf("\nSolution found!\n");
            printf("Nonce: ");
            print_hex(nonce, IN_SIZE);
            printf("Hash:  ");
            print_hex(hash, IN_SIZE);
            printf("Stats:\n");
            printf("  Time: %.1f seconds\n", duration);
            printf("  Total hashes: %" PRIu64 "\n", attempts);
            printf("  Avg hash rate: %.1f h/s\n", attempts / duration);
            printf("  Avg TOPS: %.6f\n", avg_tops);
            break;
        }

        time_t current_time = time(NULL);
        if (current_time > last_report) {
            double interval = difftime(current_time, last_report);
            uint64_t interval_hashes = attempts - last_attempts;
            double hash_rate = interval_hashes / interval;
            double tops = (hash_rate * OPS_PER_HASH) / 1e12;
            double total_time = difftime(current_time, start_time);
            
            printf("  %4.0fs    %7.0f h/s    %.6f    %12" PRIu64 "    %10d\r", 
                   total_time, hash_rate, tops, attempts, best_zero_bits);
            fflush(stdout);
            
            last_report = current_time;
            last_attempts = attempts;
        }
    }

    free_matrices(matrices);
    free_hash_buffers(buffers);
    return 0;
}