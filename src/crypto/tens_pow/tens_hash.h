#ifndef BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H
#define BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENS_IN_SIZE 32     // Input/output size
#define TENS_HIDDEN 256    // Hidden dimension size
#define TENS_ROUNDS 64      // Number of middle rounds

// Combined structure for hash computation context
typedef struct {
    // Matrix data - stored in contiguous blocks
    int8_t *middle_mats;   // Single block of size TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN
    
    // Buffer data
    uint8_t *state;         // Size TENS_HIDDEN
    uint8_t *next_state;    // Size TENS_HIDDEN
    int8_t *noise;         // Holds all noise for all rounds
} TensHashContext;

// Core functions
TensHashContext* tens_hash_init(const uint8_t seed[32]);
void tens_hash_free(TensHashContext* ctx);
void tens_hash_precomputed(const uint8_t input[TENS_IN_SIZE], TensHashContext* ctx, uint8_t output[TENS_IN_SIZE]);
void tens_hash(const uint8_t input[TENS_IN_SIZE], const uint8_t seed[TENS_IN_SIZE], uint8_t output[TENS_IN_SIZE]);

#ifdef __cplusplus
}
#endif

#endif // BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H
