#ifndef BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H
#define BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENS_IN_SIZE 32     // Input/output size

// Combined structure for hash computation context
typedef struct {
    int8_t* expansion_mat;    // [TENS_HIDDEN x INPUT_BITS]
    int8_t* hidden_mats;      // NUM_HIDDEN_LAYERS matrices, each [TENS_HIDDEN x TENS_HIDDEN]
    int8_t* compression_mat;  // [INPUT_BITS x TENS_HIDDEN]
    int8_t* state;            // working state (size: TENS_HIDDEN)
    int8_t* next_state;       // working state (size: TENS_HIDDEN)
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
