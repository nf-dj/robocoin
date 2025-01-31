#ifndef BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H
#define BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IN_SIZE 32      // Input/output size
#define HIDDEN 1024     // Hidden dimension size
#define Q 255          // 8-bit modulus
#define ROUNDS 64      // Number of middle rounds

// Structure to hold precomputed matrices
typedef struct {
    uint8_t **expand_mat;   // HIDDEN x IN_SIZE
    uint8_t **middle_mats[ROUNDS];  // ROUNDS of HIDDEN x HIDDEN
    uint8_t **compress_mat; // IN_SIZE x HIDDEN
} PrecomputedMatrices;

// Structure to hold reusable buffers
typedef struct {
    uint8_t *state;
    uint8_t *next_state;
    int8_t *noise;     // Holds all noise for all rounds
} HashBuffers;

// Core functions
PrecomputedMatrices* precompute_matrices(uint8_t seed[32]);
void free_matrices(PrecomputedMatrices* matrices);
HashBuffers* init_hash_buffers(void);
void free_hash_buffers(HashBuffers* buffers);
void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices,
                          HashBuffers* buffers, uint8_t output[IN_SIZE]);
void tens_hash(uint8_t input[IN_SIZE], uint8_t seed[IN_SIZE], uint8_t output[IN_SIZE]);

#ifdef __cplusplus
}
#endif

#endif // BITCOIN_CRYPTO_TENS_POW_TENS_HASH_H