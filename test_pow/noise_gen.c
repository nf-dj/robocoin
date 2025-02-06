#include <sodium.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NONCE_SIZE 32
#define NOISE_SIZE 256

// Single nonce noise computation
void compute_noise(const uint8_t *nonce, float *noise_out) {
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256(digest, nonce, NONCE_SIZE);
    
    for (int i = 0; i < NOISE_SIZE; i++) {
        noise_out[i] = (float)((digest[i % 32] >> (i % 8)) & 1);
    }
}

// Batch version
void compute_noise_batch(const uint8_t *nonces, float *noise_out, int batch_size) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        compute_noise(nonces + (b * NONCE_SIZE), noise_out + (b * NOISE_SIZE));
    }
}