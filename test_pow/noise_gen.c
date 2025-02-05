#include <sodium.h>
#include <stdint.h>
#include <stdlib.h>

#define NONCE_SIZE 32
#define NOISE_SIZE 256

// Compute noise for one nonce.
// noise_out must be an array of NOISE_SIZE floats.
void compute_noise(const uint8_t *nonce, float *noise_out) {
    unsigned char digest[crypto_hash_sha256_BYTES];
    // Compute SHA256 using libsodium.
    crypto_hash_sha256(digest, nonce, NONCE_SIZE);
    for (int j = 0; j < NOISE_SIZE; j++) {
        noise_out[j] = (float)((digest[j % crypto_hash_sha256_BYTES] >> (j % 8)) & 1);
    }
}

// Batch version: nonces is an array of (batch_size * NONCE_SIZE) bytes,
// noise_out is an array of (batch_size * NOISE_SIZE) floats.
void compute_noise_batch(const uint8_t *nonces, float *noise_out, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        compute_noise(nonces + i * NONCE_SIZE, noise_out + i * NOISE_SIZE);
    }
}

