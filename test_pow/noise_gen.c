#include <sodium.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE 32
#define VECTOR_SIZE 256
#define NOISE_SIZE 256

void compute_binary_and_noise_vectors(const uint8_t *input, float *binary_out, float *noise_out) {
    unsigned char first_hash[crypto_hash_sha256_BYTES];
    unsigned char second_hash[crypto_hash_sha256_BYTES];
    
    // First SHA256 for binary vector
    crypto_hash_sha256(first_hash, input, INPUT_SIZE);
    
    // Convert first hash to binary vector
    for (int i = 0; i < VECTOR_SIZE; i++) {
        binary_out[i] = (float)((first_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
    
    // Second SHA256 for noise
    crypto_hash_sha256(second_hash, first_hash, crypto_hash_sha256_BYTES);
    
    // Convert second hash to noise vector
    for (int i = 0; i < NOISE_SIZE; i++) {
        noise_out[i] = (float)((second_hash[i % 32] >> (i % 8)) & 1);
    }
}

void compute_binary_and_noise_batch(const uint8_t *inputs, float *binary_out, float *noise_out, int batch_size) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        compute_binary_and_noise_vectors(
            inputs + (b * INPUT_SIZE),
            binary_out + (b * VECTOR_SIZE),
            noise_out + (b * NOISE_SIZE)
        );
    }
}
