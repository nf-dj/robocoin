#include "tens_hash.h"
#include <stdlib.h>
#include <string.h>
#include <crypto/chacha20.h>
#include <crypto/common.h>
#include <vector>
#include <span>
#include <cstring>

static uint8_t mod256(int32_t x) {
    return ((uint8_t)x) & 0xFF;
}

static void matrix_multiply(uint8_t **A, uint8_t *in, uint8_t *out, int8_t *e, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += (int32_t)A[i][j] * in[j];
        }
        sum += e[i];
        out[i] = mod256(sum);
    }
}

static void generate_all_matrices(PrecomputedMatrices* matrices, uint8_t seed[32]) {
   size_t total_size = (HIDDEN * IN_SIZE) + (ROUNDS * HIDDEN * HIDDEN) + (IN_SIZE * HIDDEN);
   uint8_t* all_data = (uint8_t*)malloc(total_size);
   if (!all_data) return;

   // Convert seed to ChaCha20's expected format 
   std::vector<std::byte> key_bytes(32);
   std::memcpy(key_bytes.data(), seed, 32);
   Span<const std::byte> key_span(key_bytes);
   
   // Create nonce
   ChaCha20::Nonce96 nonce{};  // Zero nonce
   uint32_t counter = 0;

   // Initialize ChaCha20 with key
   ChaCha20 chacha(key_span);
   
   // Convert output buffer to expected format
   std::vector<std::byte> output_bytes(total_size); 
   Span<std::byte> output_span(output_bytes);
   
   // Generate keystream
   chacha.Seek(nonce, counter);
   chacha.Keystream(output_span);

   // Copy generated data to all_data
   std::memcpy(all_data, output_bytes.data(), total_size);

   uint8_t *curr_pos = all_data;
   
   // Copy into expand_mat
   for (int i = 0; i < HIDDEN; i++) {
       memcpy(matrices->expand_mat[i], curr_pos, IN_SIZE);
       curr_pos += IN_SIZE;
   }
   
   // Copy into middle_mats 
   for (int r = 0; r < ROUNDS; r++) {
       for (int i = 0; i < HIDDEN; i++) {
           memcpy(matrices->middle_mats[r][i], curr_pos, HIDDEN);
           curr_pos += HIDDEN;
       }
   }
   
   // Copy into compress_mat
   for (int i = 0; i < IN_SIZE; i++) {
       memcpy(matrices->compress_mat[i], curr_pos, HIDDEN);
       curr_pos += HIDDEN;
   }
   
   free(all_data);
}

static void generate_all_noise(int8_t *noise_buffer, uint8_t input[32], int total_size) {
    // Similar changes for noise generation
    std::vector<std::byte> key_bytes(32);
    std::memcpy(key_bytes.data(), input, 32);
    Span<const std::byte> key_span(key_bytes);
    
    ChaCha20::Nonce96 nonce{};
    uint32_t counter = 0;

    ChaCha20 chacha(key_span);
    
    std::vector<std::byte> output_bytes(total_size);
    Span<std::byte> output_span(output_bytes);
    
    chacha.Seek(nonce, counter);
    chacha.Keystream(output_span);
    
    std::memcpy(noise_buffer, output_bytes.data(), total_size);
}

HashBuffers* init_hash_buffers() {
    HashBuffers* buffers = static_cast<HashBuffers*>(malloc(sizeof(HashBuffers)));
    if (!buffers) return nullptr;

    buffers->state = static_cast<uint8_t*>(calloc(HIDDEN, sizeof(uint8_t)));
    buffers->next_state = static_cast<uint8_t*>(calloc(HIDDEN, sizeof(uint8_t)));

    int total_noise_size = HIDDEN + (ROUNDS * HIDDEN) + IN_SIZE;
    buffers->noise = static_cast<int8_t*>(malloc(total_noise_size * sizeof(int8_t)));

    if (!buffers->state || !buffers->next_state || !buffers->noise) {
        free(buffers->state);
        free(buffers->next_state);
        free(buffers->noise);
        free(buffers);
        return nullptr;
    }

    return buffers;
}

void free_hash_buffers(HashBuffers* buffers) {
    if (buffers) {
        free(buffers->state);
        free(buffers->next_state);
        free(buffers->noise);
        free(buffers);
    }
}

PrecomputedMatrices* precompute_matrices(uint8_t seed[32]) {
    PrecomputedMatrices* matrices = static_cast<PrecomputedMatrices*>(malloc(sizeof(PrecomputedMatrices)));
    if (!matrices) return nullptr;
    
    matrices->expand_mat = static_cast<uint8_t**>(malloc(HIDDEN * sizeof(uint8_t*)));
    if (!matrices->expand_mat) {
        free(matrices);
        return nullptr;
    }
    
    for (int i = 0; i < HIDDEN; i++) {
        matrices->expand_mat[i] = static_cast<uint8_t*>(malloc(IN_SIZE * sizeof(uint8_t)));
        if (!matrices->expand_mat[i]) {
            for (int j = 0; j < i; j++) free(matrices->expand_mat[j]);
            free(matrices->expand_mat);
            free(matrices);
            return nullptr;
        }
    }

    for (int r = 0; r < ROUNDS; r++) {
        matrices->middle_mats[r] = static_cast<uint8_t**>(malloc(HIDDEN * sizeof(uint8_t*)));
        if (!matrices->middle_mats[r]) {
            for (int i = 0; i < HIDDEN; i++) free(matrices->expand_mat[i]);
            free(matrices->expand_mat);
            for (int j = 0; j < r; j++) {
                for (int i = 0; i < HIDDEN; i++) free(matrices->middle_mats[j][i]);
                free(matrices->middle_mats[j]);
            }
            free(matrices);
            return nullptr;
        }
        
        for (int i = 0; i < HIDDEN; i++) {
            matrices->middle_mats[r][i] = static_cast<uint8_t*>(malloc(HIDDEN * sizeof(uint8_t)));
            if (!matrices->middle_mats[r][i]) {
                for (int j = 0; j < i; j++) free(matrices->middle_mats[r][j]);
                free(matrices->middle_mats[r]);
                for (int j = 0; j < r; j++) {
                    for (int k = 0; k < HIDDEN; k++) free(matrices->middle_mats[j][k]);
                    free(matrices->middle_mats[j]);
                }
                for (int j = 0; j < HIDDEN; j++) free(matrices->expand_mat[j]);
                free(matrices->expand_mat);
                free(matrices);
                return nullptr;
            }
        }
    }

    matrices->compress_mat = static_cast<uint8_t**>(malloc(IN_SIZE * sizeof(uint8_t*)));
    if (!matrices->compress_mat) {
        for (int r = 0; r < ROUNDS; r++) {
            for (int i = 0; i < HIDDEN; i++) free(matrices->middle_mats[r][i]);
            free(matrices->middle_mats[r]);
        }
        for (int i = 0; i < HIDDEN; i++) free(matrices->expand_mat[i]);
        free(matrices->expand_mat);
        free(matrices);
        return nullptr;
    }
    
    for (int i = 0; i < IN_SIZE; i++) {
        matrices->compress_mat[i] = static_cast<uint8_t*>(malloc(HIDDEN * sizeof(uint8_t)));
        if (!matrices->compress_mat[i]) {
            for (int j = 0; j < i; j++) free(matrices->compress_mat[j]);
            free(matrices->compress_mat);
            for (int r = 0; r < ROUNDS; r++) {
                for (int j = 0; j < HIDDEN; j++) free(matrices->middle_mats[r][j]);
                free(matrices->middle_mats[r]);
            }
            for (int j = 0; j < HIDDEN; j++) free(matrices->expand_mat[j]);
            free(matrices->expand_mat);
            free(matrices);
            return nullptr;
        }
    }

    generate_all_matrices(matrices, seed);
    return matrices;
}

void free_matrices(PrecomputedMatrices* matrices) {
    if (matrices) {
        if (matrices->expand_mat) {
            for (int i = 0; i < HIDDEN; i++) free(matrices->expand_mat[i]);
            free(matrices->expand_mat);
        }

        for (int r = 0; r < ROUNDS; r++) {
            if (matrices->middle_mats[r]) {
                for (int i = 0; i < HIDDEN; i++) free(matrices->middle_mats[r][i]);
                free(matrices->middle_mats[r]);
            }
        }

        if (matrices->compress_mat) {
            for (int i = 0; i < IN_SIZE; i++) free(matrices->compress_mat[i]);
            free(matrices->compress_mat);
        }

        free(matrices);
    }
}

void tens_hash_precomputed(uint8_t input[IN_SIZE], PrecomputedMatrices* matrices,
                          HashBuffers* buffers, uint8_t output[IN_SIZE]) {
    int total_noise_size = HIDDEN + (ROUNDS * HIDDEN) + IN_SIZE;
    generate_all_noise(buffers->noise, input, total_noise_size);

    int8_t *expand_noise = buffers->noise;
    int8_t *middle_noise = buffers->noise + HIDDEN;
    int8_t *compress_noise = buffers->noise + HIDDEN + (ROUNDS * HIDDEN);

    matrix_multiply(matrices->expand_mat, input, buffers->state, expand_noise, HIDDEN, IN_SIZE);

    for (uint32_t round = 0; round < ROUNDS; round++) {
        matrix_multiply(matrices->middle_mats[round], buffers->state, buffers->next_state,
                       middle_noise + (round * HIDDEN), HIDDEN, HIDDEN);

        uint8_t *temp = buffers->state;
        buffers->state = buffers->next_state;
        buffers->next_state = temp;
    }

    matrix_multiply(matrices->compress_mat, buffers->state, output, compress_noise, IN_SIZE, HIDDEN);
}

void tens_hash(uint8_t input[IN_SIZE], uint8_t seed[IN_SIZE], uint8_t output[IN_SIZE]) {
    PrecomputedMatrices* matrices = precompute_matrices(seed);
    HashBuffers* buffers = init_hash_buffers();
    
    if (!matrices || !buffers) return;

    tens_hash_precomputed(input, matrices, buffers, output);
    
    free_matrices(matrices);
    free_hash_buffers(buffers);
}
