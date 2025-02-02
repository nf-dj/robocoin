#include "tens_hash.h"
#include <stdlib.h>
#include <string.h>
#include <crypto/chacha20.h>
#include <crypto/common.h>
#include <crypto/sha256.h>
#include <vector>
#include <span>
#include <logging.h>
#include <util/strencodings.h>

// Helper function to access middle matrix elements in continuous memory
static inline const int8_t* middle_mat_elem(const int8_t* mat, int round, int row) {
    return mat + (round * TENS_HIDDEN * TENS_HIDDEN) + (row * TENS_HIDDEN);
}

static void matrix_multiply(const int8_t* A, const int8_t *in, int8_t *out, const int8_t *e, 
                          int rows, int cols, int row_stride) {
    for (int i = 0; i < rows; i++) {
        const int8_t* row = A + (i * row_stride);
        int8_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * in[j];
        }
        out[i] = sum + e[i];
    }
}

static void generate_all_matrices(TensHashContext* ctx, const uint8_t seed[32]) {
    if (!ctx || !seed) return;

    size_t total_size = (TENS_HIDDEN * TENS_IN_SIZE) + 
                       (TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN) + 
                       (TENS_IN_SIZE * TENS_HIDDEN);

    std::vector<std::byte> key_bytes(32);
    std::memcpy(key_bytes.data(), seed, 32);
    Span<const std::byte> key_span(key_bytes);
    
    ChaCha20::Nonce96 nonce{};
    uint32_t counter = 0;
    ChaCha20 chacha(key_span);
    
    std::vector<std::byte> output_bytes(total_size); 
    Span<std::byte> output_span(output_bytes);
    chacha.Seek(nonce, counter);
    chacha.Keystream(output_span);
    
    const int8_t* curr_pos = reinterpret_cast<const int8_t*>(output_bytes.data());
    
    memcpy(ctx->expand_mat, curr_pos, TENS_HIDDEN * TENS_IN_SIZE);
    curr_pos += TENS_HIDDEN * TENS_IN_SIZE;
    
    memcpy(ctx->middle_mats, curr_pos, TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN);
    curr_pos += TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN;
    
    memcpy(ctx->compress_mat, curr_pos, TENS_IN_SIZE * TENS_HIDDEN);
}

static void generate_all_noise(int8_t *noise_buffer, const uint8_t input[32], int total_size) {
    if (!noise_buffer || !input) return;

    CSHA256 sha256;
    unsigned char digest[CSHA256::OUTPUT_SIZE];
    
    sha256.Write(input, 32);
    sha256.Finalize(digest);
    
    for (int i = 0; i < total_size; i++) {
        noise_buffer[i] = digest[i % CSHA256::OUTPUT_SIZE];
    }
}

static bool alloc_context_buffers(TensHashContext* ctx) {
    if (!ctx) return false;

    // Allocate all matrices in contiguous blocks
    ctx->expand_mat = static_cast<int8_t*>(malloc(TENS_HIDDEN * TENS_IN_SIZE));
    ctx->middle_mats = static_cast<int8_t*>(malloc(TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN));
    ctx->compress_mat = static_cast<int8_t*>(malloc(TENS_IN_SIZE * TENS_HIDDEN));
    
    // Allocate buffers
    ctx->state = static_cast<int8_t*>(calloc(TENS_HIDDEN, sizeof(int8_t)));
    ctx->next_state = static_cast<int8_t*>(calloc(TENS_HIDDEN, sizeof(int8_t)));
    int total_noise_size = TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN) + TENS_IN_SIZE;
    ctx->noise = static_cast<int8_t*>(malloc(total_noise_size * sizeof(int8_t)));

    if (!ctx->expand_mat || !ctx->middle_mats || !ctx->compress_mat || 
        !ctx->state || !ctx->next_state || !ctx->noise) {
        return false;
    }

    return true;
}

TensHashContext* tens_hash_init(const uint8_t seed[32]) {
    if (!seed) return nullptr;

    TensHashContext* ctx = static_cast<TensHashContext*>(malloc(sizeof(TensHashContext)));
    if (!ctx) return nullptr;
    
    memset(ctx, 0, sizeof(TensHashContext));  // Initialize all pointers to null

    if (!alloc_context_buffers(ctx)) {
        tens_hash_free(ctx);
        return nullptr;
    }

    generate_all_matrices(ctx, seed);
    return ctx;
}

void tens_hash_free(TensHashContext* ctx) {
    if (ctx) {
        free(ctx->expand_mat);
        free(ctx->middle_mats);
        free(ctx->compress_mat);
        free(ctx->state);
        free(ctx->next_state);
        free(ctx->noise);
        free(ctx);
    }
}

void tens_hash_precomputed(const uint8_t input[TENS_IN_SIZE], TensHashContext* ctx, uint8_t output[TENS_IN_SIZE]) {
    if (!input || !ctx || !output) return;

    int total_noise_size = TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN) + TENS_IN_SIZE;
    generate_all_noise(ctx->noise, input, total_noise_size);

    int8_t *expand_noise = ctx->noise;
    int8_t *middle_noise = ctx->noise + TENS_HIDDEN;
    int8_t *compress_noise = ctx->noise + TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN);

    matrix_multiply(ctx->expand_mat, (int8_t*)input, ctx->state, expand_noise, 
                   TENS_HIDDEN, TENS_IN_SIZE, TENS_IN_SIZE);

    for (uint32_t round = 0; round < TENS_ROUNDS; round++) {
        matrix_multiply(middle_mat_elem(ctx->middle_mats, round, 0), 
                       ctx->state, ctx->next_state,
                       middle_noise + (round * TENS_HIDDEN), 
                       TENS_HIDDEN, TENS_HIDDEN, TENS_HIDDEN);

        int8_t *temp = ctx->state;
        ctx->state = ctx->next_state;
        ctx->next_state = temp;
    }

    matrix_multiply(ctx->compress_mat, ctx->state, (int8_t*)output, compress_noise, 
                   TENS_IN_SIZE, TENS_HIDDEN, TENS_HIDDEN);
}

void tens_hash(const uint8_t input[TENS_IN_SIZE], const uint8_t seed[TENS_IN_SIZE], uint8_t output[TENS_IN_SIZE]) {
    if (!input || !seed || !output) return;

    TensHashContext* ctx = tens_hash_init(seed);
    if (!ctx) return;

    tens_hash_precomputed(input, ctx, output);
    tens_hash_free(ctx);
}
