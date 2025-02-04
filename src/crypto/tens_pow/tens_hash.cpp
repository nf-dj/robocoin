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

static inline const int8_t* middle_mat_elem(const int8_t* mat, int round, int row) {
    return mat + (round * TENS_HIDDEN * TENS_HIDDEN) + (row * TENS_HIDDEN);
}

static void matrix_multiply_mod2(const int8_t* A, const uint8_t *in, uint8_t *out, const int8_t *noise, 
                               int rows, int cols, int row_stride) {
    for (int i = 0; i < rows; i++) {
        const int8_t* row = A + (i * row_stride);
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * in[j];
        }
        sum += noise[i];
        out[i] = sum & 1;
    }
}

static void generate_all_matrices(TensHashContext* ctx, const uint8_t seed[32]) {
    if (!ctx || !seed) return;

    size_t total_size = TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN;
    
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
    
    const uint8_t* curr_pos = reinterpret_cast<const uint8_t*>(output_bytes.data());
    
    for (int r = 0; r < TENS_ROUNDS; r++) {
        for (int i = 0; i < TENS_HIDDEN; i++) {
            for (int j = 0; j < TENS_HIDDEN; j++) {
                uint8_t val = curr_pos[r * TENS_HIDDEN * TENS_HIDDEN + i * TENS_HIDDEN + j] % 3;
                ctx->middle_mats[r * TENS_HIDDEN * TENS_HIDDEN + i * TENS_HIDDEN + j] = val - 1;
            }
        }
    }
}

static void generate_noise(int8_t *noise_buffer, const uint8_t input[32], int total_size) {
    if (!noise_buffer || !input) return;

    unsigned char digest[CSHA256::OUTPUT_SIZE];
    CSHA256 sha256;
    sha256.Write(input, 32);
    sha256.Finalize(digest);
    
    for (int i = 0; i < total_size; i++) {
        noise_buffer[i] = (digest[i % 32] >> (i % 8)) & 1;
    }
}

static bool alloc_context_buffers(TensHashContext* ctx) {
    if (!ctx) return false;

    ctx->middle_mats = static_cast<int8_t*>(malloc(TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN));
    ctx->state = static_cast<uint8_t*>(calloc(TENS_HIDDEN, sizeof(uint8_t)));
    ctx->next_state = static_cast<uint8_t*>(calloc(TENS_HIDDEN, sizeof(uint8_t)));
    ctx->noise = static_cast<int8_t*>(malloc(TENS_ROUNDS * TENS_HIDDEN * sizeof(int8_t)));

    if (!ctx->middle_mats || !ctx->state || !ctx->next_state || !ctx->noise) {
        return false;
    }

    return true;
}

TensHashContext* tens_hash_init(const uint8_t seed[32]) {
    if (!seed) return nullptr;

    TensHashContext* ctx = static_cast<TensHashContext*>(malloc(sizeof(TensHashContext)));
    if (!ctx) return nullptr;
    
    memset(ctx, 0, sizeof(TensHashContext));

    if (!alloc_context_buffers(ctx)) {
        tens_hash_free(ctx);
        return nullptr;
    }

    generate_all_matrices(ctx, seed);
    return ctx;
}

void tens_hash_free(TensHashContext* ctx) {
    if (ctx) {
        free(ctx->middle_mats);
        free(ctx->state);
        free(ctx->next_state);
        free(ctx->noise);
        free(ctx);
    }
}

void tens_hash_precomputed(const uint8_t input[TENS_IN_SIZE], TensHashContext* ctx, uint8_t output[TENS_IN_SIZE]) {
    if (!input || !ctx || !output) return;

    // Convert input bytes to bits
    for (int i = 0; i < TENS_IN_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            ctx->state[i*8 + j] = (input[i] >> j) & 1;
        }
    }

    generate_noise(ctx->noise, input, TENS_ROUNDS * TENS_HIDDEN);

    for (uint32_t round = 0; round < TENS_ROUNDS; round++) {
        matrix_multiply_mod2(middle_mat_elem(ctx->middle_mats, round, 0), 
                           ctx->state, ctx->next_state,
                           ctx->noise + (round * TENS_HIDDEN), 
                           TENS_HIDDEN, TENS_HIDDEN, TENS_HIDDEN);

        uint8_t *temp = ctx->state;
        ctx->state = ctx->next_state;
        ctx->next_state = temp;
    }

    // Convert bits back to bytes
    memset(output, 0, TENS_IN_SIZE);
    for (int i = 0; i < TENS_IN_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            output[i] |= ctx->state[i*8 + j] << j;
        }
    }
}

void tens_hash(const uint8_t input[TENS_IN_SIZE], const uint8_t seed[TENS_IN_SIZE], uint8_t output[TENS_IN_SIZE]) {
    if (!input || !seed || !output) return;

    TensHashContext* ctx = tens_hash_init(seed);
    if (!ctx) return;

    tens_hash_precomputed(input, ctx, output);
    tens_hash_free(ctx);
}