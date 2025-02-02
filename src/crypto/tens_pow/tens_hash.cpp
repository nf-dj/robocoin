#include "tens_hash.h"
#include <stdlib.h>
#include <string.h>
#include <crypto/chacha20.h>
#include <crypto/common.h>
#include <crypto/sha256.h>
#include <vector>
#include <span>
#include <cstring>
#include <logging.h>
#include <util/strencodings.h>

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

static void generate_all_matrices(TensHashContext* ctx, uint8_t seed[32]) {
    size_t total_size = (TENS_HIDDEN * TENS_IN_SIZE) + (TENS_ROUNDS * TENS_HIDDEN * TENS_HIDDEN) + 
                       (TENS_IN_SIZE * TENS_HIDDEN);
    uint8_t* all_data = (uint8_t*)malloc(total_size);
    if (!all_data) return;

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
    std::memcpy(all_data, output_bytes.data(), total_size);

    uint8_t *curr_pos = all_data;
    
    for (int i = 0; i < TENS_HIDDEN; i++) {
        memcpy(ctx->expand_mat[i], curr_pos, TENS_IN_SIZE);
        curr_pos += TENS_IN_SIZE;
    }
    
    for (int r = 0; r < TENS_ROUNDS; r++) {
        for (int i = 0; i < TENS_HIDDEN; i++) {
            memcpy(ctx->middle_mats[r][i], curr_pos, TENS_HIDDEN);
            curr_pos += TENS_HIDDEN;
        }
    }
    
    for (int i = 0; i < TENS_IN_SIZE; i++) {
        memcpy(ctx->compress_mat[i], curr_pos, TENS_HIDDEN);
        curr_pos += TENS_HIDDEN;
    }
    
    free(all_data);
}

static void generate_all_noise(int8_t *noise_buffer, uint8_t input[32], int total_size) {
    CSHA256 sha256;
    unsigned char digest[CSHA256::OUTPUT_SIZE];
    
    sha256.Write(input, 32);
    sha256.Finalize(digest);
    
    for (int i = 0; i < total_size; i++) {
        noise_buffer[i] = digest[i % CSHA256::OUTPUT_SIZE];
    }
}

TensHashContext* tens_hash_init(uint8_t seed[32]) {
    TensHashContext* ctx = static_cast<TensHashContext*>(malloc(sizeof(TensHashContext)));
    if (!ctx) return nullptr;
    
    ctx->state = static_cast<uint8_t*>(calloc(TENS_HIDDEN, sizeof(uint8_t)));
    ctx->next_state = static_cast<uint8_t*>(calloc(TENS_HIDDEN, sizeof(uint8_t)));
    int total_noise_size = TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN) + TENS_IN_SIZE;
    ctx->noise = static_cast<int8_t*>(malloc(total_noise_size * sizeof(int8_t)));

    if (!ctx->state || !ctx->next_state || !ctx->noise) {
        if (ctx->state) free(ctx->state);
        if (ctx->next_state) free(ctx->next_state);
        if (ctx->noise) free(ctx->noise);
        free(ctx);
        return nullptr;
    }

    ctx->expand_mat = static_cast<uint8_t**>(malloc(TENS_HIDDEN * sizeof(uint8_t*)));
    if (!ctx->expand_mat) {
        free(ctx->state);
        free(ctx->next_state);
        free(ctx->noise);
        free(ctx);
        return nullptr;
    }

    for (int i = 0; i < TENS_HIDDEN; i++) {
        ctx->expand_mat[i] = static_cast<uint8_t*>(malloc(TENS_IN_SIZE * sizeof(uint8_t)));
        if (!ctx->expand_mat[i]) {
            for (int j = 0; j < i; j++) free(ctx->expand_mat[j]);
            free(ctx->expand_mat);
            free(ctx->state);
            free(ctx->next_state);
            free(ctx->noise);
            free(ctx);
            return nullptr;
        }
    }

    for (int r = 0; r < TENS_ROUNDS; r++) {
        ctx->middle_mats[r] = static_cast<uint8_t**>(malloc(TENS_HIDDEN * sizeof(uint8_t*)));
        if (!ctx->middle_mats[r]) {
            for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->expand_mat[i]);
            free(ctx->expand_mat);
            for (int j = 0; j < r; j++) {
                for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->middle_mats[j][i]);
                free(ctx->middle_mats[j]);
            }
            free(ctx->state);
            free(ctx->next_state);
            free(ctx->noise);
            free(ctx);
            return nullptr;
        }

        for (int i = 0; i < TENS_HIDDEN; i++) {
            ctx->middle_mats[r][i] = static_cast<uint8_t*>(malloc(TENS_HIDDEN * sizeof(uint8_t)));
            if (!ctx->middle_mats[r][i]) {
                for (int j = 0; j < i; j++) free(ctx->middle_mats[r][j]);
                free(ctx->middle_mats[r]);
                for (int j = 0; j < r; j++) {
                    for (int k = 0; k < TENS_HIDDEN; k++) free(ctx->middle_mats[j][k]);
                    free(ctx->middle_mats[j]);
                }
                for (int j = 0; j < TENS_HIDDEN; j++) free(ctx->expand_mat[j]);
                free(ctx->expand_mat);
                free(ctx->state);
                free(ctx->next_state);
                free(ctx->noise);
                free(ctx);
                return nullptr;
            }
        }
    }

    ctx->compress_mat = static_cast<uint8_t**>(malloc(TENS_IN_SIZE * sizeof(uint8_t*)));
    if (!ctx->compress_mat) {
        for (int r = 0; r < TENS_ROUNDS; r++) {
            for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->middle_mats[r][i]);
            free(ctx->middle_mats[r]);
        }
        for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->expand_mat[i]);
        free(ctx->expand_mat);
        free(ctx->state);
        free(ctx->next_state);
        free(ctx->noise);
        free(ctx);
        return nullptr;
    }

    for (int i = 0; i < TENS_IN_SIZE; i++) {
        ctx->compress_mat[i] = static_cast<uint8_t*>(malloc(TENS_HIDDEN * sizeof(uint8_t)));
        if (!ctx->compress_mat[i]) {
            for (int j = 0; j < i; j++) free(ctx->compress_mat[j]);
            free(ctx->compress_mat);
            for (int r = 0; r < TENS_ROUNDS; r++) {
                for (int j = 0; j < TENS_HIDDEN; j++) free(ctx->middle_mats[r][j]);
                free(ctx->middle_mats[r]);
            }
            for (int j = 0; j < TENS_HIDDEN; j++) free(ctx->expand_mat[j]);
            free(ctx->expand_mat);
            free(ctx->state);
            free(ctx->next_state);
            free(ctx->noise);
            free(ctx);
            return nullptr;
        }
    }

    generate_all_matrices(ctx, seed);
    return ctx;
}

void tens_hash_free(TensHashContext* ctx) {
    if (ctx) {
        if (ctx->state) free(ctx->state);
        if (ctx->next_state) free(ctx->next_state);
        if (ctx->noise) free(ctx->noise);

        if (ctx->expand_mat) {
            for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->expand_mat[i]);
            free(ctx->expand_mat);
        }

        if (ctx->middle_mats) {
            for (int r = 0; r < TENS_ROUNDS; r++) {
                if (ctx->middle_mats[r]) {
                    for (int i = 0; i < TENS_HIDDEN; i++) free(ctx->middle_mats[r][i]);
                    free(ctx->middle_mats[r]);
                }
            }
        }

        if (ctx->compress_mat) {
            for (int i = 0; i < TENS_IN_SIZE; i++) free(ctx->compress_mat[i]);
            free(ctx->compress_mat);
        }

        free(ctx);
    }
}

void tens_hash_precomputed(uint8_t input[TENS_IN_SIZE], TensHashContext* ctx, uint8_t output[TENS_IN_SIZE]) {
    int total_noise_size = TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN) + TENS_IN_SIZE;
    generate_all_noise(ctx->noise, input, total_noise_size);

    int8_t *expand_noise = ctx->noise;
    int8_t *middle_noise = ctx->noise + TENS_HIDDEN;
    int8_t *compress_noise = ctx->noise + TENS_HIDDEN + (TENS_ROUNDS * TENS_HIDDEN);

    matrix_multiply(ctx->expand_mat, input, ctx->state, expand_noise, TENS_HIDDEN, TENS_IN_SIZE);

    for (uint32_t round = 0; round < TENS_ROUNDS; round++) {
        matrix_multiply(ctx->middle_mats[round], ctx->state, ctx->next_state,
                       middle_noise + (round * TENS_HIDDEN), TENS_HIDDEN, TENS_HIDDEN);

        uint8_t *temp = ctx->state;
        ctx->state = ctx->next_state;
        ctx->next_state = temp;
    }

    matrix_multiply(ctx->compress_mat, ctx->state, output, compress_noise, TENS_IN_SIZE, TENS_HIDDEN);
}

void tens_hash(uint8_t input[TENS_IN_SIZE], uint8_t seed[TENS_IN_SIZE], uint8_t output[TENS_IN_SIZE]) {
    TensHashContext* ctx = tens_hash_init(seed);
    if (!ctx) return;

    tens_hash_precomputed(input, ctx, output);
    tens_hash_free(ctx);
}