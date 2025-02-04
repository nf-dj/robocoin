#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup>
using namespace metal;

#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64
#define SIMDGROUP_SIZE 32

kernel void tensor_hash_metal(
    device const uint8_t* nonces [[ buffer(0) ]],
    device const int8_t* noise_vectors [[ buffer(1) ]],    // Just 32 bytes per nonce
    device const int8_t* expand_mat [[ buffer(2) ]],
    device const int8_t* middle_mats [[ buffer(3) ]],
    device const int8_t* compress_mat [[ buffer(4) ]],
    device uint8_t* outputs [[ buffer(5) ]],
    uint thread_idx [[ thread_position_in_grid ]],
    uint local_idx [[ thread_position_in_threadgroup ]],
    uint grid_idx [[ threadgroup_position_in_grid ]]
) {
    const uint hash_idx = thread_idx;
    const uint offset = hash_idx * IN_SIZE;
    const uint simd_lane = thread_idx % SIMDGROUP_SIZE;
    const uint simd_group = thread_idx / SIMDGROUP_SIZE;
    
    thread int8_t state[HIDDEN];
    thread int8_t next_state[HIDDEN];
    
    // Expansion phase
    for (uint i = 0; i < HIDDEN; i++) {
        int8_t sum = noise_vectors[offset + (i & 31)];
        // SIMD group processes consecutive elements
        for (uint j = simd_lane; j < IN_SIZE; j += SIMDGROUP_SIZE) {
            sum += expand_mat[i * IN_SIZE + j] * as_type<int8_t>(nonces[offset + j]);
        }
        // Reduce within SIMD group
        sum = simd_shuffle_xor(sum, 16)
             + simd_shuffle_xor(sum, 8)
             + simd_shuffle_xor(sum, 4)
             + simd_shuffle_xor(sum, 2)
             + simd_shuffle_xor(sum, 1);
        state[i] = sum;
    }
    
    // Middle rounds
    for (uint r = 0; r < ROUNDS; r++) {
        const device int8_t* current_matrix = middle_mats + (r * HIDDEN * HIDDEN);
        
        for (uint i = 0; i < HIDDEN; i++) {
            int8_t sum = noise_vectors[offset + (i & 31)];
            // SIMD group processes consecutive elements
            for (uint j = simd_lane; j < HIDDEN; j += SIMDGROUP_SIZE) {
                sum += current_matrix[i * HIDDEN + j] * state[j];
            }
            // Reduce within SIMD group
            sum = simd_shuffle_xor(sum, 16)
                 + simd_shuffle_xor(sum, 8)
                 + simd_shuffle_xor(sum, 4)
                 + simd_shuffle_xor(sum, 2)
                 + simd_shuffle_xor(sum, 1);
            next_state[i] = sum;
        }
        
        // Swap state buffers
        for (uint i = 0; i < HIDDEN; i++) {
            state[i] = next_state[i];
        }
    }
    
    // Compression phase
    for (uint i = 0; i < IN_SIZE; i++) {
        int8_t sum = noise_vectors[offset + i];
        // SIMD group processes consecutive elements
        for (uint j = simd_lane; j < HIDDEN; j += SIMDGROUP_SIZE) {
            sum += compress_mat[i * HIDDEN + j] * state[j];
        }
        // Reduce within SIMD group
        sum = simd_shuffle_xor(sum, 16)
             + simd_shuffle_xor(sum, 8)
             + simd_shuffle_xor(sum, 4)
             + simd_shuffle_xor(sum, 2)
             + simd_shuffle_xor(sum, 1);
        outputs[offset + i] = as_type<uint8_t>(sum);
    }
}