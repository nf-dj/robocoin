#include <metal_stdlib>
using namespace metal;

#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64

kernel void tensor_hash_metal(
    device const uint8_t* nonces [[ buffer(0) ]],
    device const int8_t* noise_vectors [[ buffer(1) ]],
    device const int8_t* expand_mat [[ buffer(2) ]],
    device const int8_t* middle_mats [[ buffer(3) ]],
    device const int8_t* compress_mat [[ buffer(4) ]],
    device uint8_t* outputs [[ buffer(5) ]],
    uint thread_idx [[ thread_position_in_grid ]]
) {
    // Get this thread's nonce/noise offsets
    const uint offset = thread_idx * IN_SIZE;
    device const int8_t* noise = noise_vectors + offset;
    
    // Allocate thread-local state
    thread int8_t state[HIDDEN];
    thread int8_t next_state[HIDDEN];
    
    // Expansion phase
    for (int i = 0; i < HIDDEN; i++) {
        int8_t sum = noise[i & 31];
        for (int j = 0; j < IN_SIZE; j++) {
            sum += expand_mat[i * IN_SIZE + j] * as_type<int8_t>(nonces[offset + j]);
        }
        state[i] = sum;
    }
    
    // Middle rounds
    for (uint r = 0; r < ROUNDS; r++) {
        const device int8_t* current_matrix = middle_mats + (r * HIDDEN * HIDDEN);
        
        for (int i = 0; i < HIDDEN; i++) {
            int8_t sum = noise[i & 31];
            for (int j = 0; j < HIDDEN; j++) {
                sum += current_matrix[i * HIDDEN + j] * state[j];
            }
            next_state[i] = sum;
        }
        
        // Swap state and next_state
        for (int i = 0; i < HIDDEN; i++) {
            state[i] = next_state[i];
        }
    }
    
    // Compression phase
    for (int i = 0; i < IN_SIZE; i++) {
        int8_t sum = noise[i];  // already within 32 bytes
        for (int j = 0; j < HIDDEN; j++) {
            sum += compress_mat[i * HIDDEN + j] * state[j];
        }
        outputs[offset + i] = as_type<uint8_t>(sum);
    }
}
