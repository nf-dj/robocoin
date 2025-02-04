#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64
#define TILE_SIZE 8

kernel void tensor_hash_metal(
    device const uint8_t* nonces [[ buffer(0) ]],
    device const int8_t* noise_vectors [[ buffer(1) ]],
    device const float* expand_mat [[ buffer(2) ]],
    device const float* middle_mats [[ buffer(3) ]],
    device const float* compress_mat [[ buffer(4) ]],
    device uint8_t* outputs [[ buffer(5) ]],
    uint thread_idx [[ thread_position_in_grid ]],
    uint local_idx [[ thread_position_in_threadgroup ]],
    uint grid_idx [[ threadgroup_position_in_grid ]]
) {
    const uint hash_idx = thread_idx;
    const uint offset = hash_idx * IN_SIZE;

    // Use float arrays for state
    thread float state[HIDDEN];
    thread float next_state[HIDDEN];
    
    // Initialize states
    for (uint i = 0; i < HIDDEN; i++) {
        state[i] = 0.0f;
        next_state[i] = 0.0f;
    }
    
    // Thread local storage for matrix data
    simdgroup_float8x8 mat_a;
    simdgroup_float8x8 mat_b;
    simdgroup_float8x8 mat_c;
    
    // Expansion phase
    for (uint tile_i = 0; tile_i < HIDDEN; tile_i += TILE_SIZE) {
        for (uint tile_j = 0; tile_j < IN_SIZE; tile_j += TILE_SIZE) {
            // Load matrices
            for (uint i = 0; i < TILE_SIZE; i++) {
                for (uint j = 0; j < TILE_SIZE; j++) {
                    if ((tile_i + i) < HIDDEN && (tile_j + j) < IN_SIZE) {
                        mat_a.thread_elements()[i * TILE_SIZE + j] = expand_mat[(tile_i + i) * IN_SIZE + (tile_j + j)];
                        mat_b.thread_elements()[i * TILE_SIZE + j] = float(as_type<int8_t>(nonces[offset + tile_j + j]));
                    } else {
                        mat_a.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                        mat_b.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                    }
                }
            }

            // Multiply using hardware acceleration
            simdgroup_multiply(mat_c, mat_a, mat_b);

            // Extract results
            for (uint i = 0; i < TILE_SIZE && (tile_i + i) < HIDDEN; i++) {
                float sum = 0.0f;
                for (uint j = 0; j < TILE_SIZE; j++) {
                    sum += mat_c.thread_elements()[i * TILE_SIZE + j];
                }
                state[tile_i + i] += sum;
            }
        }
    }
    
    // Add noise to expansion result
    for (uint i = 0; i < HIDDEN; i++) {
        state[i] += float(noise_vectors[offset + (i & 31)]);
    }
    
    // Middle rounds
    for (uint r = 0; r < ROUNDS; r++) {
        const device float* current_matrix = middle_mats + (r * HIDDEN * HIDDEN);
        
        for (uint tile_i = 0; tile_i < HIDDEN; tile_i += TILE_SIZE) {
            for (uint tile_j = 0; tile_j < HIDDEN; tile_j += TILE_SIZE) {
                // Load matrices
                for (uint i = 0; i < TILE_SIZE; i++) {
                    for (uint j = 0; j < TILE_SIZE; j++) {
                        if ((tile_i + i) < HIDDEN && (tile_j + j) < HIDDEN) {
                            mat_a.thread_elements()[i * TILE_SIZE + j] = current_matrix[(tile_i + i) * HIDDEN + (tile_j + j)];
                            mat_b.thread_elements()[i * TILE_SIZE + j] = state[tile_j + j];
                        } else {
                            mat_a.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                            mat_b.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                        }
                    }
                }

                // Multiply using hardware acceleration
                simdgroup_multiply(mat_c, mat_a, mat_b);

                // Extract results
                for (uint i = 0; i < TILE_SIZE && (tile_i + i) < HIDDEN; i++) {
                    float sum = 0.0f;
                    for (uint j = 0; j < TILE_SIZE; j++) {
                        sum += mat_c.thread_elements()[i * TILE_SIZE + j];
                    }
                    next_state[tile_i + i] += sum;
                }
            }
        }
        
        // Add noise and swap buffers
        for (uint i = 0; i < HIDDEN; i++) {
            next_state[i] += float(noise_vectors[offset + (i & 31)]);
            state[i] = next_state[i];
            next_state[i] = 0.0f;
        }
    }
    
    // Compression phase
    for (uint tile_i = 0; tile_i < IN_SIZE; tile_i += TILE_SIZE) {
        for (uint tile_j = 0; tile_j < HIDDEN; tile_j += TILE_SIZE) {
            // Load matrices
            for (uint i = 0; i < TILE_SIZE; i++) {
                for (uint j = 0; j < TILE_SIZE; j++) {
                    if ((tile_i + i) < IN_SIZE && (tile_j + j) < HIDDEN) {
                        mat_a.thread_elements()[i * TILE_SIZE + j] = compress_mat[(tile_i + i) * HIDDEN + (tile_j + j)];
                        mat_b.thread_elements()[i * TILE_SIZE + j] = state[tile_j + j];
                    } else {
                        mat_a.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                        mat_b.thread_elements()[i * TILE_SIZE + j] = 0.0f;
                    }
                }
            }

            // Multiply using hardware acceleration
            simdgroup_multiply(mat_c, mat_a, mat_b);

            // Store final results
            for (uint i = 0; i < TILE_SIZE && (tile_i + i) < IN_SIZE; i++) {
                float sum = 0.0f;
                for (uint j = 0; j < TILE_SIZE; j++) {
                    sum += mat_c.thread_elements()[i * TILE_SIZE + j];
                }
                outputs[offset + tile_i + i] = as_type<uint8_t>(int8_t(sum + float(noise_vectors[offset + tile_i + i])));
            }
        }
    }
}