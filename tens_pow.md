# Tensor-based Proof of Work Design

This document describes a proof of work mechanism based on the Learning with Errors (LWE) problem, implemented using tensor operations.

## Hash Function Structure

The core hash function processes 32-byte inputs through a series of tensor operations:

```
input (32 bytes) → expand (256-d) → 64 rounds → compress (32 bytes)
each round: state = (matrix × state + error) mod 256
```

Parameters:
- Input/Output: 32 bytes
- Hidden dimension: 256
- Rounds: 64 
- Arithmetic: Modulo 256
- Matrix precision: 8-bit integers

### Implementation Phases

1. Expansion Phase (32 → 256):
   ```c
   matrix_multiply(expand_mat, input, state, expand_noise, 
                  HIDDEN, IN_SIZE);
   ```

2. Middle Rounds (256 → 256):
   ```c
   for (uint32_t round = 0; round < ROUNDS; round++) {
       matrix_multiply(middle_mats[round], state, next_state, 
                      middle_noise + (round * HIDDEN),
                      HIDDEN, HIDDEN, impl_type);
   }
   ```

3. Compression Phase (256 → 32):
   ```c
   matrix_multiply(compress_mat, state, output, compress_noise, 
                  IN_SIZE, HIDDEN, impl_type);
   ```

### Error Vector Generation

Error vectors are derived deterministically from the input:
```c
crypto_hash_sha256(digest, input, IN_SIZE);
for (int i = 0; i < total_noise; i++) {
    noise[i] = digest[i % crypto_hash_sha256_BYTES];
}
```

## Hardware Implementation

The algorithm maps to standard tensor operations:
- Matrix multiplication
- 8-bit integer arithmetic
- Dense linear algebra
- Regular memory access patterns

Two reference implementations:
1. `int8`: 8-bit integer matrix multiply
2. `fp32`: 32-bit floating point operations

## Mining Protocol

Mining follows standard Bitcoin protocol:
1. Block header construction:
   - Previous block hash
   - Merkle root
   - Timestamp
   - Target
   - Nonce

2. Target validation:
   ```python
   tens_hash(block_header) < target
   ```

3. Difficulty adjustment via leading zero bits

## ONNX Implementation

Neural network format implementations:
- `tens_hash_fp32.onnx`: FP32 version
- `tens_hash_int8.onnx`: INT8 version 

The computation graph structure can be visualized as follows:

<image/svg+xml>
<svg xmlns="http://www.w3.org/2000/svg" id="export" class="canvas" preserveAspectRatio="xMidYMid meet" style="" width="214" height="588"><rect id="background" fill="#fff" pointer-events="all" width="214" height="588"/><g id="origin" transform="translate(9.725390625000001, 9.725390625000001) scale(1)"><g id="clusters" class="clusters"/><g id="edge-paths" class="edge-paths" style="pointer-events: none;"><defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto" style="fill: rgb(0, 0, 0);"><path d="M 0 0 L 10 5 L 0 10 L 4 5 z" style="stroke-width: 1;"/></marker></defs><path id="edge-input" class="edge-path" d="M25.640625,62.5L25.640625,64.25C25.640625,66,25.640625,69.5,25.640625,76.25C25.640625,83,25.640625,93,25.640625,103C25.640625,113,25.640625,123,26.299479166666668,129.66666666666666C26.958333333333332,136.33333333333334,28.276041666666668,139.66666666666666,28.934895833333332,141.33333333333334L29.59375,143" style="stroke: rgb(0, 0, 0); stroke-width: 1px; fill: none; marker-end: url(&quot;#arrowhead&quot;);"/><path id="edge-error" class="edge-path" d="M121.09336890243902,21L118.72103658536587,22.666666666666668C116.34870426829268,24.333333333333332,111.60403963414633,27.666666666666668,109.23170731707317,32.833333333333336C106.859375,38,106.859375,45,106.859375,52C106.859375,59,106.859375,66,106.859375,71.16666666666667C106.859375,76.33333333333333,106.859375,79.66666666666667,106.859375,81.33333333333333L106.859375,83" style="stroke: rgb(0, 0, 0); stroke-width: 1px; fill: none; marker-end: url(&quot;#arrowhead&quot;);"/></g><g id="nodes" class="nodes"><g id="input-name-input" class="node graph-input" transform="translate(5.34375,41.5)" style=""><g class="node-item graph-item-input" transform="translate(0,0)"><path d="M5,0h30.59375a5,5 0 0 1 5,5v11a5,5 0 0 1 -5,5h-30.59375a5,5 0 0 1 -5,-5v-11a5,5 0 0 1 5,-5z" style="stroke: rgb(0, 0, 0); fill: rgb(238, 238, 238); stroke-width: 0;"/><text x="6" y="15" style="font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe WPC&quot;, &quot;Segoe UI&quot;, Ubuntu, &quot;Droid Sans&quot;, sans-serif, &quot;PingFang SC&quot;; font-size: 11px; text-rendering: geometricprecision; user-select: none;">input</text><title>float32[1,32]</title></g></g><g id="node-id-0" class="node graph-node" transform="translate(69.359375,83)" style=""><g class="node-item node-item-type node-item-type-shape" transform="translate(0,0)"><path d="M5,0h65a5,5 0 0 1 5,5v16a0,0 0 0 1 0,0h-75a0,0 0 0 1 0,0v-16a5,5 0 0 1 5,-5z" style="stroke: rgb(0, 0, 0); fill: rgb(108, 79, 71); stroke-width: 0;"/><text x="6" y="15" style="font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe WPC&quot;, &quot;Segoe UI&quot;, Ubuntu, &quot;Droid Sans&quot;, sans-serif, &quot;PingFang SC&quot;; font-size: 11px; text-rendering: geometricprecision; user-select: none; fill: rgb(255, 255, 255);">Tile</text></g></g><g id="node-id-1" class="node graph-node" transform="translate(0,143)" style=""><g class="node-item node-item-type node-item-type-layer" transform="translate(0,0)"><path d="M5,0h65a5,5 0 0 1 5,5v16a0,0 0 0 1 0,0h-75a0,0 0 0 1 0,0v-16a5,5 0 0 1 5,-5z" style="stroke: rgb(0, 0, 0); fill: rgb(51, 85, 136); stroke-width: 0;"/><text x="6" y="15" style="font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe WPC&quot;, &quot;Segoe UI&quot;, Ubuntu, &quot;Droid Sans&quot;, sans-serif, &quot;PingFang SC&quot;; font-size: 11px; text-rendering: geometricprecision; user-select: none; fill: rgb(255, 255, 255);">Gemm</text></g></g></g></g></svg>
</image/svg+xml>

Key aspects of the graph:
1. Input layer: 1×32 dimensions
2. Error vectors: Added at each stage
3. Matrix operations: Gemm nodes
4. Modulo: Applied after each transform
5. Output: Back to 1×32 dimensions

## Research Directions

Open questions:
1. Parameter optimization
   - Hidden dimension size
   - Number of rounds
   - Error distribution

2. Security analysis
   - LWE hardness bounds
   - Memory hardness proofs
   - Attack surface evaluation

3. Implementation optimization
   - Matrix storage formats
   - Error vector generation
   - Hardware-specific tuning

## References

[1] Regev, O. "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography." Journal of the ACM 56, no. 6 (2009).

[2] Lyubashevsky, V., Peikert, C., Regev, O. "On Ideal Lattices and Learning with Errors Over Rings." EUROCRYPT 2010.
