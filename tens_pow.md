# TensHash Proof-of-Work Overview

The core idea behind TensHash is to generate random ternary matrices and then process an input vector through multiple rounds of transformations to meet a target value.

## Key Concepts

### Random Ternary Matrices
Each layer uses a matrix **A** with entries in **{-1, 0, 1}**. These matrices are generated pseudo-randomly (using ChaCha20) from a seed.

### Input Transformation
The input is a 256-bit (32-byte) binary vector. Before each matrix multiplication, it is shifted from **{0, 1}** to **{-1, +1}** via the mapping:

```
x_mapped = 2x - 1
```

### Layered Processing
The algorithm consists of three stages:

1. **Expansion**
   - Converts the 256-bit input into a hidden state of 1024 bits
   - Uses a 1024×256 matrix

2. **Hidden Layers**
   - Consists of 64 rounds
   - In each round:
     - The binary state (after being shifted) is multiplied by a 1024×1024 ternary matrix
     - A ReLU-like activation is applied (clamping the result to {0, 1}) to produce a new binary vector

3. **Compression**
   - The 1024-bit state is reduced back to 256 bits
   - Uses a 256×1024 matrix

### Proof-of-Work Goal
The final 256-bit output is compared to a target value. The goal is to find an input (or nonce) such that, after all these rounds, the output meets the target.

### Security Basis – ILP Hardness
The security of TensHash relies on the computational hardness of an underlying integer linear programming (ILP) problem. Finding a solution x ∈ {-1, +1}ⁿ such that

```
A · x ≤ target
```

(with the activation non-linearity applied after each matrix multiplication) is NP-hard. This NP-hardness, as discussed in [Garey & Johnson, 1979](https://doi.org/10.1137/0207010), ensures that no known efficient algorithm exists for quickly solving the problem.

## Implementation

The full implementation of the TensHash Proof-of-Work can be found in the repository file:
[src/crypto/tens_pow/tens_pow.cpp](src/crypto/tens_pow/tens_pow.cpp)

## Summary

- **Random Matrices**: Generated with entries in **{-1, 0, 1}**
- **Input Shifting**: Binary input is mapped to **{-1, +1}** before each multiplication
- **Multi-Round Transformation**: The network uses an expansion layer, 64 hidden rounds with activation, and a compression layer
- **PoW Condition**: The resulting 256-bit output must be less than or equal to a given target
- **Security**: Underpinned by the NP-hardness of the associated ILP problem

This summary explains the essential operation and security rationale behind the TensHash-based PoW.
