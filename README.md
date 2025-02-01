# TensCoin: Bitcoin with AI-friendly Proof of Work

TensCoin is a Bitcoin fork that reimagines proof-of-work for the AI era. By implementing a novel tensor-based mining algorithm inspired by the Learning with Errors (LWE) problem, TensCoin creates the first cryptocurrency that can share hardware efficiently between mining and AI workloads.

## Why TensCoin?

Traditional Bitcoin mining using SHA256 has led to specialized ASICs that serve no purpose beyond mining. TensCoin takes a different approach:
- Mining hardware can be repurposed for AI/ML workloads
- Cryptographic security is based on the well-studied LWE problem
- Core operations use the same primitives as neural networks

## How it Works

TensCoin replaces Bitcoin's SHA256d with a tensor-based proof-of-work that uses matrix multiplication as its core operation:

1. Block header hash -> seed -> matrices (via ChaCha20)
2. Nonce -> error vectors (via SHA256)
3. Proof-of-work function:
   ```
   input (32 bytes) -> expand (1024-d) -> 64 rounds -> compress (32 bytes)
   each round: state = (matrix × state + error) mod 256
   ```
4. Check output for required number of leading zeros

### Key Parameters
- Hidden dimension: 1024
- Matrix precision: 8-bit integers
- Arithmetic: Modulo 256
- Rounds: 64
- Input/output: 32 bytes (same as Bitcoin)

## Benefits

### 1. AI Hardware Compatibility
TensCoin's mining algorithm maps directly to AI accelerator primitives:
- Matrix multiplication units (TPUs, NPUs)
- Low-precision (8-bit) arithmetic
- Dense linear algebra operations
- High memory bandwidth utilization

This means mining hardware can efficiently switch between cryptocurrency mining and AI/ML workloads like:
- Neural network training
- Inference serving
- Large language models
- Computer vision

### 2. Cryptographic Security
Security is based on the Learning with Errors (LWE) problem, a foundation of post-quantum cryptography:
- Nonce is cryptographically bound to error vectors
- High-dimensional transformations resist shortcuts
- Multiple rounds provide nonlinearity
- Memory-hard by design

### 3. Sustainable Mining
TensCoin promotes sustainable cryptocurrency mining by:
- Enabling hardware reuse between mining and AI
- Reducing e-waste from specialized ASICs
- Creating dual-use infrastructure
- Aligning incentives with useful computation

## Technical Details

See src/crypto/tens_pow/ for the core implementation, which includes:
- ChaCha20-based deterministic matrix generation
- SHA256-based error vector derivation
- Optimized matrix multiplication routines
- Caching for repeated operations

## Roadmap

Planned improvements include:
- Hardware-specific optimizations for different AI accelerators
- Formal security analysis and parameter optimization
- Dynamic difficulty adjustment tuning
- Extended test coverage and benchmarks
- Mining pool reference implementation

## Contributing

We welcome contributions in:
- Security analysis and optimization
- Hardware implementations and benchmarks
- Parameter tuning and testing
- Documentation and examples
- Mining software integration

Please see CONTRIBUTING.md for more details.

## License

TensCoin is released under the MIT license. See COPYING for more information.

---

For the original Bitcoin Core README, see [README_BITCOIN.md](README_BITCOIN.md).