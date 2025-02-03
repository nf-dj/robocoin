# 🤖 TensCoin: Bitcoin with AI-friendly Proof of Work

TensCoin is a Bitcoin fork exploring a new approach to proof-of-work designed for the AI era. It implements a tensor-based mining algorithm inspired by the Learning with Errors (LWE) problem, aiming to enable efficient hardware sharing between cryptocurrency mining and AI workloads. By repurposing widely distributed mining hardware, TensCoin has the potential to diversify and democratize AI computation—helping to break the concentration of processing power in the data centers of a few big companies.

## Why TensCoin?

Traditional Bitcoin mining using SHA256 has led to specialized ASICs that serve no purpose beyond mining. TensCoin experiments with a different approach:
- **Dual-Use Hardware:** Mining hardware can be repurposed for both cryptocurrency mining and AI/ML workloads.
- **Decentralized AI Computation:** By encouraging individual operators to contribute processing power, TensCoin could help decentralize AI compute, reducing the current concentration in large, centralized data centers.
- **Cryptographic Security:** Security is based on the well-studied Learning with Errors (LWE) problem.
- **Neural Network Primitives:** Core operations mirror those used in modern neural network computations.

## How it Works

TensCoin replaces Bitcoin's SHA256d with a tensor-based proof-of-work that uses matrix multiplication as its core operation:

1. **Matrix Generation:** The block header hash is used as a seed (via ChaCha20) to generate matrices.
2. **Error Vector Derivation:** The nonce is processed using SHA256 to derive error vectors.
3. **Proof-of-Work Function:**  
   ```
   input (32 bytes) -> expand (256-d) -> 64 rounds -> compress (32 bytes)
   each round: state = (matrix × state + error) mod 256
   ```
4. **Validation:** Check the final output for the required number of leading zeros.

See [Proof of Work Design](tens_pow.md) for more detailes.

### Key Parameters
- **Hidden Dimension:** 256
- **Matrix Precision:** 8-bit integers  
- **Arithmetic:** Modulo 256  
- **Rounds:** 64  
- **Input/Output:** 32 bytes (same as Bitcoin)

## Benefits

### 1. AI Hardware Compatibility
TensCoin's mining algorithm is designed with modern AI accelerators in mind:
- Utilizes operations common to TPUs, NPUs, and GPUs.
- Leverages low-precision (8-bit) arithmetic and dense linear algebra operations.
- Maximizes the use of high memory bandwidth.
- **Dual-Purpose Hardware:** The same hardware can mine cryptocurrency and run AI/ML workloads (e.g., neural network training, inference serving, large language models, and computer vision), promoting a broader distribution of compute resources.

### 2. Cryptographic Security
Security is underpinned by the LWE problem—a well-studied foundation of post-quantum cryptography:
- The nonce is cryptographically bound to error vectors.
- High-dimensional transformations and lattice problems ensure robust security.
- Memory-hard design adds an extra layer of protection.

[1] Regev, O. "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography." Journal of the ACM 56, no. 6 (2009).

### 3. Sustainable and Decentralized Mining
TensCoin aims to transform the mining process into one that is both sustainable and socially beneficial:
- **Hardware Reuse:** By enabling dual use of mining hardware, TensCoin reduces electronic waste from specialized ASICs.
- **Decentralization of AI:** TensCoin’s model could distribute AI compute across a global network of small-scale miners rather than relying on a few enormous data centers, making AI computation more accessible and democratic.
- **Incentivized Useful Computation:** Aligns economic incentives with the production of computations that are useful for both securing the blockchain and supporting AI workloads.

## Technical Details

See the `src/crypto/tens_pow/` folder for the core implementation details, which include:
- ChaCha20-based deterministic matrix generation.
- SHA256-based error vector derivation.
- Optimized matrix multiplication routines.
- Caching for repeated operations.

## Roadmap

Areas for research and improvement include:
- Hardware-specific optimizations for various AI accelerators.
- Formal security analysis and parameter optimization.
- Dynamic difficulty adjustment tuning.
- Extended test coverage and performance benchmarks.
- Development of a mining pool reference implementation.

## Contributing

We welcome contributions in:
- Security analysis and optimization.
- Hardware implementations and benchmarks.
- Parameter tuning and testing.
- Documentation and example projects.
- Mining software integration.

## License

TensCoin is released under the MIT license. See [COPYING](COPYING) for more information.

---

For the original Bitcoin Core README, see [README_BITCOIN.md](README_BITCOIN.md).

