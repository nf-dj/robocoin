# 🤖 TensCoin

## What is TensCoin?

TensCoin is a Bitcoin fork that reimagines cryptocurrency mining for the AI era. It introduces a novel proof-of-work algorithm designed so that optimal mining hardware is also efficient at running AI workloads. Instead of using SHA256 hashing, TensCoin's mining process is based on operations common in neural networks—particularly matrix multiplication.

## Why TensCoin?

The current state of cryptocurrency mining faces several problems:

1. **Mining Centralization**: Bitcoin mining has become highly centralized due to specialized ASIC requirements, limited manufacturers, and high capital costs. This concentration of mining power undermines the decentralization goals of cryptocurrency.

2. **Wasted Hardware Potential**: Traditional Bitcoin mining ASICs can only perform SHA256 hashing, serving no purpose beyond cryptocurrency mining. When mining becomes unprofitable, this specialized hardware often becomes electronic waste.

TensCoin addresses these problems by:
- **Enabling Hardware Reuse**: Mining hardware can be repurposed for AI/ML workloads
- **Democratizing AI Compute**: By incentivizing individuals to operate dual-purpose mining hardware
- **Reducing E-waste**: Extended hardware life beyond mining profitability

## How Does It Work?

TensCoin replaces Bitcoin's SHA256d with TensHash, a proof-of-work algorithm based on matrix operations common in neural networks. The algorithm processes inputs through multiple rounds of matrix multiplication with special properties that make it both cryptographically secure and efficient on AI hardware.

The mining difficulty is rooted in the computational hardness of integer linear programming (ILP) problems, which are known to be NP-hard [Garey & Johnson, 1979](https://doi.org/10.1137/0207010).

For detailed technical information about the proof-of-work design, including the mathematical foundations, implementation details, and security analysis, see [TensHash Documentation](tens_pow.md).

## Contributing

We welcome contributions in:
- Hardware implementations and benchmarks
- Security analysis
- Mining software integration
- Documentation and testing

## License

TensCoin is released under the MIT license. See [COPYING](COPYING) for more information.

---

For the original Bitcoin Core README, see [README_BITCOIN.md](README_BITCOIN.md).
