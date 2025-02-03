#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
import time
from typing import Tuple

# Load the ONNX model
MODEL_PATH = "tens_hash_fp32.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def nonce_to_input(nonce: int) -> np.ndarray:
    """Convert a nonce to input array format.
    Returns float32 array with shape [1,32]
    """
    # Convert nonce to 32 bytes (padded with zeros)
    nonce_bytes = nonce.to_bytes(32, byteorder='big')
    return np.frombuffer(nonce_bytes, dtype=np.uint8).astype(np.float32).reshape(1, 32)

def compute_error(input_array: np.ndarray) -> np.ndarray:
    """Compute error vector from input using SHA-256.
    Returns array with shape [1, 32]
    """
    # Get the SHA-256 digest of the input
    input_bytes = input_array.astype(np.uint8).tobytes()
    import hashlib
    digest = hashlib.sha256(input_bytes).digest()
    
    # Error is first 32 bytes shaped to [1, 32]
    error = np.array([digest[i] for i in range(32)], dtype=np.float32).reshape(1, 32)
    
    return error

def count_leading_zero_bits(byte_array: bytes) -> int:
    """Count number of leading zero bits in output."""
    for i, byte in enumerate(byte_array):
        if byte != 0:
            # Count leading zeros in this byte
            return i * 8 + len(bin(byte)[2:].rjust(8, '0').split('1')[0])
    return len(byte_array) * 8

def run_model(input_array: np.ndarray) -> bytes:
    """Run the ONNX model and return raw output bytes."""
    # Compute error vector from input
    error = compute_error(input_array)
    
    # Run inference
    output = session.run(None, {
        "input": input_array,
        "error": error
    })[0]
    
    # Convert to bytes
    return output.astype(np.uint8).tobytes()

def search_pow(target_bits: int, seed_hex: str = "0" * 64, max_nonce: int = None) -> Tuple[int, bytes]:
    """Search for nonce that produces given number of leading zero bits.
    Returns (nonce, hash_output) on success or raises ValueError if max_nonce reached.
    """
    print(f"Seed: {seed_hex}")
    print(f"Difficulty: {target_bits} leading 0 bits")
    print("Progress:")
    print("  Time    Hash Rate      TOPS         Total Hashes    Best Bits")
    print("  ----    ---------    --------      ------------    ----------", flush=True)
    
    nonce = 0
    start_time = time.time()
    hashes = 0
    last_print = start_time
    best_bits = 0

    # Calculate ops per hash: 2 matrix mults per round + sha256 + initial/final
    OPS_PER_HASH = 2 * 1e6  # Rough estimate

    while True:
        if max_nonce is not None and nonce >= max_nonce:
            raise ValueError(f"Failed to find solution with {target_bits} leading zero bits after {nonce} attempts")
        
        # Run model on current nonce
        input_array = nonce_to_input(nonce)
        output = run_model(input_array)
        hashes += 1

        # Check leading zeros and update best
        bits = count_leading_zero_bits(output)
        best_bits = max(best_bits, bits)

        # Check if solution found
        if bits >= target_bits:
            elapsed = time.time() - start_time
            print()  # New line after progress
            return nonce, output

        # Print progress every second
        now = time.time()
        if now - last_print >= 1.0:
            elapsed = now - start_time
            hash_rate = hashes/elapsed
            tops = (hash_rate * OPS_PER_HASH) / 1e12  # Tera-ops per second
            print(f"\r    {int(elapsed):3}s    {int(hash_rate):7} h/s    {tops:8.6f}    {hashes:12}    {best_bits:10}", end="", flush=True)
            last_print = now

        nonce += 1

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python tens_pow_onnx_fp32.py <target_bits>")
    
    try:
        target_bits = int(sys.argv[1])
        if target_bits < 0 or target_bits > 256:  # 256 bits = 32 bytes
            raise ValueError("Target bits must be between 0 and 256")
    except ValueError as e:
        sys.exit(f"Error: {str(e)}")

    try:
        # Use all zeros as seed
        seed_hex = "0" * 64
        
        # Search for solution
        nonce, output = search_pow(target_bits, seed_hex)
        print(f"\nSuccess! Found nonce: {nonce}")
        print(f"Output hash: {output.hex()}")
        print(f"Leading zero bits: {count_leading_zero_bits(output)}")
    except ValueError as e:
        sys.exit(f"Error: {str(e)}")
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
