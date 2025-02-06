#!/usr/bin/env python3
import hashlib
import binascii
import struct
import subprocess

def hex_string_to_bytes(hex_str):
    return binascii.unhexlify(hex_str)

def check_block():
    # Values from log
    block_hex = "000000200ee0283b66d6d7c36b14aa9f9c038a87af3c3b834250508a21a8db00ce8acdf452a6852f8199a3db1f64cd7c5eff30dfd9549e2937a7fe9470d146a50e0b758bfe73a4671e00ffff00000000000000000000000000000000000000000000000000000000002275bd01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0114ffffffff0100f2052a010000001976a914fb06d8d29a33c73e1168050e1f64353876e796d688ac00000000"
    reported_hash = "0000007ab548f41b95fb3f0b9ec7c026bc5630b962a52026ac3c7520627481ff"
    target_str = "00ffff000000000000000000000000000000000000000000000000000000"
    reported_seed = "93037c6b32d9ca06fa1ce7c03ab660216f374520fe6bb1cdac762807c4cdab3f"
    nonce = "00000000000000000000000000000000000000000000000000000000002275bd"

    print("Block Header Analysis:")
    print("-" * 50)

    # Parse header components
    block_bytes = bytes.fromhex(block_hex)
    header = block_bytes[:80]  # First 80 bytes is header
    version = header[0:4]
    prev_hash = header[4:36]
    merkle_root = header[36:68]
    timestamp = header[68:72]
    bits = header[72:76]

    # Construct header with zero nonce
    header_with_zero_nonce = version + prev_hash + merkle_root + timestamp + bits + bytes(4)
    print(f"Header Components (80 bytes with nonce=0):")
    print(f"Version:     {version.hex()}")
    print(f"Prev Hash:   {prev_hash[::-1].hex()}")  # Display reversed for readability
    print(f"Merkle Root: {merkle_root[::-1].hex()}")  # Display reversed for readability
    print(f"Timestamp:   {timestamp.hex()}")
    print(f"Bits:        {bits.hex()}")
    print(f"Nonce:       {bytes(4).hex()}")
    print(f"Full header with zero nonce: {header_with_zero_nonce.hex()}")
    
    # Calculate seed (double SHA256 of header with zero nonce)
    calculated_seed = hashlib.sha256(hashlib.sha256(header_with_zero_nonce).digest()).digest().hex()
    print(f"\nSeed Verification (SHA256d of header with nonce=0):")
    print(f"Calculated:  {calculated_seed}")
    print(f"Reported:    {reported_seed}")
    print(f"Match:       {calculated_seed == reported_seed}")

    # Verify PoW hash meets target using tens_hash CLI
    print(f"\nPoW Hash Verification:")
    try:
        result = subprocess.run(['../test_pow/tens_hash', calculated_seed, nonce], 
                              capture_output=True, text=True)
        pow_hash = result.stdout.strip()
        print(f"Input Nonce:    {nonce}")
        print(f"Target:         {target_str}")
        print(f"Calculated PoW: {pow_hash}")
        print(f"Reported PoW:   {reported_hash}")
        print(f"Hash Match:     {pow_hash == reported_hash}")
        
        # Check if hash meets target
        hash_int = int(pow_hash, 16)
        target_int = int(target_str, 16)
        print(f"Meets Target:   {hash_int <= target_int}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tens_hash: {e}")
        print(f"stderr: {e.stderr}")

if __name__ == "__main__":
    check_block()