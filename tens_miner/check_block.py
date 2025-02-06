#!/usr/bin/env python3
import hashlib
import binascii
import struct
import subprocess
from datetime import datetime

def double_sha256(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def hex_dump(data, bytes_per_line=32):
    """Print a nicely formatted hex dump with annotations"""
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        print(f'{i:04x}: {hex_part}')

def check_block():
    # Values from log - fixed block hex removing extra zeros between bits and nonce
    block_hex = "000000200ee0283b66d6d7c36b14aa9f9c038a87af3c3b834250508a21a8db00ce8acdf452a6852f8199a3db1f64cd7c5eff30dfd9549e2937a7fe9470d146a50e0b758bfe73a4671e00ffff002275bd01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0114ffffffff0100f2052a010000001976a914fb06d8d29a33c73e1168050e1f64353876e796d688ac00000000"
    reported_hash = "0000007ab548f41b95fb3f0b9ec7c026bc5630b962a52026ac3c7520627481ff"
    target_str = "00ffff000000000000000000000000000000000000000000000000000000"
    reported_seed = "93037c6b32d9ca06fa1ce7c03ab660216f374520fe6bb1cdac762807c4cdab3f"
    nonce = "002275bd".zfill(64)  # Pad to 64 characters
    node_merkle = "6a1d288f71e99c37506b2e0c5b75fd352660a5907469a8a5904e4f8ea04b0a98"

    # Parse block
    block_bytes = bytes.fromhex(block_hex)
    
    print("Header Components:")
    header = block_bytes[:80]
    version = header[0:4]
    prev_hash = header[4:36]
    merkle_root = header[36:68]
    timestamp = header[68:72]
    bits = header[72:76]
    pow_nonce = header[76:80]
    
    # Convert timestamp bytes to integer (little-endian)
    timestamp_int = int.from_bytes(timestamp, 'little')
    timestamp_dt = datetime.fromtimestamp(timestamp_int)
    
    print(f"Version:     {version.hex()}")
    print(f"Prev Hash:   {prev_hash[::-1].hex()}")
    print(f"Merkle Root: {merkle_root[::-1].hex()}")
    print(f"Timestamp:   {timestamp.hex()} (int: {timestamp_int}, time: {timestamp_dt})")
    print(f"Bits:        {bits.hex()}")
    print(f"PoW Nonce:   {pow_nonce.hex()}")
    
    print("\nMerkle Root Calculation:")
    # Get transaction data after header
    tx_data = block_bytes[80:]
    print(f"Raw tx data: {tx_data.hex()}")
    
    # Parse transaction components
    version = tx_data[0:4]  # Version field
    input_count = tx_data[4]  # Number of inputs
    cursor = 5  # Current position in tx_data
    
    # Skip past input(s)
    for i in range(input_count):
        cursor += 32  # Previous tx hash
        cursor += 4   # Previous tx index
        script_len = tx_data[cursor]
        cursor += 1   # Script length field
        cursor += script_len  # Script
        cursor += 4   # Sequence
    
    # Get number of outputs
    output_count = tx_data[cursor]
    cursor += 1
    
    # Skip past output(s)
    for i in range(output_count):
        cursor += 8   # Value
        script_len = tx_data[cursor]
        cursor += 1   # Script length field
        cursor += script_len  # Script
    
    cursor += 4  # Locktime
    
    # Extract just the transaction
    tx_only = tx_data[:cursor]
    print(f"Transaction only: {tx_only.hex()}")
    
    # Calculate merkle root (double SHA256 of the transaction)
    calculated_merkle = double_sha256(tx_only)
    print(f"Calculated merkle (internal): {calculated_merkle.hex()}")
    print(f"Calculated merkle (reversed): {calculated_merkle[::-1].hex()}")
    print(f"Header merkle (reversed):     {merkle_root[::-1].hex()}")
    print(f"Node merkle:                  {node_merkle}")
    print(f"Match header:                 {calculated_merkle == merkle_root}")
    print(f"Match node:                   {calculated_merkle[::-1].hex() == node_merkle}")
    
    # For PoW verification
    header_with_zero_nonce = version + prev_hash + merkle_root + timestamp + bits + bytes(4)
    calculated_seed = hashlib.sha256(hashlib.sha256(header_with_zero_nonce).digest()).digest().hex()
    
    print(f"\nPoW Parameters:")
    print(f"Calculated seed: {calculated_seed}")
    print(f"Reported seed:   {reported_seed}")
    print(f"Nonce to use:    {nonce}")
    print(f"Reported hash:   {reported_hash}")
    
    try:
        result = subprocess.run(['../test_pow/tens_hash', calculated_seed, nonce], 
                              capture_output=True, text=True)
        pow_hash = result.stdout.strip()
        print(f"\nPoW Verification:")
        print(f"Command:      ../test_pow/tens_hash {calculated_seed} {nonce}")
        print(f"Output hash:  {pow_hash}")
        print(f"Match:        {pow_hash == reported_hash}")
        print(f"Meets target: {int(pow_hash, 16) <= int(target_str, 16)}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running tens_hash: {e}")
        print(f"stderr: {e.stderr}")

if __name__ == "__main__":
    check_block()