#!/usr/bin/env python3
import json
import requests
import binascii
import struct
import hashlib

def hex_string_to_bytes(hex_str):
    return binascii.unhexlify(hex_str)

def double_sha256(data):
    """Compute double SHA256 hash."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def submit_block():
    url = "http://65.109.105.241:8332"
    username = "tenscoin"
    password = "tenscoin"
    
    # Standard Bitcoin block header construction (80 bytes)
    version = struct.pack("<I", 536870912)  # 4 bytes, little-endian
    prev_hash = bytes.fromhex("f4cd8ace00dba8218a505042833b3caf878a039c9faa146bc3d7d6663b28e00e")[::-1]  # 32 bytes, internal byte order
    merkle_root = bytes.fromhex("8b750b0ea546d17094fea737299e54d9df30ff5e7ccd641fdba399812f85a652")[::-1]  # 32 bytes, internal byte order
    timestamp = struct.pack("<I", 1738830846)  # 4 bytes, little-endian
    bits = bytes.fromhex("1e00ffff")  # 4 bytes
    pow_nonce = bytes.fromhex("bd752200")  # 4 bytes, little-endian
    
    # For seed: same header but with nonce=0
    zero_nonce = bytes(4)
    
    # Build headers
    header_for_seed = version + prev_hash + merkle_root + timestamp + bits + zero_nonce
    header_for_block = version + prev_hash + merkle_root + timestamp + bits + pow_nonce
    
    print("Block Header Analysis:")
    print("-" * 50)
    
    # Calculate seed (block hash with nonce=0)
    seed = double_sha256(header_for_seed).hex()
    pow_seed = "93037c6b32d9ca06fa1ce7c03ab660216f374520fe6bb1cdac762807c4cdab3f"
    
    print("Header Components:")
    print(f"Version:     {version.hex()}")
    print(f"Prev Hash:   {prev_hash[::-1].hex()}")  # Display in external byte order
    print(f"Merkle Root: {merkle_root[::-1].hex()}")  # Display in external byte order
    print(f"Timestamp:   {timestamp.hex()}")
    print(f"Bits:        {bits.hex()}")
    print(f"PoW Nonce:   {pow_nonce.hex()}")
    print()
    
    print("Seed Verification:")
    print(f"Header with nonce=0: {header_for_seed.hex()}")
    print(f"Calculated seed:     {seed}")
    print(f"Original PoW seed:   {pow_seed}")
    print(f"Seed match:          {seed == pow_seed}")
    print()
    
    # Transaction data
    tx_data = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0114ffffffff0100f2052a010000001976a914fb06d8d29a33c73e1168050e1f64353876e796d688ac00000000"
    
    # Full block = header + transactions
    block_hex = header_for_block.hex() + tx_data
    
    print(f"Block hex length: {len(block_hex)} chars")
    print()

    # Submit block
    headers = {'content-type': 'application/json'}
    payload = {
        "method": "submitblock",
        "params": [block_hex],
        "jsonrpc": "2.0",
        "id": 1,
    }
    
    try:
        print("Getting current block template...")
        template_payload = {
            "method": "getblocktemplate",
            "params": [{"rules": ["segwit"]}],
            "jsonrpc": "2.0",
            "id": 1,
        }
        response = requests.post(url, auth=(username, password), headers=headers, data=json.dumps(template_payload))
        template = response.json()
        print(json.dumps(template, indent=2))
        
        print("\nSubmitting block...")
        response = requests.post(url, auth=(username, password), headers=headers, data=json.dumps(payload))
        result = response.json()
        print("\nRPC Response:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    submit_block()