"""Transaction utilities for TensCoin miner."""

import hashlib
import struct
from typing import List, Tuple

def create_coinbase_input(height: int) -> bytes:
    """Create a coinbase input including block height.
    
    Args:
        height: Block height
        
    Returns:
        Serialized coinbase input
    """
    # Serialize height
    height_bytes = height.to_bytes((height.bit_length() + 7) // 8, 'little')
    
    # Script length is variable
    script_sig = height_bytes
    script_len = len(script_sig).to_bytes(1, 'little')  # Assuming script < 255 bytes
    
    # Build input:
    # prev_tx (32 bytes all zero) | prev_index (4 bytes -1) | script_len | script_sig | sequence (4 bytes -1)
    return (
        b'\x00' * 32 +  # prev_tx
        b'\xff\xff\xff\xff' +  # prev_index (-1)
        script_len +
        script_sig +
        b'\xff\xff\xff\xff'  # sequence
    )

def address_to_script(address: str) -> bytes:
    """Convert bech32 address to output script.
    
    Args:
        address: Bech32 address starting with 'tens1'
        
    Returns:
        Output script bytes
    """
    # For now just doing P2PKH for testing
    # TODO: Implement proper bech32 decode and script creation
    dummy_hash = hashlib.sha256(address.encode()).digest()[:20]
    
    return (
        b'\x76' +  # OP_DUP
        b'\xa9' +  # OP_HASH160
        b'\x14' +  # Push 20 bytes
        dummy_hash +
        b'\x88' +  # OP_EQUALVERIFY
        b'\xac'    # OP_CHECKSIG
    )

def create_coinbase_tx(address: str, height: int, value: int = 5000000000) -> bytes:
    """Create complete coinbase transaction.
    
    Args:
        address: Output address
        height: Block height for input script
        value: Output value in satoshis (50 TensCoin default)
        
    Returns:
        Complete serialized transaction
    """
    # Version 1
    tx = struct.pack('<I', 1)
    
    # One input
    tx += b'\x01'
    tx += create_coinbase_input(height)
    
    # One output
    tx += b'\x01'
    tx += struct.pack('<Q', value)  # 8-byte value
    out_script = address_to_script(address)
    tx += bytes([len(out_script)])  # script length
    tx += out_script
    
    # Locktime 0
    tx += b'\x00\x00\x00\x00'
    
    return tx

def hash_tx(tx: bytes) -> bytes:
    """Double SHA256 hash a transaction.
    
    Args:
        tx: Raw transaction bytes
        
    Returns:
        32-byte transaction hash
    """
    return hashlib.sha256(hashlib.sha256(tx).digest()).digest()

def calc_merkle_root(tx_hashes: List[bytes]) -> bytes:
    """Calculate the merkle root from transaction hashes.
    
    Args:
        tx_hashes: List of 32-byte transaction hashes
        
    Returns:
        32-byte merkle root
    """
    if not tx_hashes:
        return b'\x00' * 32
        
    if len(tx_hashes) == 1:
        return tx_hashes[0]
    
    # Make sure we have even number of hashes
    if len(tx_hashes) % 2 == 1:
        tx_hashes.append(tx_hashes[-1])
    
    next_level = []
    for i in range(0, len(tx_hashes), 2):
        combined = tx_hashes[i] + tx_hashes[i+1]
        next_hash = hashlib.sha256(hashlib.sha256(combined).digest()).digest()
        next_level.append(next_hash)
    
    return calc_merkle_root(next_level)
