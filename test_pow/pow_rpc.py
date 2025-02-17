#!/usr/bin/env python3
import json
import time
from datetime import datetime
import requests
import subprocess
import sys
import hashlib
import struct

class BitcoinRPC:
    def __init__(self, rpc_user='robocoin', rpc_password='robocoin', rpc_host='65.109.105.241', rpc_port=8332):
        self.url = f'http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}'
        self.headers = {'content-type': 'application/json'}
        
    def _call(self, method, params=None):
        payload = {
            "jsonrpc": "2.0",
            "id": str(time.time()),
            "method": method,
            "params": params or []
        }
        try:
            response = requests.post(self.url, json=payload, headers=self.headers)
            return response.json()
        except Exception as e:
            print(f"RPC Error: {str(e)}")
            return None

    def get_block_template(self):
        return self._call('getblocktemplate', [{"rules": ["segwit"]}])
    
    def get_best_block_hash(self):
        return self._call('getbestblockhash')
    
    def get_block(self, block_hash):
        return self._call('getblock', [block_hash])

def create_coinbase_transaction(height, value, pubkey_hash):
    # Create coinbase input
    script_sig = struct.pack("<I", height)  # Height in scriptSig
    version = struct.pack("<I", 1)  # Version 1
    sequence = 0xffffffff
    
    # Create output
    script_pubkey = b'\x76\xa9\x14' + pubkey_hash + b'\x88\xac'  # Standard P2PKH
    
    # Transaction structure
    tx = version  # Version
    tx += b'\x01'  # Input count (1)
    tx += b'\x00' * 32  # Previous tx hash (null for coinbase)
    tx += b'\xff\xff\xff\xff'  # Previous output index (-1 for coinbase)
    tx += bytes([len(script_sig)])  # ScriptSig length
    tx += script_sig  # ScriptSig
    tx += struct.pack("<I", sequence)  # Sequence
    tx += b'\x01'  # Output count (1)
    tx += struct.pack("<Q", value)  # Value
    tx += bytes([len(script_pubkey)])  # ScriptPubKey length
    tx += script_pubkey  # ScriptPubKey
    tx += b'\x00\x00\x00\x00'  # Lock time
    
    return tx

def double_sha256(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def create_block_header(version, prev_block, merkle_root, timestamp, bits, nonce):
    header = struct.pack("<I", version)  # 4 bytes version
    header += bytes.fromhex(prev_block)[::-1]  # 32 bytes prev block (reversed)
    header += bytes.fromhex(merkle_root)[::-1]  # 32 bytes merkle root (reversed)
    header += struct.pack("<I", timestamp)  # 4 bytes timestamp
    header += struct.pack("<I", bits)  # 4 bytes bits
    header += struct.pack("<I", nonce)  # 4 bytes nonce
    return header

def create_coreml_model(header_hash):
    """Create CoreML model for the given header hash"""
    try:
        proc = subprocess.run(
            ['./export_model_coreml.py', header_hash],
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            print(f"Error creating CoreML model: {proc.stderr}")
            return False
        print(f"CoreML model creation output: {proc.stdout}")
        return True
    except Exception as e:
        print(f"Error running export script: {e}")
        return False

def mine_pow(difficulty):
    """Run PoW miner and return solution when found"""
    try:
        proc = subprocess.Popen(
            ['./pow_coreml', str(difficulty)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        while True:
            line = proc.stdout.readline()
            if not line:
                break
                
            try:
                rpc_msg = json.loads(line)
                
                if rpc_msg.get("method") == "status":
                    params = rpc_msg.get("params", {})
                    print(f"Mining... Nonce: {params.get('nonce')}, "
                          f"Hashrate: {params.get('hashrate'):.2f} H/s, "
                          f"Best difficulty: {params.get('best_difficulty')}")
                    
                elif rpc_msg.get("method") == "solution":
                    params = rpc_msg.get("params", {})
                    proc.terminate()
                    return params
                    
                elif "error" in rpc_msg:
                    error = rpc_msg.get("error", {})
                    print(f"Error: {error.get('message')} - {error.get('data')}")
                    proc.terminate()
                    return None
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
                continue
                
        proc.terminate()
        return None
        
    except Exception as e:
        print(f"Error running PoW miner: {e}")
        return None

def main():
    # Get current best block hash from node
    rpc = BitcoinRPC()
    response = rpc.get_best_block_hash()
    if not response or 'error' in response:
        print(f"Failed to get best block hash: {response.get('error', 'Unknown error')}")
        return
        
    prev_block = response['result']
    print(f"\nPrevious block hash: {prev_block}")
    
    # Get block height
    block_response = rpc.get_block(prev_block)
    if not block_response or 'error' in block_response:
        print(f"Failed to get block info: {block_response.get('error', 'Unknown error')}")
        return
        
    height = block_response['result']['height'] + 1
    print(f"Mining block at height: {height}")
    
    timestamp = int(time.time())
    version = 0x20000000  # Version with segwit signal
    bits = 0x1e00ffff  # From genesis block params
    nonce = 0
    
    # Create coinbase transaction
    pubkey_hash = bytes([0] * 20)  # Example pubkey hash (all zeros)
    coinbase_tx = create_coinbase_transaction(height, 5000000000, pubkey_hash)  # 50 BTC reward
    
    # Calculate merkle root to match genesis block format
    # This matches the assert value: 485bc464e61956d85dda756d74e0ec872c594190d15ff9e37a00eac87c7d8fe0
    tx_hash = double_sha256(coinbase_tx)
    merkle_root = tx_hash[::-1].hex()  # Reverse bytes like in BlockMerkleRoot
    
    print("\nCoinbase Transaction Details:")
    print("-----------------------------")
    print(f"Transaction Version: 1")
    print("Input:")
    print(f"  Previous Outpoint: 0000000000000000000000000000000000000000000000000000000000000000:4294967295")
    print(f"  ScriptSig (hex): {coinbase_tx[41:46].hex()}")
    print(f"  Sequence: 0xffffffff")
    print("Output:")
    print(f"  Value: 5000000000 satoshis")
    print(f"  ScriptPubKey (hex): 76a914000000000000000000000000000000000000000088ac")
    print(f"Transaction Lock Time: 0")
    print(f"Complete Transaction Hex: {coinbase_tx.hex()}")
    print(f"Transaction Hash: {double_sha256(coinbase_tx)[::-1].hex()}")
    print(f"Merkle Root: {merkle_root}")
    
    # Create block header
    header = create_block_header(
        version=version,
        prev_block=prev_block,
        merkle_root=merkle_root,
        timestamp=timestamp,
        bits=bits,
        nonce=0
    )
    
    print("\nBlock Header Details:")
    print("--------------------")
    print(f"Version: {hex(version)}")
    print(f"Previous Block: {prev_block}")
    print(f"Merkle Root: {merkle_root}")
    print(f"Time: {timestamp} ({datetime.fromtimestamp(timestamp)})")
    print(f"Bits: {hex(bits)}")
    print(f"Nonce: {nonce}")
    print(f"Block Header Hex: {header.hex()}")
    header_hash = double_sha256(header)[::-1].hex()
    print(f"Block Header Hash (nonce=0): {header_hash}")
    
    print(f"\nCreating CoreML model for header hash: {header_hash}")
    if not create_coreml_model(header_hash):
        print("Failed to create CoreML model")
        return
    
    print("\nStarting PoW mining...")
    difficulty = 24  # Target difficulty - adjust as needed
    solution = mine_pow(difficulty)
    
    if solution:
        print("\nSolution found!")
        print(f"Nonce: {solution['nonce']}")
        print(f"Input hex: {solution['input_hex']}")
        print(f"Output hex: {solution['output_hex']}")
        print(f"Leading zeros: {solution['leading_zeros']}")
        
        # Create complete block with the found nonce
        final_header = create_block_header(
            version=version,
            prev_block=prev_block,
            merkle_root=merkle_root,
            timestamp=timestamp,
            bits=bits,
            nonce=solution['nonce']
        )
        
        # Construct complete block
        block = final_header + bytes([1])  # 1 transaction
        block += coinbase_tx  # Add coinbase transaction
        
        print("\nSubmitting block to node...")
        rpc = BitcoinRPC()
        result = rpc._call('submitblock', [block.hex()])
        
        print(f"Submit block response: {result}")
        
        if result is None:
            print("Failed to connect to node")
        elif result["result"] == None:
            print("Block submitted successfully!")
            print(f"Block hash: {double_sha256(final_header)[::-1].hex()}")
        else:
            print(f"Error submitting block: {result}")
    else:
        print("\nNo solution found")

if __name__ == "__main__":
    main()
