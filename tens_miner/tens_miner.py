"""TensHash miner implementation."""

import time
import json
import struct
import threading
from typing import Optional, Tuple, Dict

import numpy as np
from .constants import *
from .rpc import RPCClient

class TensHashMiner:
    """Tensor-based proof of work miner."""
    
    def __init__(self):
        self.running = False
        self.current_job: Optional[Dict] = None
        self.rpc = RPCClient()
        
    def _tensor_hash(self, data: bytes, nonce: int) -> np.ndarray:
        """Compute tensor-based hash of data with nonce.
        
        Args:
            data: Input data to hash
            nonce: Nonce value to append
            
        Returns:
            Final hash result as numpy array
        """
        # Your existing tensor hash implementation here
        pass
        
    def _check_hash(self, hash_result: np.ndarray, target: int) -> bool:
        """Check if hash meets difficulty target.
        
        Args:
            hash_result: Hash to check
            target: Target difficulty
            
        Returns:
            True if hash meets target, False otherwise
        """
        # Convert tensor result to comparable integer
        hash_int = int.from_bytes(hash_result.tobytes(), byteorder='little')
        return hash_int < target
        
    def _get_new_job(self) -> Dict:
        """Get new mining job from RPC.
        
        Returns:
            Dict containing job details
        """
        template = self.rpc.get_block_template()
        
        # Extract relevant fields
        version = template['version']
        prev_hash = bytes.fromhex(template['previousblockhash'])
        merkle_root = bytes.fromhex(template['merkleroot'])
        timestamp = template['curtime']
        bits = int(template['bits'], 16)
        
        return {
            'version': version,
            'prev_hash': prev_hash,
            'merkle_root': merkle_root,
            'timestamp': timestamp,
            'bits': bits,
            'target': bits_to_target(bits),
            'height': template['height'],
            'template': template
        }
        
    def _build_header(self, job: Dict, nonce: int) -> bytes:
        """Build block header for hashing.
        
        Args:
            job: Current job dict
            nonce: Nonce to include
            
        Returns:
            Raw block header bytes
        """
        header = struct.pack("<I", job['version'])
        header += job['prev_hash']
        header += job['merkle_root']
        header += struct.pack("<I", job['timestamp'])
        header += struct.pack("<I", job['bits'])
        header += struct.pack("<I", nonce)
        return header
        
    def mine(self):
        """Main mining loop."""
        self.running = True
        
        while self.running:
            try:
                # Get new job if needed
                if not self.current_job:
                    self.current_job = self._get_new_job()
                    print(f"New job at height {self.current_job['height']}")
                
                # Mine a batch
                for nonce in range(BATCH_SIZE):
                    if not self.running:
                        break
                        
                    header = self._build_header(self.current_job, nonce)
                    hash_result = self._tensor_hash(header, nonce)
                    
                    if self._check_hash(hash_result, self.current_job['target']):
                        print(f"Found solution! Nonce: {nonce}")
                        # Build full block
                        block = self._build_block(self.current_job, nonce)
                        # Submit
                        result = self.rpc.submit_block(block.hex())
                        if result is None:
                            print("Block accepted!")
                        else:
                            print(f"Block rejected: {result}")
                        # Get new job
                        self.current_job = None
                        break
                
                # Update job periodically
                if time.time() - self.current_job['timestamp'] > 60:
                    self.current_job = None
                    
            except Exception as e:
                print(f"Mining error: {e}")
                time.sleep(1)
                self.current_job = None
                
    def start(self):
        """Start mining in background thread."""
        self.thread = threading.Thread(target=self.mine)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop mining."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
            
def bits_to_target(bits: int) -> int:
    """Convert compact bits representation to full target."""
    size = bits >> 24
    word = bits & 0x007fffff
    
    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))

# For testing
if __name__ == '__main__':
    miner = TensHashMiner()
    miner.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        miner.stop()