"""Mining worker thread implementation."""

import time
import torch
import binascii
import hashlib
from PyQt6.QtCore import QThread, pyqtSignal
from .mining import TensCoinMiner, nonces_from_counter  # Updated import
from .utils import count_leading_zero_bits
from .transactions import create_coinbase_tx, hash_tx, calc_merkle_root

def hex_string_to_bytes(hex_str):
    """Convert a hex string to bytes."""
    return binascii.unhexlify(hex_str)

def reverse_bytes(b):
    """Reverse a byte string."""
    return b[::-1]

def double_sha256(data):
    """Compute double SHA256 hash."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

class MiningWorker(QThread):
    progress = pyqtSignal(dict)
    solution = pyqtSignal(dict)
    status = pyqtSignal(str)
    
    def __init__(self, rpc_client, batch_size, device, mining_address):
        super().__init__()
        self.rpc = rpc_client
        self.batch_size = batch_size
        self.device = device
        self.mining_address = mining_address
        self.should_stop = False
        
        self.attempts = 0
        self.best_zero_bits = 0
        self.start_time = None
    
    def stop(self):
        self.should_stop = True
    
    def build_block_header(self, template):
        """Build block header from template fields."""
        # Create coinbase transaction
        coinbase_tx = create_coinbase_tx(
            address=self.mining_address,
            height=template['height'],
            value=template.get('coinbasevalue', 5000000000)
        )
        
        # Calculate merkle root
        tx_hashes = [hash_tx(coinbase_tx)]
        for tx in template.get('transactions', []):
            if 'hash' in tx:
                tx_hashes.append(bytes.fromhex(tx['hash'])[::-1])
        merkle_root = calc_merkle_root(tx_hashes)
        
        # Store coinbase for block assembly
        self.current_coinbase = coinbase_tx
        self.current_merkle_root = merkle_root
        
        # Convert version to 4-byte little-endian
        version = template['version'].to_bytes(4, 'little').hex()
        
        # Previous block hash (needs to be reversed)
        prev_hash = reverse_bytes(hex_string_to_bytes(template['previousblockhash'])).hex()
        
        # Use our calculated merkle root
        merkle_root_hex = merkle_root[::-1].hex()
        
        # Time as 4-byte little-endian
        time = template['curtime'].to_bytes(4, 'little').hex()
        
        # Bits (already in correct format)
        bits = template['bits']
        
        # Combine all fields
        header = version + prev_hash + merkle_root_hex + time + bits
        return header
        
    def run(self):
        try:
            # Get initial mining info
            mining_info = self.rpc.get_mining_info()
            target = mining_info.get('target', '0' * 64)
            self.status.emit(f"Starting mining with target {target}")
            
            # Get work from block template
            template = self.rpc.get_block_template()
            if not template:
                self.status.emit("Failed to get block template from node")
                return
            
            # Build initial header
            header_base = self.build_block_header(template)
            print(f"Initial header: {header_base}")  # Debug print
            print(f"Header length: {len(header_base)//2} bytes")  # Should be 76 bytes without nonce
            
            # Hash header to get 32-byte seed for miner
            header_bytes = hex_string_to_bytes(header_base)
            seed = double_sha256(header_bytes).hex()
            print(f"Seed for mining: {seed}")  # Should be 32 bytes
            
            bits = template.get('bits')
            self.status.emit(f"Got block template - version: {template['version']}, bits: {bits}")
                
            # Initialize miner with seed
            miner = TensCoinMiner(seed, self.device)  # Updated class name
            miner.eval()
            self.start_time = time.time()
            attempts = 0
            
            while not self.should_stop:
                # Check for new blocks before each batch
                new_template = self.rpc.get_block_template()
                if new_template:
                    if new_template.get('previousblockhash') != template.get('previousblockhash'):
                        # New block found - update everything
                        template = new_template
                        header_base = self.build_block_header(template)
                        header_bytes = hex_string_to_bytes(header_base)
                        seed = double_sha256(header_bytes).hex()
                        miner = TensCoinMiner(seed, self.device)
                        miner.eval()
                        self.status.emit(f"New block detected: {template['previousblockhash']}")
                
                nonce_batch = nonces_from_counter(attempts, self.batch_size)
                attempts += self.batch_size
                self.attempts = attempts
                
                with torch.no_grad():
                    out_batch = miner.forward_batch(nonce_batch)
                
                out_batch_cpu = out_batch.cpu().numpy()
                nonce_batch_cpu = nonce_batch.cpu().numpy()
                
                for i in range(self.batch_size):
                    hash_result = bytes(out_batch_cpu[i].tolist())[::-1]
                    zeros = count_leading_zero_bits(hash_result)
                    
                    if zeros > self.best_zero_bits:
                        self.best_zero_bits = zeros
                        self.progress.emit({
                            'attempts': attempts,
                            'best_zero_bits': zeros,
                            'elapsed_time': time.time() - self.start_time
                        })
                    
                    if zeros >= mining_info.get('target_bits', 24):  # Default to 24 if not specified
                        nonce_bytes = bytes(nonce_batch_cpu[i].tolist())[::-1]
                        
                        # Build full block with our nonce
                        block_hex = header_base + nonce_bytes.hex()
                        
                        # Add transactions starting with our coinbase
                        block_hex += self.current_coinbase.hex()
                        for tx in template.get('transactions', []):
                            if 'data' in tx:
                                block_hex += tx['data']
                        
                        # Submit solution
                        success = self.rpc.submit_block(block_hex)
                        
                        if success:
                            self.solution.emit({
                                'nonce': nonce_bytes,
                                'hash': hash_result,
                                'attempts': attempts,
                                'time': time.time() - self.start_time
                            })
                            # Get new template
                            template = self.rpc.get_block_template()
                            if template:
                                header_base = self.build_block_header(template)
                                header_bytes = hex_string_to_bytes(header_base)
                                seed = double_sha256(header_bytes).hex()
                                miner = TensCoinMiner(seed, self.device)  # Updated class name
                                miner.eval()
                                self.status.emit(f"Got new block template - bits: {template['bits']}")
                        else:
                            self.status.emit("Block rejected by node")
                
                # Regular progress updates
                if attempts % (self.batch_size * 10) == 0:
                    self.progress.emit({
                        'attempts': attempts,
                        'best_zero_bits': self.best_zero_bits,
                        'elapsed_time': time.time() - self.start_time
                    })
        
        except Exception as e:
            import traceback
            self.status.emit(f"Error in mining worker: {str(e)}\n{traceback.format_exc()}")
