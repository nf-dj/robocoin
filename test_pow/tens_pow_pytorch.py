import torch
import torch.nn as nn
import time
import argparse
from hashlib import sha256
from Crypto.Cipher import ChaCha20
import numpy as np

IN_SIZE = 32
HIDDEN = 256
ROUNDS = 64

class TensHash(nn.Module):
    def __init__(self, seed_hex, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.seed = bytes.fromhex(seed_hex)
        
        # Generate matrices using ChaCha20
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0'*8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        # Create and store matrices
        self.matrices = nn.ParameterList()
        for r in range(ROUNDS):
            matrix = torch.zeros((HIDDEN, HIDDEN), dtype=torch.float32, device=self.device)
            start_idx = r * HIDDEN * HIDDEN
            for i in range(HIDDEN):
                for j in range(HIDDEN):
                    val = random_bytes[start_idx + i * HIDDEN + j] % 3
                    matrix[i, j] = val - 1
            self.matrices.append(nn.Parameter(matrix, requires_grad=False))

    def generate_noise(self, nonce):
        """Generate noise bits exactly like C implementation."""
        hash_bytes = sha256(nonce.cpu().numpy().tobytes()).digest()
        noise = np.zeros((ROUNDS * HIDDEN), dtype=np.float32)
        
        for i in range(ROUNDS * HIDDEN):
            byte_idx = i // 8  # Integer division like C
            bit_idx = i % 8   # Exact C bit indexing
            if byte_idx < 32:  # Stay within hash bounds
                noise[i] = (hash_bytes[byte_idx] >> bit_idx) & 1
            
        return torch.from_numpy(noise).reshape(ROUNDS, HIDDEN).to(self.device)

    def forward(self, nonces):
        batch_size = nonces.shape[0]
        x = torch.zeros((batch_size, HIDDEN), dtype=torch.float32, device=self.device)
        
        # Convert bytes to bits like C
        nonces_cpu = nonces.cpu().numpy()
        for batch in range(batch_size):
            for i in range(IN_SIZE):
                for j in range(8):
                    x[batch, i*8 + j] = (nonces_cpu[batch, i] >> j) & 1
        
        # Generate noise for each nonce
        noise = torch.zeros((batch_size, ROUNDS, HIDDEN), dtype=torch.float32, device=self.device)
        for batch in range(batch_size):
            noise[batch] = self.generate_noise(nonces[batch])
        
        # Apply rounds
        for r in range(ROUNDS):
            # Matrix multiplication like C: Ax
            matmul = torch.matmul(self.matrices[r], x.T).T
            # Add noise and mod 2 like C
            x = torch.fmod(matmul + noise[:, r], 2.0)
            
        return x

def bits_to_bytes_batched(bits):
    batch_size = bits.shape[0]
    bytes_out = torch.zeros((batch_size, IN_SIZE), dtype=torch.uint8, device=bits.device)
    bits = (bits > 0.5)
    
    # Convert bits to bytes like C
    for i in range(batch_size):
        for byte_idx in range(IN_SIZE):
            byte = 0
            for bit_idx in range(8):
                if bits[i, byte_idx * 8 + bit_idx]:
                    byte |= 1 << bit_idx
            bytes_out[i, byte_idx] = byte
    return bytes_out

def count_leading_zeros_batched(bits):
    batch_size = bits.shape[0]
    leading_zeros = torch.zeros(batch_size, dtype=torch.int32, device=bits.device)
    bits_cpu = bits.cpu()
    
    # Count leading zeros like C
    for i in range(batch_size):
        zeros = 0
        found_one = False
        for byte_idx in range(IN_SIZE-1, -1, -1):
            for bit_idx in range(7, -1, -1):
                idx = byte_idx * 8 + bit_idx
                if bits_cpu[i, idx] > 0.5:
                    found_one = True
                    leading_zeros[i] = zeros
                    break
                zeros += 1
            if found_one:
                break
        if not found_one:
            leading_zeros[i] = HIDDEN
    return leading_zeros

class TensPoW:
    def __init__(self, seed_hex, difficulty, device="cpu"):
        self.device = device
        self.model = TensHash(seed_hex, device).to(device)
        self.difficulty = difficulty
        
    def mine(self, batch_size=1024):
        counter = 0
        start_time = time.time()
        last_report_time = start_time
        total_hashes = 0
        best_zeros = 0
        
        print(f"Mining with {self.device} implementation:")
        print(f"  Seed: {self.model.seed.hex()}")
        print(f"  Difficulty: {self.difficulty} leading 0 bits")
        print(f"  Batch size: {batch_size}")
        print("\nProgress:")
        print("  Time    Hash Rate      TOPS         Total Hashes    Best Bits")
        print("  ----    ---------    --------      ------------    ----------")
        
        try:
            nonces = torch.zeros((batch_size, IN_SIZE), dtype=torch.uint8, device=self.device)
            
            while True:
                # Set nonces for this batch
                for i in range(batch_size):
                    nonce = counter + i
                    for byte_idx in range(8):
                        nonces[i, byte_idx] = (nonce >> (8 * byte_idx)) & 0xFF

                # Run hash in batches
                with torch.no_grad():
                    output_bits = self.model(nonces)
                
                # Check results
                zeros = count_leading_zeros_batched(output_bits)
                max_zeros = zeros.max().item()
                best_zeros = max(best_zeros, max_zeros)
                
                # Check for solution
                if max_zeros >= self.difficulty:
                    success_idx = (zeros >= self.difficulty).nonzero()[0][0].item()
                    winning_nonce = nonces[success_idx]
                    output_bytes = bits_to_bytes_batched(output_bits)[success_idx]
                    
                    print("\nSolution found!")
                    print(f"Nonce: {winning_nonce.cpu().numpy().tobytes().hex()}")
                    print(f"Hash: {output_bytes.cpu().numpy().tobytes().hex()}")
                    
                    total_time = time.time() - start_time
                    hash_rate = total_hashes / total_time if total_time > 0 else 0
                    tops = (hash_rate * HIDDEN * HIDDEN * 2 * ROUNDS) / 1e12
                    
                    print(f"\nStats:")
                    print(f"Time: {total_time:.1f} seconds")
                    print(f"Total hashes: {total_hashes:,}")
                    print(f"Hash rate: {hash_rate:,.1f} H/s")
                    print(f"Avg TOPS: {tops:.6f}")
                    return winning_nonce
                
                # Update stats
                total_hashes = counter + batch_size
                current_time = time.time()
                if current_time - last_report_time >= 1.0:
                    interval = current_time - last_report_time
                    hash_rate = batch_size / interval
                    tops = (hash_rate * HIDDEN * HIDDEN * 2 * ROUNDS) / 1e12
                    total_time = current_time - start_time
                    print(f"  {total_time:4.0f}s    {hash_rate:7.0f} h/s    {tops:.6f}    {total_hashes:12d}    {best_zeros:10d}", flush=True)
                    last_report_time = current_time
                
                counter += batch_size
                
        except KeyboardInterrupt:
            print("\nMining stopped by user")
            return None

def main():
    parser = argparse.ArgumentParser(description='TensPoW PyTorch Miner')
    parser.add_argument('seed', help='64-character hex seed')
    parser.add_argument('difficulty', type=int, help='Required number of leading zero bits')
    parser.add_argument('--device', default='cpu', help='Device to use (mps/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (default: 1024)')
    args = parser.parse_args()
    
    if len(args.seed) != 64:
        parser.error("Seed must be 64 hex characters")
    if args.difficulty < 1 or args.difficulty > 256:
        parser.error("Difficulty must be between 1 and 256")
    
    miner = TensPoW(args.seed, args.difficulty, args.device)
    miner.mine(batch_size=args.batch_size)

if __name__ == "__main__":
    main()