import torch
import torch.nn as nn
import time
import argparse
from hashlib import sha256
from Crypto.Cipher import ChaCha20

IN_SIZE = 32
HIDDEN = 256
ROUNDS = 64

class TensHash(nn.Module):
    def __init__(self, seed_hex, device="mps"):
        super().__init__()
        self.device = torch.device(device if torch.backends.mps.is_available() and device == "mps" else "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = bytes.fromhex(seed_hex)
        
        # Generate matrices using ChaCha20
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0'*8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        # Create and store matrices as nn.Parameters
        self.matrices = nn.ParameterList([
            nn.Parameter(
                torch.tensor(
                    [(random_bytes[r*HIDDEN*HIDDEN + i] % 3) - 1 for i in range(HIDDEN*HIDDEN)],
                    dtype=torch.float32, device=self.device
                ).reshape(HIDDEN, HIDDEN),
                requires_grad=False
            ) for r in range(ROUNDS)
        ])

    def bytes_to_bits_batched(self, nonces):
        """Convert bytes to bit array in batches."""
        batch_size = nonces.shape[0]
        bits = torch.zeros((batch_size, HIDDEN), dtype=torch.float32, device=self.device)
        nonces_cpu = nonces.cpu().numpy()
        
        for i in range(batch_size):
            for byte_idx in range(IN_SIZE):
                for bit_idx in range(8):
                    bits[i, byte_idx * 8 + bit_idx] = (nonces_cpu[i, byte_idx] >> bit_idx) & 1
        
        return bits.to(torch.float32)

    def generate_noise_batched(self, nonces):
        """Generate noise from input in batches."""
        batch_size = nonces.shape[0]
        noise = torch.zeros((batch_size, ROUNDS, HIDDEN), dtype=torch.float32, device=self.device)
        
        # Process on CPU for SHA256
        nonces_cpu = nonces.cpu().numpy()
        for i in range(batch_size):
            hash_bytes = sha256(nonces_cpu[i].tobytes()).digest()
            for r in range(ROUNDS):
                for h in range(HIDDEN):
                    idx = (r * HIDDEN + h)
                    byte_idx = idx % 32
                    bit_idx = idx % 8
                    noise[i, r, h] = (hash_bytes[byte_idx] >> bit_idx) & 1
        
        return noise.to(torch.float32)

    def forward(self, nonces):
        """Process entire batch at once."""
        batch_size = nonces.shape[0]
        
        # Convert input to bits and generate noise (only byte/bit ops)
        x = self.bytes_to_bits_batched(nonces)
        noise = self.generate_noise_batched(nonces)
        
        # All operations in FP32
        for r in range(ROUNDS):
            x = torch.fmod(
                torch.matmul(x, self.matrices[r].T) + noise[:, r],
                2.0
            )
            
        return x

def count_leading_zeros_fast(bits):
    """Optimized leading zeros counting directly on GPU."""
    # Threshold and convert to binary (0/1)
    bits = (bits > 0.5).float()
    
    # Get the position of the first 1 in each row
    # We want to find the last 1 when bits are in normal order, so we flip the order
    flipped_bits = torch.flip(bits, [1])
    first_one = torch.argmax(flipped_bits, dim=1)
    
    # Convert to leading zeros count
    leading_zeros = HIDDEN - first_one - 1
    
    # Handle case where all bits are zero
    all_zeros = (bits.sum(dim=1) == 0)
    leading_zeros[all_zeros] = HIDDEN
    
    return leading_zeros.int()

class TensPoW:
    def __init__(self, seed_hex, difficulty, device="mps"):
        self.device = device
        self.model = TensHash(seed_hex, device).to(device)
        self.difficulty = difficulty
        
    def mine(self, batch_size=1024):
        """Mine for valid nonce using batched operations."""
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
            nonces = torch.zeros((batch_size, IN_SIZE), dtype=torch.uint8, device=self.model.device)
            
            while True:
                # Set nonces for this batch
                for i in range(batch_size):
                    nonce = counter + i
                    for byte_idx in range(8):
                        nonces[i, byte_idx] = (nonce >> (byte_idx * 8)) & 0xFF
                
                # Run hash in batches
                with torch.no_grad():
                    output_bits = self.model(nonces)
                
                # Check results directly on bits
                zeros = count_leading_zeros_fast(output_bits)
                max_zeros = zeros.max().item()
                best_zeros = max(best_zeros, max_zeros)
                
                # Check for solution
                if max_zeros >= self.difficulty:
                    success_idx = (zeros >= self.difficulty).nonzero()[0][0].item()
                    winning_nonce = nonces[success_idx]
                    
                    print("\nSolution found!")
                    print(f"Nonce: {winning_nonce.cpu().numpy().tobytes().hex()}")
                    
                    total_time = time.time() - start_time
                    hash_rate = total_hashes / total_time
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
    parser.add_argument('--device', default='mps', help='Device to use (mps/cuda/cpu)')
    args = parser.parse_args()
    
    if len(args.seed) != 64:
        parser.error("Seed must be 64 hex characters")
    if args.difficulty < 1 or args.difficulty > 256:
        parser.error("Difficulty must be between 1 and 256")
    
    miner = TensPoW(args.seed, args.difficulty, args.device)
    miner.mine()

if __name__ == "__main__":
    main()