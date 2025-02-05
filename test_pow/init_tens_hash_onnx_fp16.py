import torch
import torch.nn as nn
import time
import argparse
from hashlib import sha256
import numpy as np
from Crypto.Cipher import ChaCha20

# Constants matching C implementation
IN_SIZE = 32
HIDDEN = 256
ROUNDS = 64

# Use MPS (Metal Performance Shaders) if available, otherwise fallback to CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Allow FP16 execution but fallback to FP32 if error occurs
use_fp16 = True  # Try FP16 first
dtype = torch.float16 if use_fp16 else torch.float32

class TernaryLinear(nn.Module):
    """Custom Linear Layer with Ternary Weights (-1, 0, 1) and Binary Bias (0,1)"""
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype=dtype).to(device), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((out_features,), dtype=dtype).to(device), requires_grad=False)

    def forward(self, x):
        try:
            result = torch.matmul(x, self.weight.T) + self.bias
            result = torch.fmod(result, 2.0)
            return result
        except RuntimeError as e:
            if "trunc_divide op with float16 input" in str(e):
                print("⚠️ FP16 failed! Falling back to FP32.")
                global use_fp16, dtype
                use_fp16 = False
                dtype = torch.float32
                self.weight = nn.Parameter(self.weight.to(dtype), requires_grad=False)
                self.bias = nn.Parameter(self.bias.to(dtype), requires_grad=False)
                x = x.to(dtype)
                result = torch.matmul(x, self.weight.T) + self.bias
                result = torch.fmod(result, 2.0)
                return result

class TensPowModel(nn.Module):
    def __init__(self, seed):
        super(TensPowModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Generate matrices using ChaCha20 like in C implementation
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=seed, nonce=b'\0'*8)
        random_bytes = cipher.encrypt(b'\0' * total_size)

        # Convert bytes to matrices
        for r in range(ROUNDS):
            layer = TernaryLinear(HIDDEN, HIDDEN)
            start_idx = r * HIDDEN * HIDDEN
            mat_bytes = random_bytes[start_idx:start_idx + HIDDEN * HIDDEN]
            
            # Convert to ternary values exactly as in C implementation
            weights = torch.tensor(
                [(x % 3) - 1 for x in mat_bytes], 
                dtype=dtype
            ).reshape(HIDDEN, HIDDEN).to(device)
            
            # Set weights and biases
            layer.weight.data = weights
            # Note: Original C implementation uses ternary weights but binary bias
            layer.bias.data = torch.zeros(HIDDEN, dtype=dtype).to(device)
            
            self.layers.append(layer)

    def forward(self, x, noise):
        # x: batch of input values
        # noise: precomputed noise for each round
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.fmod(x + noise[:, i, :], 2.0)
        return x

def generate_noise(input_batch):
    """Generate deterministic noise for each input in batch using SHA256"""
    batch_size = input_batch.shape[0]
    noise = torch.zeros((batch_size, ROUNDS, HIDDEN), dtype=dtype).to(device)
    
    for i in range(batch_size):
        input_bytes = input_batch[i].cpu().byte().numpy().tobytes()
        hash_bytes = sha256(input_bytes).digest()
        # Expand hash to get noise for all rounds
        for r in range(ROUNDS):
            for h in range(HIDDEN):
                byte_idx = (r * HIDDEN + h) % 32
                bit_idx = (r * HIDDEN + h) % 8
                noise[i, r, h] = (hash_bytes[byte_idx] >> bit_idx) & 1
                
    return noise

def count_leading_zeros(hash_tensor):
    """Count leading zeros in binary representation"""
    # Convert to CPU for bit manipulation
    hash_bytes = hash_tensor.cpu().byte().numpy()
    
    leading_zeros = torch.zeros(hash_bytes.shape[0], dtype=torch.int)
    for i in range(hash_bytes.shape[0]):
        zeros = 0
        for byte in reversed(hash_bytes[i]):
            if byte == 0:
                zeros += 8
                continue
            # Count remaining zeros in this byte
            for bit in range(7, -1, -1):
                if byte & (1 << bit):
                    break
                zeros += 1
            break
        leading_zeros[i] = zeros
    return leading_zeros

def bytes_to_bits(input_bytes, batch_size):
    """Convert batch of byte arrays to bit arrays"""
    result = torch.zeros((batch_size, HIDDEN), dtype=dtype).to(device)
    for i in range(batch_size):
        for byte_idx in range(IN_SIZE):
            for bit_idx in range(8):
                bit = (input_bytes[i, byte_idx] >> bit_idx) & 1
                result[i, byte_idx * 8 + bit_idx] = bit
    return result

def bits_to_bytes(bits):
    """Convert bit array back to bytes"""
    batch_size = bits.shape[0]
    result = torch.zeros((batch_size, IN_SIZE), dtype=torch.uint8).to(device)
    bits = bits.bool()
    
    for i in range(batch_size):
        for byte_idx in range(IN_SIZE):
            byte = 0
            for bit_idx in range(8):
                if bits[i, byte_idx * 8 + bit_idx]:
                    byte |= 1 << bit_idx
            result[i, byte_idx] = byte
    
    return result

def main():
    parser = argparse.ArgumentParser(description='TensPoW GPU Miner')
    parser.add_argument('seed', help='64-character hex seed')
    parser.add_argument('difficulty', type=int, help='Required number of leading zero bits')
    
    args = parser.parse_args()
    
    if len(args.seed) != 64:
        parser.error("Seed must be 64 hex characters")
    if args.difficulty < 1 or args.difficulty > 256:
        parser.error("Difficulty must be between 1 and 256")

    # Convert hex seed to bytes
    seed_bytes = bytes.fromhex(args.seed)
    batch_size = 1024  # Can be tuned based on GPU memory
        
    print(f"Mining with {device} implementation:")
    print(f"  Seed: {args.seed}")
    print(f"  Difficulty: {args.difficulty} leading 0 bits")
    print(f"  Batch size: {batch_size}")
    print("\nProgress:")
    print("  Time    Hash Rate      TOPS         Total Hashes    Best Bits")
    print("  ----    ---------    --------      ------------    ----------")

    # Initialize model
    model = TensPowModel(seed_bytes).to(device)
    
    # Statistics tracking
    start_time = time.time()
    last_report_time = start_time
    total_hashes = 0
    best_zeros = 0
    
    try:
        counter = 0
        while True:
            # Generate batch of nonces
            nonces = torch.zeros((batch_size, IN_SIZE), dtype=torch.uint8).to(device)
            for i in range(batch_size):
                nonce = counter + i
                for byte_idx in range(8):  # Only use first 8 bytes for nonce
                    nonces[i, byte_idx] = (nonce >> (byte_idx * 8)) & 0xFF
            
            # Convert nonces to bits
            input_bits = bytes_to_bits(nonces, batch_size)
            
            # Generate noise for each input in batch
            noise = generate_noise(nonces)
            
            # Run the model
            output_bits = model(input_bits, noise)
            
            # Convert output bits back to bytes
            hashes = bits_to_bytes(output_bits)
            
            # Count leading zeros for each hash
            zeros = count_leading_zeros(hashes)
            max_zeros = zeros.max().item()
            best_zeros = max(best_zeros, max_zeros)
            
            # Check if any hash meets difficulty
            if max_zeros >= args.difficulty:
                success_idx = (zeros >= args.difficulty).nonzero()[0][0].item()
                winning_nonce = nonces[success_idx]
                winning_hash = hashes[success_idx]
                
                # Print results
                print("\nSolution found!")
                print(f"Nonce: {winning_nonce.cpu().numpy().hex()}")
                print(f"Hash:  {winning_hash.cpu().numpy().hex()}")
                
                total_time = time.time() - start_time
                total_hashes += counter + success_idx
                hash_rate = total_hashes / total_time
                tops = (hash_rate * HIDDEN * HIDDEN * 2 * ROUNDS) / 1e12
                
                print(f"\nStats:")
                print(f"Time: {total_time:.1f} seconds")
                print(f"Total hashes: {total_hashes:,}")
                print(f"Hash rate: {hash_rate:,.1f} H/s")
                print(f"Avg TOPS: {tops:.6f}")
                break
            
            # Update counter and stats
            counter += batch_size
            total_hashes += batch_size
            
            # Progress report every second
            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                interval = current_time - last_report_time
                hash_rate = batch_size / interval
                tops = (hash_rate * HIDDEN * HIDDEN * 2 * ROUNDS) / 1e12
                
                total_time = current_time - start_time
                print(f"  {total_time:4.0f}    {hash_rate:7.0f} h/s    {tops:.6f}    {total_hashes:12}    {best_zeros:10}", end="\r")
                
                last_report_time = current_time
                
    except KeyboardInterrupt:
        print("\nMining stopped by user")

if __name__ == "__main__":
    main()
