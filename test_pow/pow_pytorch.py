#!/usr/bin/env python3
import argparse
import time
from typing import List, Tuple
import numpy as np
from Crypto.Cipher import ChaCha20
import torch
import torch.nn as nn

# Network dimensions (same as CoreML version)
INPUT_SIZE = 256      # Input vector size
HIDDEN_SIZE = 1024    # Hidden layer size
NUM_HIDDEN_LAYERS = 64
BATCH_SIZE = 2048     # Batch size for inference
NUM_NONZERO = 128

class HashNetwork(nn.Module):
    def __init__(self, matrices):
        super().__init__()
        
        # Create layers from matrices
        self.expansion = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)
        self.expansion.weight.data = torch.from_numpy(matrices[0][1])
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False) 
            for _ in range(NUM_HIDDEN_LAYERS)
        ])
        for i, layer in enumerate(self.hidden_layers):
            layer.weight.data = torch.from_numpy(matrices[i + 1][1])
            
        self.compression = nn.Linear(HIDDEN_SIZE, INPUT_SIZE, bias=False)
        self.compression.weight.data = torch.from_numpy(matrices[-1][1])

    def forward(self, x):
        # Map input from {0,1} to {-1,1}
        x = 2.0 * x - 1.0
        
        # Expansion layer
        x = self.expansion(x)
        x = torch.clip(x, 0.0, 1.0)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            identity = x
            x = 2.0 * x - 1.0
            x = layer(x)
            x = x + identity  # Residual connection
            x = torch.clip(x, 0.0, 1.0)
        
        # Compression layer
        x = 2.0 * x - 1.0
        x = self.compression(x)
        x = torch.clip(x, 0.0, 1.0)
        
        return x

def hex_to_bytes(hex_str):
    """Convert hex string to bytes."""
    b = bytes.fromhex(hex_str)
    if len(b) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex characters)")
    return b

def generate_dense_matrix(rows, cols, key, nonce_int):
    """Generate a constant matrix using ChaCha20-based RNG."""
    nonce = nonce_int.to_bytes(8, byteorder='big')
    cipher = ChaCha20.new(key=key, nonce=nonce)
    
    needed = rows * cols
    random_bytes = cipher.encrypt(b'\x00' * needed)
    data = np.frombuffer(random_bytes, dtype=np.uint8)
    
    mods = data % 4
    mapping = np.empty_like(mods, dtype=np.int8)
    mapping[mods == 0] = 0
    mapping[mods == 1] = 0
    mapping[mods == 2] = 1
    mapping[mods == 3] = -1
    
    return mapping.reshape((rows, cols)).astype(np.float32)

def generate_model(seed: bytes) -> HashNetwork:
    """Generate the model."""
    print("Generating model...")
    layers = []
    nonce_counter = 0
    
    # Expansion layer
    expansion_mat = generate_dense_matrix(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter)
    nonce_counter += 1
    layers.append(('expansion', expansion_mat))
    
    # Hidden layers
    for i in range(NUM_HIDDEN_LAYERS):
        mat_hidden = generate_dense_matrix(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter)
        nonce_counter += 1
        layers.append(('hidden', mat_hidden))
    
    # Compression layer
    compression_mat = generate_dense_matrix(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter)
    nonce_counter += 1
    layers.append(('compression', compression_mat))
    
    return HashNetwork(layers)

class ProofOfWorkMiner:
    def __init__(self, model: HashNetwork, target_difficulty: int):
        self.target_difficulty = target_difficulty
        self.model = model
        
        # Use MPS (Metal) if available, otherwise CUDA, otherwise CPU
        if torch.backends.mps.is_available():
            print("Using MPS (Metal) device")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA device")
            self.device = torch.device("cuda")
        else:
            print("Using CPU device")
            self.device = torch.device("cpu")
            
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize tracking variables
        self.nonce = 0
        self.total_hashes = 0
        self.start_time = time.time()
        self.best_difficulty = 0
        
        # Pre-allocate arrays for batch processing
        self.nonce_bytes = torch.zeros((BATCH_SIZE, 32), dtype=torch.uint8)
        self.binary_input = torch.zeros((BATCH_SIZE, INPUT_SIZE), dtype=torch.float32, device=self.device)

    def prepare_batch(self) -> None:
        """Prepare a batch of inputs for the model."""
        # Generate sequential nonces using numpy first
        nonces = np.arange(self.nonce, self.nonce + BATCH_SIZE, dtype=np.uint64)
        nonce_bytes = np.zeros((BATCH_SIZE, 32), dtype=np.uint8)
        
        # Convert nonces to bytes (last 8 bytes of each 32-byte input)
        for i in range(8):
            nonce_bytes[:, 24 + i] = (nonces >> (8 * (7 - i))) & 0xFF
            
        # Convert to binary using numpy's unpackbits
        bits = np.unpackbits(nonce_bytes, axis=1)[:, :INPUT_SIZE]
        
        # Convert to torch tensor and move to device
        self.binary_input = torch.from_numpy(bits).float().to(self.device)
        self.nonce_bytes = torch.from_numpy(nonce_bytes)

    def count_leading_zeros(self, output: torch.Tensor) -> int:
        """Count leading zeros in binary output."""
        # Convert to binary (0 or 1)
        binary = (output > 0.5).int()
        nonzeros = binary.nonzero()
        return nonzeros[0].item() if len(nonzeros) > 0 else INPUT_SIZE

    def print_status(self):
        """Print current mining status."""
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed
        tops = (hashrate * (NUM_HIDDEN_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * INPUT_SIZE * 4)) / 1e12
        print(f"Nonce: {self.nonce} | Hashrate: {hashrate:.2f} H/s | "
              f"TOPS: {tops:.2f} | Best difficulty: {self.best_difficulty}")

    def check_solution(self, binary_output: torch.Tensor, batch_index: int) -> bool:
        """Check if a solution meets the target difficulty."""
        zeros = self.count_leading_zeros(binary_output[batch_index])
        self.best_difficulty = max(self.best_difficulty, zeros)
        
        if zeros >= self.target_difficulty:
            solution_nonce = self.nonce + batch_index
            print("\nSolution found!")
            print(f"Nonce: {solution_nonce}")
            print(f"Leading zeros: {zeros}")
            print(f"Solution input (hex): {bytes(self.nonce_bytes[batch_index].numpy()).hex()}")
            
            # Convert output bits to bytes
            # Convert output bits to bytes using numpy
            output_bits = (binary_output[batch_index].cpu().numpy() > 0.5)
            output_bytes = np.packbits(output_bits)
            print(f"Model output (hex): {bytes(output_bytes).hex()}")
            return True
        return False

    @torch.no_grad()  # Disable gradient computation for inference
    def mine(self):
        """Main mining loop."""
        print(f"Mining with difficulty: {self.target_difficulty}")
        last_status_time = time.time()
        
        while True:
            self.prepare_batch()
            binary_output = self.model(self.binary_input)
            
            for i in range(BATCH_SIZE):
                if self.check_solution(binary_output, i):
                    return
            
            self.nonce += BATCH_SIZE
            self.total_hashes += BATCH_SIZE
            
            if time.time() - last_status_time >= 1.0:
                self.print_status()
                last_status_time = time.time()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Proof of Work on Metal")
    parser.add_argument("seed", type=str, help="32-byte hex seed (64 hex characters)")
    parser.add_argument("difficulty", type=int, help="Target difficulty (number of leading zeros)")
    args = parser.parse_args()
    
    # Generate model from seed
    seed = hex_to_bytes(args.seed)
    model = generate_model(seed)
    
    # Start mining
    miner = ProofOfWorkMiner(model, args.difficulty)
    miner.mine()

if __name__ == "__main__":
    main()
