#!/usr/bin/env python3
import argparse
import time
from typing import List, Tuple
import numpy as np
from Crypto.Cipher import ChaCha20
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# Network dimensions
INPUT_SIZE = 256      # Input vector size
HIDDEN_SIZE = 1024    # Hidden layer size
NUM_HIDDEN_LAYERS = 64
BATCH_SIZE = 2048     # Batch size for the Core ML model
NUM_NONZERO = 128
MODEL_PATH = "tens_hash.mlpackage"

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

def generate_model(seed: bytes):
    """Generate the CoreML model."""
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
    
    # Define the MIL program
    input_specs = [mb.TensorSpec(shape=(INPUT_SIZE, BATCH_SIZE))]
    
    @mb.program(input_specs=input_specs)
    def transform_prog(input):
        x = input
        for idx, (name, mat) in enumerate(layers):
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            dot = mb.matmul(x=mb.const(val=mat), y=x_mapped)
            if name == "hidden":
                dot = mb.add(x=dot, y=x_mapped)
            x = mb.clip(x=dot, alpha=0.0, beta=1.0)
        return x
    
    mlmodel = ct.convert(
        transform_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="output")]
    )
    
    mlmodel.save(MODEL_PATH)
    print("Model generated and saved.")

class ProofOfWorkMiner:
    def __init__(self, target_difficulty: int):
        self.target_difficulty = target_difficulty
        
        # Load CoreML model with compute units set to ALL
        self.model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.ALL)
        
        # Initialize tracking variables
        self.nonce = 0
        self.total_hashes = 0
        self.start_time = time.time()
        self.best_difficulty = 0
        
        # Pre-allocate arrays for batch processing
        self.binary_input = np.zeros((INPUT_SIZE, BATCH_SIZE), dtype=np.float32)
        self.nonce_bytes = np.zeros((BATCH_SIZE, 32), dtype=np.uint8)  # 32 bytes per nonce

    def prepare_batch(self) -> None:
        """Prepare a batch of inputs for the model."""
        nonces = np.arange(self.nonce, self.nonce + BATCH_SIZE, dtype=np.uint64)
        for i in range(8):
            self.nonce_bytes[:, 24 + i] = (nonces >> (8 * (7 - i))) & 0xFF
        bits = np.unpackbits(self.nonce_bytes, axis=1)
        self.binary_input = bits.reshape(BATCH_SIZE, -1)[:, :INPUT_SIZE].T.astype(np.float32)

    def run_inference(self) -> np.ndarray:
        """Run model inference on a batch of inputs."""
        outputs = self.model.predict({'input': self.binary_input})
        return np.asarray(outputs['clip_65']).T

    def count_leading_zeros(self, output: np.ndarray) -> int:
        """Count leading zeros in binary output."""
        nonzero_indices = np.nonzero(output)[0]
        return nonzero_indices[0] if len(nonzero_indices) > 0 else INPUT_SIZE

    def print_status(self):
        """Print current mining status."""
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed
        tops = (hashrate * (NUM_HIDDEN_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * INPUT_SIZE * 4)) / 1e12
        print(f"Nonce: {self.nonce} | Hashrate: {hashrate:.2f} H/s | "
              f"TOPS: {tops:.2f} | Best difficulty: {self.best_difficulty}")

    def check_solution(self, binary_output: np.ndarray, batch_index: int) -> bool:
        """Check if a solution meets the target difficulty."""
        zeros = self.count_leading_zeros(binary_output[batch_index])
        self.best_difficulty = max(self.best_difficulty, zeros)
        
        if zeros >= self.target_difficulty:
            solution_nonce = self.nonce + batch_index
            print("\nSolution found!")
            print(f"Nonce: {solution_nonce}")
            print(f"Leading zeros: {zeros}")
            print(f"Solution input (hex): {bytes(self.nonce_bytes[batch_index]).hex()}")
            output_bytes = np.packbits(binary_output[batch_index].astype(bool))
            print(f"Model output (hex): {bytes(output_bytes).hex()}")
            return True
        return False

    def mine(self):
        """Main mining loop."""
        print(f"Mining with difficulty: {self.target_difficulty}")
        last_status_time = time.time()
        
        while True:
            self.prepare_batch()
            binary_output = self.run_inference()
            
            for i in range(BATCH_SIZE):
                if self.check_solution(binary_output, i):
                    return
            
            self.nonce += BATCH_SIZE
            self.total_hashes += BATCH_SIZE
            
            if time.time() - last_status_time >= 1.0:
                self.print_status()
                last_status_time = time.time()

def main():
    parser = argparse.ArgumentParser(description="CoreML Proof of Work")
    parser.add_argument("seed", type=str, help="32-byte hex seed (64 hex characters)")
    parser.add_argument("difficulty", type=int, help="Target difficulty (number of leading zeros)")
    args = parser.parse_args()
    
    # Generate model from seed
    seed = hex_to_bytes(args.seed)
    generate_model(seed)
    
    # Start mining
    miner = ProofOfWorkMiner(args.difficulty)
    miner.mine()

if __name__ == "__main__":
    main()
