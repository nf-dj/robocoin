#!/usr/bin/env python3
import argparse
import time
from typing import List, Tuple
import numpy as np
from Crypto.Cipher import ChaCha20
import tensorflow as tf

# Network dimensions
INPUT_SIZE = 256      # Input vector size
HIDDEN_SIZE = 1024    # Hidden layer size
NUM_HIDDEN_LAYERS = 64
BATCH_SIZE = 2048     # Batch size for inference
NUM_NONZERO = 128

# Enable Metal
tf.config.list_physical_devices('GPU')  # This will initialize Metal
print("TensorFlow devices:", tf.config.list_physical_devices())

class HashNetwork(tf.keras.Model):
    def __init__(self, matrices):
        super().__init__()
        
        # Create variables with consistent FP16 type
        def create_variable(mat):
            return tf.Variable(
                tf.cast(mat, tf.float16),
                trainable=False,
                dtype=tf.float16
            )
        
        self.expansion = create_variable(matrices[0][1])
        self.hidden_layers = [create_variable(matrices[i+1][1]) for i in range(NUM_HIDDEN_LAYERS)]
        self.compression = create_variable(matrices[-1][1])

    @tf.function(jit_compile=True)  # Enable XLA compilation
    def call(self, x):  # x shape: (INPUT_SIZE, BATCH_SIZE)
        # Ensure input is float16
        x = tf.cast(x, tf.float16)
        
        # Map input from {0,1} to {-1,1}
        x = 2.0 * x - 1.0
        
        # Expansion layer
        x = tf.matmul(self.expansion, x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            x_mapped = 2.0 * x - 1.0
            x = tf.clip_by_value(tf.matmul(layer, x_mapped) + x_mapped, 0.0, 1.0)
        
        # Compression layer
        x = tf.clip_by_value(tf.matmul(self.compression, (2.0 * x - 1.0)), 0.0, 1.0)
        
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
    
    print(f"Generated layers")
    
    # Create and compile model
    model = HashNetwork(layers)
    # Warm up XLA
    dummy_input = tf.zeros((INPUT_SIZE, BATCH_SIZE), dtype=tf.float16)
    _ = model(dummy_input)
    
    return model

class ProofOfWorkMiner:
    def __init__(self, model: HashNetwork, target_difficulty: int):
        self.model = model
        self.target_difficulty = target_difficulty
        
        # Initialize tracking variables
        self.nonce = 0
        self.total_hashes = 0
        self.start_time = time.time()
        self.best_difficulty = 0
        
        # Pre-allocate arrays
        self.nonce_bytes = np.zeros((BATCH_SIZE, 32), dtype=np.uint8)
        self.binary_input = tf.Variable(
            tf.zeros((INPUT_SIZE, BATCH_SIZE), dtype=tf.float16),
            trainable=False,
            dtype=tf.float16
        )

    def prepare_batch(self) -> None:
        """Prepare a batch of inputs for the model."""
        nonces = np.arange(self.nonce, self.nonce + BATCH_SIZE, dtype=np.uint64)
        for i in range(8):
            self.nonce_bytes[:, 24 + i] = (nonces >> (8 * (7 - i))) & 0xFF
            
        # Convert to binary and ensure FP16
        bits = np.unpackbits(self.nonce_bytes, axis=1)[:, :INPUT_SIZE].T.astype(np.float16)
        # Update tensor in place
        self.binary_input.assign(bits)

    def count_leading_zeros(self, output: tf.Tensor) -> int:
        """Count leading zeros in binary output."""
        binary = tf.cast(output > 0.5, tf.int32).numpy()
        nonzero_indices = np.nonzero(binary)[0]
        return nonzero_indices[0] if len(nonzero_indices) > 0 else INPUT_SIZE

    def print_status(self):
        """Print current mining status."""
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed
        tops = (hashrate * (NUM_HIDDEN_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * INPUT_SIZE * 4)) / 1e12
        print(f"Nonce: {self.nonce} | Hashrate: {hashrate:.2f} H/s | "
              f"TOPS: {tops:.2f} | Best difficulty: {self.best_difficulty}")

    def check_solution(self, binary_output: tf.Tensor, batch_index: int) -> bool:
        """Check if a solution meets the target difficulty."""
        column = binary_output[:, batch_index]
        zeros = self.count_leading_zeros(column)
        self.best_difficulty = max(self.best_difficulty, zeros)
        
        if zeros >= self.target_difficulty:
            solution_nonce = self.nonce + batch_index
            print("\nSolution found!")
            print(f"Nonce: {solution_nonce}")
            print(f"Leading zeros: {zeros}")
            print(f"Solution input (hex): {bytes(self.nonce_bytes[batch_index]).hex()}")
            
            # Convert output to bytes
            output_bits = tf.cast(column > 0.5, tf.int32).numpy().astype(np.uint8)
            output_bytes = np.packbits(output_bits)
            print(f"Model output (hex): {bytes(output_bytes).hex()}")
            return True
        return False

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
    parser = argparse.ArgumentParser(description="TensorFlow Proof of Work with Metal")
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
