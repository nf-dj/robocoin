import argparse
import time
from typing import List, Tuple
import numpy as np
import coremltools as ct

# Constants matching the original implementation
INPUT_SIZE = 32        # 32 bytes per nonce input
VECTOR_SIZE = 256      # 256 bits per sample
HIDDEN_SIZE = 1024
BATCH_SIZE = 2048
ROUNDS = 64

class ProofOfWorkMiner:
    def __init__(self, model_path: str, target_difficulty: int, debug: bool = False):
        self.target_difficulty = target_difficulty
        self.debug = debug
        
        # Load CoreML model
        spec = ct.utils.load_spec(model_path)
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        
        # Initialize tracking variables
        self.nonce = 0
        self.total_hashes = 0
        self.start_time = time.time()
        self.best_difficulty = 0
        
        # Pre-allocate arrays for batch processing
        self.binary_input = np.zeros((VECTOR_SIZE, BATCH_SIZE), dtype=np.float32)
        self.nonce_bytes = np.zeros((BATCH_SIZE, INPUT_SIZE), dtype=np.uint8)

    def prepare_batch(self) -> None:
        """Prepare a batch of inputs for the model."""
        # Generate sequential nonces efficiently using numpy
        nonces = np.arange(self.nonce, self.nonce + BATCH_SIZE, dtype=np.uint64)
        
        # Convert nonces to bytes (last 8 bytes of each 32-byte input)
        for i in range(8):
            self.nonce_bytes[:, 24 + i] = (nonces >> (8 * (7 - i))) & 0xFF
            
        # Convert to binary representation efficiently
        bits = np.unpackbits(self.nonce_bytes, axis=1)
        self.binary_input = bits.reshape(BATCH_SIZE, -1)[:, :VECTOR_SIZE].T.astype(np.float32)

    def run_inference(self) -> np.ndarray:
        """Run model inference on a batch of inputs."""
        # Create input dictionary and run inference
        outputs = self.model.predict({'input': self.binary_input})
        
        # Get output array and ensure correct shape
        output_array = outputs['clip_65']
        return np.asarray(output_array).T

    def count_leading_zeros(self, output: np.ndarray) -> int:
        """Count leading zeros in binary output."""
        nonzero_indices = np.nonzero(output)[0]
        return nonzero_indices[0] if len(nonzero_indices) > 0 else VECTOR_SIZE

    def print_debug_info(self, binary_output: np.ndarray):
        """Print debug information for the first sample in batch."""
        print(f"Nonce: {bytes(self.nonce_bytes[0]).hex()}")
        print(f"Input vector: [{', '.join(map(str, self.binary_input[:, 0][:20]))}...]")
        print(f"Output vector: [{', '.join(map(str, binary_output[0, :20]))}...]")
        
        # Convert output bits to bytes
        output_bytes = np.packbits(binary_output[0].astype(bool))
        print(f"Output (hex): {bytes(output_bytes).hex()}")

    def print_status(self):
        """Print current mining status."""
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed
        # Calculate TOPS (Tera Operations Per Second)
        tops = (hashrate * (ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * VECTOR_SIZE * 4)) / 1e12
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
        print(f"Mining with difficulty: {self.target_difficulty}"
              f"{' (Debug mode enabled)' if self.debug else ''}")
        
        last_status_time = time.time()
        
        while True:
            # Prepare batch using pre-allocated arrays
            self.prepare_batch()
            
            # Run inference
            binary_output = self.run_inference()
            
            # Debug output for first sample
            if self.debug:
                self.print_debug_info(binary_output)
            
            # Check each sample in batch
            for i in range(BATCH_SIZE):
                if self.check_solution(binary_output, i):
                    return
            
            # Update counters
            self.nonce += BATCH_SIZE
            self.total_hashes += BATCH_SIZE
            
            # Print status every second
            if time.time() - last_status_time >= 1.0:
                self.print_status()
                last_status_time = time.time()

def main():
    parser = argparse.ArgumentParser(description='CoreML Proof of Work Miner')
    parser.add_argument('difficulty', type=int, help='Target difficulty (number of leading zeros)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--model', default='test_coreml.mlpackage', 
                      help='Path to CoreML model package')
    args = parser.parse_args()

    miner = ProofOfWorkMiner(args.model, args.difficulty, args.debug)
    miner.mine()

if __name__ == '__main__':
    main()
