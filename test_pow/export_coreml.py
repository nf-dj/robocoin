import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse
from Crypto.Cipher import ChaCha20

#ROUNDS=64
ROUNDS=1
#BATCH_SIZE = 8192
BATCH_SIZE = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Export CoreML model with seed-based matrix generation")
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num):
    """Generate a 256x256 ternary matrix using the same method as tens_hash_np.py but with round number in nonce"""
    input_size, output_size = 256, 256
    pos_count = neg_count = 32
    total_nonzero = pos_count + neg_count
    
    # Use round number in nonce
    nonce = round_num.to_bytes(4, 'big') + b'\x00' * 4  # 4 bytes for round, 4 bytes zeros
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    
    # Generate all random values at once using ChaCha20
    total_rand_vals = input_size * output_size
    rand_bytes = cipher.encrypt(b'\x00' * (total_rand_vals * 4))
    rand_vals = np.frombuffer(rand_bytes, dtype=np.uint32).reshape(input_size, output_size)
    
    # Initialize the matrix
    A = np.zeros((input_size, output_size), dtype=np.float32)
    
    # Pre-generate sign array
    base_signs = np.array([1] * pos_count + [-1] * neg_count, dtype=np.float32)
    
    # Process each row
    for i in range(input_size):
        # Sort indices based on random values
        chosen_indices = np.argsort(rand_vals[i])[:total_nonzero]
        
        # Place signs at sorted positions
        A[i, chosen_indices] = base_signs
    
    # Verify the matrix
    for i in range(input_size):
        pos_count_actual = np.sum(A[i] == 1)
        neg_count_actual = np.sum(A[i] == -1)
        if pos_count_actual != pos_count or neg_count_actual != neg_count:
            raise ValueError(f"Row {i} has {pos_count_actual} +1s and {neg_count_actual} -1s (expected 32 each)")
    
    return A

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    
    if len(seed) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex chars)")

    # Define input specifications
    input_specs = [
        mb.TensorSpec(shape=(BATCH_SIZE, 256)),  # input
        mb.TensorSpec(shape=(BATCH_SIZE, 256)),  # noise
    ]

    # Generate matrices for all rounds
    matrices = [generate_ternary_matrix_from_seed(seed, round_num) for round_num in range(ROUNDS)]
    print("matrix[0]", matrices[0])
    biases = [-np.sum(matrix, axis=0) for matrix in matrices]
    print("bias[0]", biases[0])

    # Define the MIL program
    @mb.program(input_specs=input_specs)
    def matmul_scaled_bias_clamped_relu_prog(input, noise):
        x = input
        # Apply rounds
        for round_num in range(ROUNDS):
            x = mb.matmul(x=x, y=matrices[round_num])
            x = mb.mul(x=x, y=2.0)
            x = mb.add(x=x, y=biases[round_num])
            x = mb.add(x=x, y=noise)
            x = mb.clip(x=x, alpha=0.0, beta=1.0)
        return x

    # Convert to Core ML model with FP16 precision
    mlmodel = ct.convert(
        matmul_scaled_bias_clamped_relu_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="output")]
    )

    # Save as .mlpackage
    mlmodel.save("test_coreml.mlpackage")
    print("Model saved as .mlpackage")

if __name__ == "__main__":
    main()
