import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse
from Crypto.Cipher import ChaCha20

POS_COUNT=16
NEG_COUNT=16
ROUNDS=64
BATCH_SIZE = 8192


def parse_args():
    parser = argparse.ArgumentParser(description="Export CoreML model with seed-based matrix generation")
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num):
    """Generate a 256x256 ternary matrix using the same method as tens_hash_np.py but with round number in nonce"""
    input_size, output_size = 256, 256
    total_nonzero = POS_COUNT + NEG_COUNT
    
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
    base_signs = np.array([1] * POS_COUNT + [-1] * NEG_COUNT, dtype=np.float32)
    
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
        if pos_count_actual != POS_COUNT or neg_count_actual != NEG_COUNT:
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
    print("matrix[-1]", matrices[-1])

    biases = [-np.sum(matrix, axis=0) for matrix in matrices]
    print("bias[0]", biases[0])
    print("bias[-1]", biases[-1])

    @mb.program(input_specs=input_specs)
    def matmul_scaled_bias_mod2_prog(input, noise):
        x = input
        # Apply rounds
        for round_num in range(ROUNDS):
            x = mb.matmul(x=x, y=matrices[round_num])
            #x = mb.mul(x=x, y=2.0)
            #x = mb.add(x=x, y=biases[round_num])
            x = mb.add(x=x, y=noise)
            # Compute mod 2:
            #   Divide by 2
            x_div = mb.mul(x=x, y=0.5)
            #   Cast quotient to int32 (truncating to floor for nonnegative values)
            x_int = mb.cast(x=x_div, dtype="int32")
            #   Cast back to float (fp32)
            x_int_float = mb.cast(x=x_int, dtype="fp32")
            #   Multiply the integer quotient (as float) by 2
            x_mul = mb.mul(x=x_int_float, y=2.0)
            #   Subtract from the original x to get the remainder
            x = mb.sub(x=x, y=x_mul)
        return x

    # Convert to Core ML model with FP16 precision
    mlmodel = ct.convert(
        matmul_scaled_bias_mod2_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="output")]
    )

    # Save as .mlpackage
    mlmodel.save("test_coreml.mlpackage")
    print("Model saved as .mlpackage")

if __name__ == "__main__":
    main()
