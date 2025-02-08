import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse
from Crypto.Cipher import ChaCha20

def parse_args():
    parser = argparse.ArgumentParser(description="Export CoreML model with seed-based matrix generation")
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num):
    """Generate a 256x256 ternary matrix for a specific round using ChaCha20"""
    input_size, output_size = 256, 256
    A = np.zeros((input_size, output_size), dtype=np.float32)
    pos_count = neg_count = 32

    for i in range(input_size):
        # Include round number in nonce to get different matrix per round
        nonce = round_num.to_bytes(4, 'big') + i.to_bytes(4, 'big')
        cipher = ChaCha20.new(key=seed, nonce=nonce)
        
        rand_bytes = cipher.encrypt(b'\x00' * (output_size * 4))
        rand_ints = np.frombuffer(rand_bytes, dtype=np.int32)
        chosen_indices = np.argsort(rand_ints)[:64]
        
        rand_bytes_shuffle = cipher.encrypt(b'\x00' * (64 * 4))
        shuffle_ints = np.frombuffer(rand_bytes_shuffle, dtype=np.int32)
        shuffle_perm = np.argsort(shuffle_ints)
        sign_vector = np.array([1] * pos_count + [-1] * neg_count, dtype=np.float32)
        sign_vector = sign_vector[shuffle_perm]
        
        A[i, chosen_indices] = sign_vector
    return A

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    
    if len(seed) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex chars)")

    batch_size = 8192  # Fixed batch size

    # Define input specifications
    input_specs = [
        mb.TensorSpec(shape=(batch_size, 256)),  # input
        mb.TensorSpec(shape=(batch_size, 256)),  # noise
    ]

    # Generate matrices for all rounds
    matrices = [generate_ternary_matrix_from_seed(seed, round_num) for round_num in range(64)]

    # Define the MIL program
    @mb.program(input_specs=input_specs)
    def matmul_scaled_bias_clamped_relu_prog(input, bias):
        x = input
        # Apply rounds
        for round_num in range(64):
            x = mb.matmul(x=x, y=matrices[round_num])
            x = mb.mul(x=x, y=2.0)
            x = mb.add(x=x, y=bias)
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
