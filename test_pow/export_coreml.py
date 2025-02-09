import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse
from Crypto.Cipher import ChaCha20
from collections import deque


SIZE=256
ROUNDS=64
BATCH_SIZE = 8192


def parse_args():
    parser = argparse.ArgumentParser(description="Export CoreML model with seed-based matrix generation")
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num):
    """
    Generate a 256x256 ternary matrix A (dtype float32) with entries in {-1, 0, 1}
    such that each row has exactly one +1 and one -1, and each column also gets exactly
    one +1 and one -1.

    The plus ones are assigned using a random permutation, and the minus ones are assigned
    using a random derangement (i.e. a permutation with no fixed points relative to the plus
    permutation). This ensures that in each row the -1 is placed in a column different from the +1.
    
    The randomness is seeded using the provided seed (a bytes object) and the round number.
    """
    # Combine the seed bytes and round number into an integer seed.
    seed_int = int.from_bytes(seed, 'big') ^ (round_num + 0xABCDEF)
    rng = np.random.default_rng(seed_int)
    
    # Generate a random permutation for plus ones.
    plus_perm = rng.permutation(SIZE)
    
    # Generate a random permutation for minus ones that is a derangement relative to plus_perm.
    # (i.e. for every row i, minus_perm[i] != plus_perm[i])
    while True:
        minus_perm = rng.permutation(SIZE)
        if np.all(minus_perm != plus_perm):
            break

    print("plus_perm",plus_perm)
    print("minus_perm",minus_perm)

    # Create the matrix with zeros.
    A = np.zeros((SIZE, SIZE), dtype=np.int8)
    
    # Place the +1's and -1's.
    for i in range(SIZE):
        A[i, plus_perm[i]] = 1
        A[i, minus_perm[i]] = -1

    return A.astype(np.float32)

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

    # Define the MIL program
    @mb.program(input_specs=input_specs)
    def matmul_scaled_bias_clamped_relu_prog(input, noise):
        x = input
        # Apply rounds
        for round_num in range(ROUNDS):
            x = mb.matmul(x=x, y=matrices[round_num])
            #x = mb.mul(x=x, y=2.0)
            x = mb.mul(x=x, y=0.0) # XXX
            #x = mb.add(x=x, y=biases[round_num])
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
