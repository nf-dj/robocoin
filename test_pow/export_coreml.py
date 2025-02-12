import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse

# Constants (matching the C version)
SIZE = 256
ROUNDS = 16          # Using 16 rounds (default in the C version)
BATCH_SIZE = 8192
MAX_ATTEMPTS = 1000
DOT_THRESHOLD = 5

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CoreML model with seed-based matrix generation (C-style ternary transform)"
    )
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num):
    """
    Generate a 256x256 ternary matrix (dtype float32) whose rows are generated like in the C code.
    
    For each row:
      - For each column, sample an integer in [0, 16).
         * If the sampled value is 0, set the entry to +1.
         * If the sampled value is 1, set the entry to -1.
         * Otherwise, set the entry to 0.
      - Accept the candidate row only if the absolute dot product with every previously
        accepted row is ≤ DOT_THRESHOLD.
    
    Deterministic seeding is achieved by combining the provided 32-byte seed with the round number.
    """
    seed_int = int.from_bytes(seed, 'big') ^ (round_num + 0xABCDEF)
    rng = np.random.default_rng(seed_int)
    A = np.zeros((SIZE, SIZE), dtype=np.int8)
    
    for i in range(SIZE):
        valid = False
        for attempt in range(MAX_ATTEMPTS):
            # Generate a candidate row.
            vals = rng.integers(low=0, high=16, size=SIZE)
            # Set entry: if value==0 then +1; if value==1 then -1; else 0.
            row = np.where(vals == 0, 1, np.where(vals == 1, -1, 0)).astype(np.int8)
            # Validate by checking dot product with all previously accepted rows.
            is_valid = True
            for j in range(i):
                dot = np.abs(np.dot(row, A[j]))
                if dot > DOT_THRESHOLD:
                    is_valid = False
                    break
            if is_valid:
                A[i] = row
                valid = True
                break
        if not valid:
            raise ValueError(f"Failed to generate valid row {i} after {MAX_ATTEMPTS} attempts for round {round_num}")
    return A.astype(np.float32)

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    if len(seed) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex chars)")
    
    # Generate one matrix per round using the C-style generation.
    matrices = [generate_ternary_matrix_from_seed(seed, round_num) for round_num in range(ROUNDS)]
    print("Matrix for round 0 (first 5 rows):\n", matrices[0][:5])
    print("Matrix for round -1 (first 5 rows):\n", matrices[-1][:5])
    
    # Define input specifications.
    # "input" is the initial bit vector (values 0 or 1) and "noise" provides fallback bits.
    input_specs = [
        mb.TensorSpec(shape=(BATCH_SIZE, SIZE)),  # input bits
        mb.TensorSpec(shape=(BATCH_SIZE, SIZE)),  # noise bits
    ]
    
    @mb.program(input_specs=input_specs)
    def ternary_transform_prog(input, noise):
        x = input
        for round_num in range(ROUNDS):
            # Map x from {0,1} to {-1,1}: (2 * x - 1)
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            
            # Use the corresponding matrix
            matrix_const = mb.const(val=matrices[round_num])
            
            # Compute the dot product: dot = x_mapped * matrix_const
            dot = mb.matmul(x=x_mapped, y=matrix_const)
            
            # Add noise to dot.
            temp_add = mb.add(x=dot, y=noise)
            # Clip the result between 0 and 1.
            round_out = mb.clip(x=temp_add, alpha=0.0, beta=1.0)
            x = round_out
        return x
    
    # Convert the MIL program to a Core ML model with FP16 precision.
    mlmodel = ct.convert(
        ternary_transform_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="output")]
    )
    
    # Save the model.
    mlmodel.save("test_coreml.mlpackage")
    print("Model saved as 'test_coreml.mlpackage'.")

if __name__ == "__main__":
    main()

