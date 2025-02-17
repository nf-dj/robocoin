import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
import argparse
from Crypto.Cipher import ChaCha20
import sys

SIZE = 256
ROUNDS = 16
BATCH_SIZE = 128
MAX_ATTEMPTS = 1000
DOT_THRESHOLD = 5

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CoreML model with seed-based matrix generation (C-style ternary transform using ChaCha20)"
    )
    parser.add_argument('seed', type=str, help='32-byte hex seed for matrix generation')
    return parser.parse_args()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, round_num, debug=False):
    """
    Generate a 256x256 ternary matrix (dtype float32) using a ChaCha20-based RNG
    to mimic the C version's random matrix generation.

    In this version, the provided 32-byte seed is used directly as the ChaCha20 key,
    and the round number (converted into an 8-byte big-endian integer) is used as the nonce.
    
    For each row:
      - Generate 256 random 4-bit numbers by reading SIZE/2 bytes from the ChaCha20 stream.
      - Each byte yields two 4-bit nibbles (high and low); these 4-bit values are mapped as:
            0 -> +1
            1 -> -1
         otherwise -> 0.
      - The candidate row is accepted only if the absolute dot product with every previously
        accepted row is ≤ DOT_THRESHOLD.
    """
    # Convert the round number into an 8-byte nonce (big-endian)
    nonce = round_num.to_bytes(8, byteorder='big')
    
    # Create a ChaCha20 cipher instance using the key and nonce.
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    
    A = np.zeros((SIZE, SIZE), dtype=np.int8)
    for i in range(SIZE):
        valid = False
        attempts = 0
        while attempts < MAX_ATTEMPTS and not valid:
            attempts += 1
            # Request SIZE/2 bytes to get 256 nibbles (since each byte gives two 4-bit numbers)
            nbytes = SIZE // 2
            random_bytes = cipher.encrypt(b'\x00' * nbytes)
            #print("random_bytes", random_bytes.hex())
            # Convert bytes into an array of unsigned 8-bit integers.
            arr = np.frombuffer(random_bytes, dtype=np.uint8)
            # Extract high and low nibbles from each byte to form an array of 256 4-bit numbers.
            row_vals = np.empty(SIZE, dtype=np.uint8)
            row_vals[0::2] = arr >> 4         # high nibble
            row_vals[1::2] = arr & 0x0F         # low nibble
            
            # Map the nibble values:
            #   if nibble == 0 then +1; if nibble == 1 then -1; else 0.
            row = np.where(row_vals == 0, 1, np.where(row_vals == 1, -1, 0)).astype(np.int8)
            
            # Validate candidate row by checking dot products with all previously accepted rows.
            is_valid = True
            for j in range(i):
                dot = np.abs(np.dot(row, A[j]))
                if dot > DOT_THRESHOLD:
                    is_valid = False
                    if debug:
                        print(f"Round {round_num}, row {i} attempt {attempts} rejected: dot with row {j} = {dot}")
                    break
            if is_valid:
                A[i] = row
                valid = True
                if debug:
                    print(f"Round {round_num}, row {i} accepted after {attempts} attempts.")
        if not valid:
            raise ValueError(f"Failed to generate valid row {i} after {MAX_ATTEMPTS} attempts for round {round_num}")
    return A.astype(np.float32)

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    if len(seed) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex characters)")
    
    # Generate one matrix per round using the ChaCha20-based generation.
    matrices = [generate_ternary_matrix_from_seed(seed, round_num, debug=True) for round_num in range(ROUNDS)]
    
    # Debug prints: show the first 5 rows of round 0 and the final round.
    print("Matrix for round 0 (first 5 rows):")
    print(matrices[0][:5])
    print("Matrix for round -1 (first 5 rows):")
    print(matrices[-1][:5])
    
    # Additional debug: print a simple checksum (sum of elements) for each matrix.
    for round_num, matrix in enumerate(matrices):
        print(f"Round {round_num} matrix sum: {np.sum(matrix)}")
    
    # Define input specifications for the Core ML model.
    # "input" is the initial bit vector (values 0 or 1) and "noise" provides fallback bits.
    input_specs = [
        mb.TensorSpec(shape=(BATCH_SIZE, SIZE)),  # input bits
        mb.TensorSpec(shape=(BATCH_SIZE, SIZE)),  # noise bits
    ]
    
    @mb.program(input_specs=input_specs)
    def ternary_transform_prog(input, noise):
        x = input
        for round_num in range(ROUNDS):
            # Map input bits from {0, 1} to {-1, 1} by performing (2 * x - 1)
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            
            # Use the corresponding constant matrix for this round.
            matrix_const = mb.const(val=matrices[round_num])
            
            # Compute the dot product between x_mapped and the constant matrix.
            dot = mb.matmul(x=x_mapped, y=matrix_const)
            
            # Add noise to the dot product and clip the result between 0 and 1.
            temp_add = mb.add(x=dot, y=noise)
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
    
    # Save the Core ML model.
    mlmodel.save("test_coreml.mlpackage")
    print("Model saved as 'test_coreml.mlpackage'.")
    
    # Retrieve and print the output feature names from the model specification.
    spec = mlmodel.get_spec()
    output_feature_names = [feature.name for feature in spec.description.output]
    print("Output feature names:", output_feature_names)

if __name__ == "__main__":
    main()
