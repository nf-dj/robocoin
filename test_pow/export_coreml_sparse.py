import argparse
import numpy as np
from Crypto.Cipher import ChaCha20
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# Network dimensions.
INPUT_SIZE = 256      # Input vector size.
HIDDEN_SIZE = 1024    # Hidden layer size.
NUM_HIDDEN_LAYERS = 64
BATCH_SIZE = 2048     # Batch size for the Core ML model.

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a CoreML model that uses ChaCha20-generated matrices without biases. "
                    "The model accepts a 256-bit input, expands to a higher dimension, applies several hidden "
                    "transformations, and compresses back to a 256-bit output."
    )
    parser.add_argument("seed", type=str, help="32-byte hex seed (64 hex characters)")
    return parser.parse_args()

def hex_to_bytes(hex_str):
    b = bytes.fromhex(hex_str)
    if len(b) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex characters)")
    return b

def generate_sparse_matrix(rows, cols, seed, round_num):
    """
    Generate a sparse matrix with exactly 127 non-zeros per row.
    For each non-zero element:
    - Uses 2 bytes: first byte (a) and second byte (b)
    - MSB of first byte determines value (-1 or 1)
    - Remaining 15 bits determine position mod cols
    """
    nonce = round_num.to_bytes(8, byteorder='big')
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    
    # Need 2 bytes per position, 127 positions per row
    bytes_per_row = 127 * 2
    random_bytes = cipher.encrypt(b'\x00' * (rows * bytes_per_row))
    
    matrix = np.zeros((rows, cols), dtype=np.float32)
    
    for row in range(rows):
        # Get bytes for this row
        row_start = row * bytes_per_row
        for i in range(127):
            # Get 2 bytes for this position
            byte1 = random_bytes[row_start + i*2]
            byte2 = random_bytes[row_start + i*2 + 1]
            
            # MSB of byte1 determines value
            val = 1 if (byte1 & 0x80) else -1
            
            # Remaining 15 bits determine position
            pos = ((byte1 & 0x7F) << 8 | byte2) % cols
            
            matrix[row, pos] = val
            
    return matrix

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    
    # Each layer is stored as a tuple: (name, constant_matrix)
    layers = []
    nonce_counter = 0  # Use a different nonce for each matrix.
    
    # Expansion layer: generate a matrix of shape (HIDDEN_SIZE, INPUT_SIZE)
    # then transpose it to effectively have a constant of shape (INPUT_SIZE, HIDDEN_SIZE).
    mat1 = generate_sparse_matrix(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter)
    nonce_counter += 1
    expansion_mat = np.transpose(mat1)  # Now shape is (INPUT_SIZE, HIDDEN_SIZE)
    layers.append(('expansion', expansion_mat))
    
    # Hidden layers: each is a (HIDDEN_SIZE, HIDDEN_SIZE) matrix.
    for i in range(NUM_HIDDEN_LAYERS):
        mat_hidden = generate_sparse_matrix(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter)
        nonce_counter += 1
        layers.append(('hidden', mat_hidden))
    
    # Compression layer: generate a matrix of shape (INPUT_SIZE, HIDDEN_SIZE)
    # then transpose it to get a constant of shape (HIDDEN_SIZE, INPUT_SIZE).
    mat_final = generate_sparse_matrix(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter)
    nonce_counter += 1
    compression_mat = np.transpose(mat_final)  # Now shape is (HIDDEN_SIZE, INPUT_SIZE)
    layers.append(('compression', compression_mat))
    
    # Debug: print shapes and sums for each layer.
    for idx, (name, mat) in enumerate(layers):
        print(f"Layer {idx} ({name}) - matrix shape: {mat.shape}, matrix sum: {np.sum(mat)}")
    
    # Define the MIL program with a single input tensor of shape (BATCH_SIZE, INPUT_SIZE).
    input_specs = [mb.TensorSpec(shape=(BATCH_SIZE, INPUT_SIZE))]
    
    @mb.program(input_specs=input_specs)
    def transform_prog(input):
        x = input
        # For each layer:
        #   1. Map the input from {0, 1} to {-1, +1} via (2*x - 1)
        #   2. Multiply by the constant matrix.
        #   3. Clip the result to [0, 1].
        for idx, (name, mat) in enumerate(layers):
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            dot = mb.matmul(x=x_mapped, y=mb.const(val=mat))
            x = mb.clip(x=dot, alpha=0.0, beta=1.0)
        return x
    
    # Convert the MIL program to a Core ML ML program (using FP16 precision).
    mlmodel = ct.convert(
        transform_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        outputs=[ct.TensorType(name="output")]
    )
    
    mlmodel.save("test_coreml.mlpackage")
    print("Model saved.")
    
    spec = mlmodel.get_spec()
    output_feature_names = [feature.name for feature in spec.description.output]
    print("Output feature names:", output_feature_names)

if __name__ == "__main__":
    main()
