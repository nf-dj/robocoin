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
NUM_NONZERO = 128

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a CoreML model that uses ChaCha20-generated matrices without biases. "
                    "The model accepts a 256-bit input (with shape (256, batch_size): each column is a sample), "
                    "expands to a higher dimension, applies several hidden transformations, and compresses "
                    "back to a 256-bit output."
    )
    parser.add_argument("seed", type=str, help="32-byte hex seed (64 hex characters)")
    return parser.parse_args()

def hex_to_bytes(hex_str):
    b = bytes.fromhex(hex_str)
    if len(b) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex characters)")
    return b

def generate_dense_matrix(rows, cols, key, nonce_int):
    """
    Generate a constant matrix of shape (rows, cols) whose entries are chosen from {-1, 0, 1}
    using a ChaCha20-based RNG. This simplified version uses one byte per trit.
    """
    # Create an 8-byte nonce from the given integer.
    nonce = nonce_int.to_bytes(8, byteorder='big')
    cipher = ChaCha20.new(key=key, nonce=nonce)
    
    needed = rows * cols
    # Generate exactly the number of bytes needed.
    random_bytes = cipher.encrypt(b'\x00' * needed)
    data = np.frombuffer(random_bytes, dtype=np.uint8)
    
    # Map each byte to a value in {-1, 0, 1} by taking modulo 3.
    mods = data % 4
    # Mapping: 0 -> 0, 1 -> 1, 2 -> 0, 3 -> -1.
    mapping = np.empty_like(mods, dtype=np.int8)
    mapping[mods == 0] = 0
    mapping[mods == 1] = 1
    mapping[mods == 2] = 0
    mapping[mods == 3] = -1
    
    # Reshape the flat array into the desired matrix shape and convert to float32.
    mat = mapping.reshape((rows, cols)).astype(np.float32)
    return mat

def generate_sparse_matrix(rows, cols, seed, round_num):
    """
    Generate a sparse matrix with at most NUM_NONZERO non-zeros per row.
    For each non-zero element:
      - Uses 2 bytes: first byte (a) and second byte (b)
      - MSB of first byte determines value (-1 or 1)
      - Remaining 15 bits determine position mod cols
    """
    nonce = round_num.to_bytes(8, byteorder='big')
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    
    # Need 2 bytes per row, NUM_NONZERO  positions per row.
    bytes_per_row = NUM_NONZERO * 2
    random_bytes = cipher.encrypt(b'\x00' * (rows * bytes_per_row))
    
    matrix = np.zeros((rows, cols), dtype=np.float32)
    
    for row in range(rows):
        row_start = row * bytes_per_row
        for i in range(NUM_NONZERO):
            byte1 = random_bytes[row_start + i*2]
            byte2 = random_bytes[row_start + i*2 + 1]
            val = 1 if (byte1 & 0x80) else -1
            pos = (((byte1 & 0x7F) << 8) | byte2) % cols
            matrix[row, pos] = val
            
    return matrix

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    
    # Each layer is stored as a tuple: (name, constant_matrix)
    # Now we generate matrices in the orientation that works directly for A.x,
    # assuming that each sample is a column vector.
    layers = []
    nonce_counter = 0
    
    # Expansion layer: from 256 to 1024.
    # Generate matrix with shape (HIDDEN_SIZE, INPUT_SIZE) = (1024, 256).
    expansion_mat = generate_dense_matrix(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter)
    nonce_counter += 1
    layers.append(('expansion', expansion_mat))
    
    # Hidden layers: each is (HIDDEN_SIZE, HIDDEN_SIZE).
    for i in range(NUM_HIDDEN_LAYERS):
        mat_hidden = generate_dense_matrix(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter)
        nonce_counter += 1
        layers.append(('hidden', mat_hidden))
    
    # Compression layer: from 1024 back to 256.
    # Generate matrix with shape (INPUT_SIZE, HIDDEN_SIZE) = (256, 1024).
    compression_mat = generate_dense_matrix(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter)
    nonce_counter += 1
    layers.append(('compression', compression_mat))
    
    # Debug print for each layer.
    for idx, (name, mat) in enumerate(layers):
        print(f"Layer {idx} ({name}) - matrix shape: {mat.shape}, matrix sum: {np.sum(mat)}")
    
    # Define the MIL program.
    # Note: The input tensor is now defined as (INPUT_SIZE, BATCH_SIZE),
    # so that each column is one 256-bit sample.
    input_specs = [mb.TensorSpec(shape=(INPUT_SIZE, BATCH_SIZE))]
    
    @mb.program(input_specs=input_specs)
    def transform_prog(input):
        x = input  # x has shape (INPUT_SIZE, BATCH_SIZE)
        # For each layer:
        #  1. Map input from {0, 1} to {-1, +1} via (2*x - 1).
        #  2. Multiply by the constant matrix A (using A · x).
        #  3. Clip the result to [0, 1].
        for idx, (name, mat) in enumerate(layers):
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            # A.x multiplication:
            #  - For expansion, A has shape (HIDDEN_SIZE, INPUT_SIZE) and x_mapped is (INPUT_SIZE, BATCH_SIZE),
            #    yielding (HIDDEN_SIZE, BATCH_SIZE).
            #  - For hidden layers, A is (HIDDEN_SIZE, HIDDEN_SIZE).
            #  - For compression, A is (INPUT_SIZE, HIDDEN_SIZE).
            dot = mb.matmul(x=mb.const(val=mat), y=x_mapped)
            x = mb.clip(x=dot, alpha=0.0, beta=1.0)
        return x  # Final output shape is (INPUT_SIZE, BATCH_SIZE).
    
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
