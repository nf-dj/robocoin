import argparse
import numpy as np
from Crypto.Cipher import ChaCha20
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# Network dimensions.
INPUT_SIZE = 256      # Input vector size.
HIDDEN_SIZE = 1024    # Hidden layer size.
NUM_HIDDEN_LAYERS = 64  # (Note: description mentions 16 rounds; adjust if needed)
BATCH_SIZE = 2048     # Batch size for the Core ML model.

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a CoreML model that uses ChaCha20-generated matrices. "
                    "The model accepts a 256-bit input, expands to 4096-dim, applies several hidden "
                    "transformations, and compresses back to a 256-bit output."
    )
    parser.add_argument("seed", type=str, help="32-byte hex seed (64 hex characters)")
    return parser.parse_args()

def hex_to_bytes(hex_str):
    b = bytes.fromhex(hex_str)
    if len(b) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex characters)")
    return b

def generate_matrix_chacha(rows, cols, key, nonce_int):
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
    mods = data % 3
    # Mapping: 0 -> -1, 1 -> 0, 2 -> 1.
    mapping = np.empty_like(mods, dtype=np.int8)
    mapping[mods == 0] = -1
    mapping[mods == 1] = 0
    mapping[mods == 2] = 1
    
    # Reshape the flat array into the desired matrix shape and convert to float32.
    mat = mapping.reshape((rows, cols)).astype(np.float32)
    return mat

def main():
    args = parse_args()
    seed = hex_to_bytes(args.seed)
    
    # Each layer will be stored as a tuple: (name, constant_matrix, bias)
    layers = []
    nonce_counter = 0  # Use a different nonce for each matrix.
    
    # Expansion layer: generate a matrix of shape (HIDDEN_SIZE, INPUT_SIZE)
    # then transpose it to effectively have a constant of shape (INPUT_SIZE, HIDDEN_SIZE).
    mat1 = generate_matrix_chacha(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter)
    nonce_counter += 1
    expansion_mat = np.transpose(mat1)  # Now shape is (INPUT_SIZE, HIDDEN_SIZE)
    # Compute bias as negative of the average weight per column.
    expansion_bias = -np.sum(expansion_mat, axis=0, keepdims=True) / expansion_mat.shape[0]
    layers.append(('expansion', expansion_mat, expansion_bias))
    
    # Hidden layers: each is a (HIDDEN_SIZE, HIDDEN_SIZE) matrix.
    for i in range(NUM_HIDDEN_LAYERS):
        mat_hidden = generate_matrix_chacha(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter)
        nonce_counter += 1
        hidden_bias = -np.sum(mat_hidden, axis=0, keepdims=True) / mat_hidden.shape[0]
        layers.append(('hidden', mat_hidden, hidden_bias))
    
    # Compression layer: generate a matrix of shape (INPUT_SIZE, HIDDEN_SIZE)
    # then transpose it to get a constant of shape (HIDDEN_SIZE, INPUT_SIZE).
    mat_final = generate_matrix_chacha(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter)
    nonce_counter += 1
    compression_mat = np.transpose(mat_final)  # Now shape is (HIDDEN_SIZE, INPUT_SIZE)
    compression_bias = -np.sum(compression_mat, axis=0, keepdims=True) / compression_mat.shape[0]
    layers.append(('compression', compression_mat, compression_bias))
    
    # Debug: print shapes and sums for each layer.
    for idx, (name, mat, bias) in enumerate(layers):
        print(f"Layer {idx} ({name}) - matrix shape: {mat.shape}, matrix sum: {np.sum(mat)}, "
              f"bias shape: {bias.shape}, bias sum: {np.sum(bias)}")
    
    # Define the MIL program with a single input tensor of shape (BATCH_SIZE, INPUT_SIZE).
    input_specs = [mb.TensorSpec(shape=(BATCH_SIZE, INPUT_SIZE))]
    
    @mb.program(input_specs=input_specs)
    def transform_prog(input):
        x = input
        # For each layer:
        #   1. Map the input from {0,1} to {-1, +1} via (2*x - 1)
        #   2. Multiply by the constant matrix.
        #   3. Add the precomputed bias.
        #   4. Clip the result to [0, 1].
        for idx, (name, mat, bias) in enumerate(layers):
            temp = mb.mul(x=x, y=2.0)
            x_mapped = mb.sub(x=temp, y=1.0)
            dot = mb.matmul(x=x_mapped, y=mb.const(val=mat))
            dot_bias = mb.add(x=dot, y=mb.const(val=bias))
            x = mb.clip(x=dot_bias, alpha=0.0, beta=1.0)
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
