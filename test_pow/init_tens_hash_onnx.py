#!/usr/bin/env python3
import sys
import onnx
import onnx.helper as helper
import numpy as np
from Crypto.Cipher import ChaCha20

# Constants
NUM_ROUNDS = 64
INPUT_SIZE = 32
HIDDEN_SIZE = 256   # Internal matrix size
OUTPUT_SIZE = 32
SEED_SIZE = 32

# --- Helper functions ---

def parse_seed(seed_hex):
    """Parse a 64-character hex string into a 32-byte seed."""
    if len(seed_hex) != 64:
        raise ValueError("Seed must be exactly 64 hex characters (32 bytes)")
    return bytes.fromhex(seed_hex)

def crypto_stream_chacha20_xor(message, nonce, key):
    """
    Encrypts a zero-filled message using ChaCha20 to generate a keystream.
    
    PyCryptodome's ChaCha20 expects:
      - key: 32 bytes
      - nonce: 8 bytes (default for PyCryptodome's ChaCha20)
    
    This function returns the ciphertext, which is the keystream XORed with zeros,
    i.e. just the keystream.
    """
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(message)

def generate_matrices(seed):
    """
    Generate matrices using ChaCha20 from a 32-byte seed.
    
    Total size in bytes:
      - Expansion matrix: HIDDEN_SIZE * INPUT_SIZE
      - Middle matrices: NUM_ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE
      - Reduction matrix: HIDDEN_SIZE * OUTPUT_SIZE
    
    We generate a keystream by encrypting a zero‑filled message.
    The keystream is interpreted as int8 values then cast to float32.
    """
    total_size = (HIDDEN_SIZE * INPUT_SIZE) + (NUM_ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)
    nonce = b'\x00' * 8  # Fixed nonce (8 bytes, as used in the C program)
    zero_message = bytes(total_size)  # Zero-filled message
    keystream = crypto_stream_chacha20_xor(zero_message, nonce, seed)
    
    # Interpret the keystream as int8, then cast to float32.
    data = np.frombuffer(keystream, dtype=np.int8).astype(np.float32)
    
    pos = 0
    # The expansion matrix is used as [1, INPUT_SIZE] x [INPUT_SIZE, HIDDEN_SIZE] => [1, HIDDEN_SIZE].
    # Reshape the expansion matrix as [INPUT_SIZE, HIDDEN_SIZE].
    expand_matrix = data[pos: pos + (INPUT_SIZE * HIDDEN_SIZE)].reshape(INPUT_SIZE, HIDDEN_SIZE)
    pos += (INPUT_SIZE * HIDDEN_SIZE)
    
    middle_matrices = []
    for _ in range(NUM_ROUNDS):
        # Each middle matrix is HIDDEN_SIZE x HIDDEN_SIZE.
        m = data[pos: pos + (HIDDEN_SIZE * HIDDEN_SIZE)].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        middle_matrices.append(m)
        pos += (HIDDEN_SIZE * HIDDEN_SIZE)
    
    # The reduction matrix reduces from [1, HIDDEN_SIZE] to [1, OUTPUT_SIZE].
    # For that, we need a weight of shape [HIDDEN_SIZE, OUTPUT_SIZE].
    reduce_matrix = data[pos: pos + (HIDDEN_SIZE * OUTPUT_SIZE)].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    
    return expand_matrix, middle_matrices, reduce_matrix

# --- Main Model Generation ---
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: {} <seed_hex>".format(sys.argv[0]))
    
    seed_hex = sys.argv[1].strip()
    try:
        seed = parse_seed(seed_hex)
    except ValueError as e:
        sys.exit("Error: " + str(e))
    
    # Generate matrices from the seed using ChaCha20
    expand_matrix, middle_matrices, reduce_matrix = generate_matrices(seed)
    
    # Create ONNX model graph.
    # The graph has two inputs ("input" and "error"), each of shape [1, 32].  
    # All MatMul operations are in float32, with casts and modulo applied via BitwiseAnd.
    
    # Define inputs (int8, later cast to float32)
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [1, INPUT_SIZE])
    error_tensor = helper.make_tensor_value_info("error", onnx.TensorProto.INT8, [1, INPUT_SIZE])
    
    initializers = []
    nodes = []
    
    # --- Step 1: Cast inputs to float32 ---
    input_cast = helper.make_node("Cast", ["input"], ["input_f32"], to=onnx.TensorProto.FLOAT)
    error_cast = helper.make_node("Cast", ["error"], ["error_f32"], to=onnx.TensorProto.FLOAT)
    nodes.extend([input_cast, error_cast])
    
    # --- Step 2: Expand input from [1,32] to [1,256] ---
    # Use expansion weight of shape [INPUT_SIZE, HIDDEN_SIZE]
    expand_weight_tensor = helper.make_tensor("expand_weights", onnx.TensorProto.FLOAT, expand_matrix.shape, expand_matrix.flatten().tolist())
    initializers.append(expand_weight_tensor)
    
    expand_matmul = helper.make_node("MatMul", ["input_f32", "expand_weights"], ["expanded_input"])
    nodes.append(expand_matmul)
    
    # Expand "error" from [1,32] to [1,256] via Tile.
    tile_repeats = np.array([1, HIDDEN_SIZE // INPUT_SIZE], dtype=np.int64)  # e.g. [1, 8]
    tile_tensor = helper.make_tensor("tile_repeats", onnx.TensorProto.INT64, tile_repeats.shape, tile_repeats.flatten().tolist())
    initializers.append(tile_tensor)
    
    expand_error = helper.make_node("Tile", ["error_f32", "tile_repeats"], ["expanded_error"])
    nodes.append(expand_error)
    
    # Add error to expanded input
    expand_add = helper.make_node("Add", ["expanded_input", "expanded_error"], ["expanded_added"])
    nodes.append(expand_add)
    
    # Cast to int32 before modulo
    expand_cast_int = helper.make_node("Cast", ["expanded_added"], ["expanded_int32"], to=onnx.TensorProto.INT32)
    nodes.append(expand_cast_int)
    
    # Define modulo tensor (255 as int32)
    modulo_tensor = helper.make_tensor("modulo", onnx.TensorProto.INT32, [], [255])
    initializers.append(modulo_tensor)
    
    # Apply modulo via BitwiseAnd, then cast back to float32 for subsequent MatMul.
    expand_modulo = helper.make_node("BitwiseAnd", ["expanded_int32", "modulo"], ["expanded_modulo_int32"])
    nodes.append(expand_modulo)
    
    expand_modulo_cast = helper.make_node("Cast", ["expanded_modulo_int32"], ["expanded_modulo_f32"], to=onnx.TensorProto.FLOAT)
    nodes.append(expand_modulo_cast)
    
    prev_output = "expanded_modulo_f32"
    
    # --- Step 3: 64 Rounds of 256x256 MatMul with error addition and modulo ---
    for i in range(NUM_ROUNDS):
        weight_name = f"weights_{i}"
        # Each middle weight is a float32 matrix of shape [HIDDEN_SIZE, HIDDEN_SIZE]
        m = middle_matrices[i]
        weight_tensor = helper.make_tensor(weight_name, onnx.TensorProto.FLOAT, m.shape, m.flatten().tolist())
        initializers.append(weight_tensor)
        
        matmul_output = f"matmul_{i}"
        matmul_node = helper.make_node("MatMul", [prev_output, weight_name], [matmul_output])
        nodes.append(matmul_node)
        
        add_output = f"add_{i}"
        add_node = helper.make_node("Add", [matmul_output, "expanded_error"], [add_output])
        nodes.append(add_node)
        
        cast_round_int = f"cast_{i}"
        cast_node = helper.make_node("Cast", [add_output], [cast_round_int], to=onnx.TensorProto.INT32)
        nodes.append(cast_node)
        
        mod_round = f"modulo_{i}"
        mod_node = helper.make_node("BitwiseAnd", [cast_round_int, "modulo"], [mod_round])
        nodes.append(mod_node)
        
        cast_round_f32 = f"cast_{i}_f32"
        cast_back_node = helper.make_node("Cast", [mod_round], [cast_round_f32], to=onnx.TensorProto.FLOAT)
        nodes.append(cast_back_node)
        
        prev_output = cast_round_f32  # Use as input for next round
    
    # --- Step 4: Reduce output from [1,256] to [1,32] ---
    # Use reduction weight of shape [HIDDEN_SIZE, OUTPUT_SIZE]
    reduce_weight_tensor = helper.make_tensor("reduce_weights", onnx.TensorProto.FLOAT, reduce_matrix.shape, reduce_matrix.flatten().tolist())
    initializers.append(reduce_weight_tensor)
    
    reduce_matmul = helper.make_node("MatMul", [prev_output, "reduce_weights"], ["reduced_output"])
    nodes.append(reduce_matmul)
    
    # Add error to reduced output (error_f32 is [1,32])
    reduce_add = helper.make_node("Add", ["reduced_output", "error_f32"], ["reduced_added"])
    nodes.append(reduce_add)
    
    # Final modulo: cast to int32, apply BitwiseAnd, output final int32 result.
    reduce_cast_int = helper.make_node("Cast", ["reduced_added"], ["reduced_int32"], to=onnx.TensorProto.INT32)
    nodes.append(reduce_cast_int)
    
    final_modulo = helper.make_node("BitwiseAnd", ["reduced_int32", "modulo"], ["output"])
    nodes.append(final_modulo)
    
    # Define output tensor (final output is int32, shape [1, OUTPUT_SIZE])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.INT32, [1, OUTPUT_SIZE])
    
    # Create the ONNX graph and model (forcing opset 20 for compatibility)
    graph = helper.make_graph(
        nodes=nodes,
        name="TensHash",
        inputs=[input_tensor, error_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    
    model = helper.make_model(graph, producer_name="chacha20-tens-hash", opset_imports=[helper.make_opsetid("", 20)])
    onnx.save(model, "tens_hash.onnx")
    
    print("ONNX model saved as tens_hash.onnx")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit("Error: " + str(e))

