#!/usr/bin/env python3
"""
This script creates an ONNX model that uses only uint8 matrix multiplications
(using MatMulInteger). The model uses uint8 inputs and weights, produces int32
results from MatMulInteger, then applies an explicit modulo 256 reduction and
casts the result back to uint8. This ensures that arithmetic is done in a wrap‑around
(mod 256) fashion and that the weight values match the FP32 model.
"""

import sys
import onnx
import onnx.helper as helper
import numpy as np
from Crypto.Cipher import ChaCha20

# Constants
INPUT_SIZE = 32
HIDDEN_SIZE = 256
OUTPUT_SIZE = 32
SEED_SIZE = 32

def parse_seed(seed_hex):
    if len(seed_hex) != 64:
        raise ValueError("Seed must be exactly 64 hex characters (32 bytes)")
    return bytes.fromhex(seed_hex)

def crypto_stream_chacha20_xor(message, nonce, key):
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(message)

def generate_matrices(seed, num_rounds):
    total_size = (HIDDEN_SIZE * INPUT_SIZE) + (num_rounds * HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)
    nonce = b'\x00' * 8
    zero_message = bytes(total_size)
    keystream = crypto_stream_chacha20_xor(zero_message, nonce, seed)
    
    # Use uint8 directly so that values ≥128 remain as such.
    data = np.frombuffer(keystream, dtype=np.uint8)
    
    print("Using seed:", seed.hex())
    pos = 0

    # --- Expand matrix ---
    # In the FP32 version, the expand matrix is computed as:
    #    expand_matrix = data[pos:pos+(INPUT_SIZE*HIDDEN_SIZE)].reshape(INPUT_SIZE, HIDDEN_SIZE).T
    # and printed as:
    #    first 8 values: 118 184 218 65 159 7 41 183
    #
    # To get the same printed values, we do the same here:
    print("Expand matrix (first 8 values):")
    expand_data = data[pos: pos + (INPUT_SIZE * HIDDEN_SIZE)]
    for i in range(4):
        for j in range(2):
            print(str(int(expand_data[i * INPUT_SIZE + j])), end=" ")
    print()
    # This yields a matrix with shape (INPUT_SIZE, HIDDEN_SIZE) when reshaped,
    # but the FP32 version transposes it for printing.
    expand_matrix_full = expand_data.reshape(INPUT_SIZE, HIDDEN_SIZE).T  # shape (HIDDEN_SIZE, INPUT_SIZE)
    # In the FP32 model, GEMM is done with transB=1 so the effective weight is expand_matrix_full.T.
    # Here, we want to feed the effective weight directly.
    expand_weight = expand_matrix_full.T   # shape (INPUT_SIZE, HIDDEN_SIZE)
    pos += (INPUT_SIZE * HIDDEN_SIZE)
    
    print("\nMatrix shapes:")
    print(f"  expand_weight: {expand_weight.shape}")  # expected (32,256)
    
    # --- Middle matrices ---
    middle_matrices = []
    for r in range(num_rounds):
        middle_data = data[pos: pos + (HIDDEN_SIZE * HIDDEN_SIZE)]
        if r == 0:
            print("First middle matrix (first 8 values):")
            for i in range(4):
                for j in range(2):
                    print(str(int(middle_data[i * HIDDEN_SIZE + j])), end=" ")
            print()
        # In the FP32 version, m = middle_data.reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        # and GEMM uses transB=1 (effective weight = m.T).
        # Here, we supply the effective weight directly:
        m = middle_data.reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        effective_middle = m.T  # shape (HIDDEN_SIZE, HIDDEN_SIZE)
        middle_matrices.append(effective_middle)
        pos += (HIDDEN_SIZE * HIDDEN_SIZE)
        if r == 0:
            print(f"  middle_matrices[0]: {effective_middle.shape}")
    
    # --- Reduce matrix ---
    # In the FP32 version, reduce_matrix = reduce_data.reshape(OUTPUT_SIZE, HIDDEN_SIZE)
    # and then used with transB=1 so effective weight = reduce_matrix.T (shape (HIDDEN_SIZE, OUTPUT_SIZE)).
    print("Compress matrix (first 8 values):")
    reduce_data = data[pos: pos + (HIDDEN_SIZE * OUTPUT_SIZE)]
    for i in range(4):
        for j in range(2):
            print(str(int(reduce_data[i * HIDDEN_SIZE + j])), end=" ")
    print()
    rmat = reduce_data.reshape(OUTPUT_SIZE, HIDDEN_SIZE)
    reduce_weight = rmat.T  # shape (HIDDEN_SIZE, OUTPUT_SIZE)
    
    return expand_weight, middle_matrices, reduce_weight

def main(seed_hex, num_rounds):
    try:
        seed = parse_seed(seed_hex)
    except ValueError as e:
        sys.exit("Error: " + str(e))
    
    expand_weight, middle_matrices, reduce_weight = generate_matrices(seed, num_rounds)
    
    print("\nMatrix shapes:")
    print(f"  expand_weight: {expand_weight.shape}")      # Expected: (32,256)
    print(f"  middle_matrices[0]: {middle_matrices[0].shape}")  # Expected: (256,256)
    print(f"  reduce_weight: {reduce_weight.shape}")        # Expected: (256,32)
    
    # --- Define graph inputs (all uint8) ---
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, INPUT_SIZE])
    error_tensor = helper.make_tensor_value_info("error", onnx.TensorProto.UINT8, [1, INPUT_SIZE])
    
    initializers = []
    nodes = []
    
    # Constant for modulo operation (256 as int32)
    const_256 = helper.make_tensor("const_256", onnx.TensorProto.INT32, [], [256])
    initializers.append(const_256)
    
    # Zero point constant for MatMulInteger (scalar uint8 0)
    zp = helper.make_tensor("zp", onnx.TensorProto.UINT8, [], [0])
    initializers.append(zp)
    
    # --- Tiling error ---
    # For the expansion and middle layers, we need the "error" input repeated to match HIDDEN_SIZE.
    repeats = helper.make_tensor("repeats", onnx.TensorProto.INT64, [2], [1, 8])  # Tile [1,32] -> [1,256]
    initializers.append(repeats)
    tile_error = helper.make_node("Tile", ["error", "repeats"], ["error_256"])
    nodes.append(tile_error)
    
    # --- Expansion Layer ---
    # MatMulInteger: [1, INPUT_SIZE] x [INPUT_SIZE, HIDDEN_SIZE] -> [1, HIDDEN_SIZE]
    # Here, expand_weight has shape (32,256), which yields the desired result.
    expand_weight_tensor = helper.make_tensor("expand_weights", onnx.TensorProto.UINT8, expand_weight.shape, expand_weight.flatten().tolist())
    initializers.append(expand_weight_tensor)
    matmul_exp = helper.make_node("MatMulInteger", ["input", "expand_weights", "zp", "zp"], ["expand_mm"], name="matmul_exp")
    nodes.append(matmul_exp)
    
    # Add bias: error_256 (uint8) is cast to int32 before addition.
    cast_error_exp = helper.make_node("Cast", ["error_256"], ["error_256_int32"], to=onnx.TensorProto.INT32, name="cast_error_exp")
    nodes.append(cast_error_exp)
    add_exp = helper.make_node("Add", ["expand_mm", "error_256_int32"], ["expand_add_int32"], name="add_exp")
    nodes.append(add_exp)
    
    # Apply modulo 256 on the int32 result.
    mod_exp = helper.make_node("Mod", ["expand_add_int32", "const_256"], ["expand_mod_int32"], name="mod_exp")
    nodes.append(mod_exp)
    
    # Cast result back to uint8.
    cast_exp = helper.make_node("Cast", ["expand_mod_int32"], ["expand_final"], to=onnx.TensorProto.UINT8, name="cast_exp")
    nodes.append(cast_exp)
    
    prev_output = "expand_final"
    
    # --- Middle Layers ---
    for i in range(num_rounds):
        weight_name = f"weights_{i}"
        m = middle_matrices[i]  # effective weight already computed (shape (256,256))
        weight_tensor = helper.make_tensor(weight_name, onnx.TensorProto.UINT8, m.shape, m.flatten().tolist())
        initializers.append(weight_tensor)
        
        matmul_mid = helper.make_node("MatMulInteger", [prev_output, weight_name, "zp", "zp"], [f"gemm_{i}_mm"], name=f"matmul_mid_{i}")
        nodes.append(matmul_mid)
        
        cast_error_mid = helper.make_node("Cast", ["error_256"], [f"error_256_int32_{i}"], to=onnx.TensorProto.INT32, name=f"cast_error_mid_{i}")
        nodes.append(cast_error_mid)
        add_mid = helper.make_node("Add", [f"gemm_{i}_mm", f"error_256_int32_{i}"], [f"gemm_{i}_add_int32"], name=f"add_mid_{i}")
        nodes.append(add_mid)
        mod_mid = helper.make_node("Mod", [f"gemm_{i}_add_int32", "const_256"], [f"hidden_{i}_int32"], name=f"mod_mid_{i}")
        nodes.append(mod_mid)
        cast_mid = helper.make_node("Cast", [f"hidden_{i}_int32"], [f"hidden_{i}"], to=onnx.TensorProto.UINT8, name=f"cast_mid_{i}")
        nodes.append(cast_mid)
        
        prev_output = f"hidden_{i}"
    
    # --- Final Reduction Layer ---
    # reduce_weight has shape (256,32) so that:
    # [1, HIDDEN_SIZE] x [HIDDEN_SIZE, OUTPUT_SIZE] = [1, OUTPUT_SIZE].
    reduce_weight_tensor = helper.make_tensor("reduce_weights", onnx.TensorProto.UINT8, reduce_weight.shape, reduce_weight.flatten().tolist())
    initializers.append(reduce_weight_tensor)
    matmul_fin = helper.make_node("MatMulInteger", [prev_output, "reduce_weights", "zp", "zp"], ["final_mm"], name="matmul_fin")
    nodes.append(matmul_fin)
    
    cast_error_fin = helper.make_node("Cast", ["error"], ["error_int32"], to=onnx.TensorProto.INT32, name="cast_error_fin")
    nodes.append(cast_error_fin)
    add_fin = helper.make_node("Add", ["final_mm", "error_int32"], ["final_add_int32"], name="add_fin")
    nodes.append(add_fin)
    mod_fin = helper.make_node("Mod", ["final_add_int32", "const_256"], ["final_mod_int32"], name="mod_fin")
    nodes.append(mod_fin)
    cast_fin = helper.make_node("Cast", ["final_mod_int32"], ["output"], to=onnx.TensorProto.UINT8, name="cast_fin")
    nodes.append(cast_fin)
    
    # Define the output tensor: uint8 [1, OUTPUT_SIZE]
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, [1, OUTPUT_SIZE])
    
    graph = helper.make_graph(
        nodes=nodes,
        name="TensHashUint8_Mod",
        inputs=[input_tensor, error_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    
    model = helper.make_model(graph, producer_name="tens-hash-uint8-mod", opset_imports=[helper.make_opsetid("", 13)])
    
    print("\nModel Structure:")
    print("Inputs:")
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")
    print("Nodes:")
    for i, node in enumerate(model.graph.node):
        print(f"  Node {i} - {node.op_type}:")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")
    print("Outputs:")
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")
    
    onnx.save(model, "tens_hash_int8.onnx")
    print("\nOptimized UINT8 ONNX model saved as tens_hash_int8.onnx")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: {} <seed_hex> <num_rounds>".format(sys.argv[0]))
    seed_hex = sys.argv[1].strip()
    try:
        num_rounds = int(sys.argv[2])
        if num_rounds <= 0:
            raise ValueError("Number of rounds must be positive")
    except ValueError:
        sys.exit("Error: num_rounds must be a positive integer")
    try:
        main(seed_hex, num_rounds)
    except Exception as e:
        sys.exit("Error: " + str(e))

