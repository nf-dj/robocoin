#!/usr/bin/env python3
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
    
    # Match C version's uint8_t data type
    data = np.frombuffer(keystream, dtype=np.uint8).astype(np.float32)
    
    print("Using seed:", seed.hex())
    
    pos = 0
    # Store matrices in same memory layout as C but reshape for direct ONNX GEMM
    print("Expand matrix (first 8 values):")
    expand_data = data[pos: pos + (INPUT_SIZE * HIDDEN_SIZE)]
    for i in range(4):
        for j in range(2):
            print(str(int(expand_data[i * INPUT_SIZE + j])), end=" ")
    print()

    expand_matrix = expand_data.reshape(INPUT_SIZE, HIDDEN_SIZE).T
    pos += (INPUT_SIZE * HIDDEN_SIZE)
    
    print("\nMatrix shapes:")
    print(f"  expand_matrix: {expand_matrix.shape}")
    
    middle_matrices = []
    for r in range(num_rounds):
        middle_data = data[pos: pos + (HIDDEN_SIZE * HIDDEN_SIZE)]
        if r == 0:
            print("First middle matrix (first 8 values):")
            for i in range(4):
                for j in range(2):
                    print(str(int(middle_data[i * HIDDEN_SIZE + j])), end=" ")
            print()

        m = middle_data.reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        middle_matrices.append(m)
        pos += (HIDDEN_SIZE * HIDDEN_SIZE)
        if r == 0:
            print(f"  middle_matrices[0]: {m.shape}")
    
    print("Compress matrix (first 8 values):")
    reduce_data = data[pos: pos + (HIDDEN_SIZE * OUTPUT_SIZE)]
    for i in range(4):
        for j in range(2):
            print(str(int(reduce_data[i * HIDDEN_SIZE + j])), end=" ")
    print()
    
    reduce_matrix = reduce_data.reshape(OUTPUT_SIZE, HIDDEN_SIZE)  # Reshape to [32, 256]

    
    return expand_matrix, middle_matrices, reduce_matrix

def main(seed_hex, num_rounds):
    try:
        seed = parse_seed(seed_hex)
    except ValueError as e:
        sys.exit("Error: " + str(e))
    
    expand_matrix, middle_matrices, reduce_matrix = generate_matrices(seed, num_rounds)
    
    print("\nMatrix shapes:")
    print(f"  expand_matrix: {expand_matrix.shape}")
    print(f"  middle_matrices[0]: {middle_matrices[0].shape}")
    print(f"  reduce_matrix: {reduce_matrix.shape}")
    
    # Define graph inputs - using FP32 directly
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, INPUT_SIZE])
    error_tensor = helper.make_tensor_value_info("error", onnx.TensorProto.FLOAT, [1, INPUT_SIZE])
    
    initializers = []
    nodes = []

    # Add constant for modulo 256
    const_256 = helper.make_tensor("const_256", onnx.TensorProto.FLOAT, [], [256.0])
    initializers.append(const_256)
    
    # Add expansion matrix
    expand_weight_tensor = helper.make_tensor(
        "expand_weights", 
        onnx.TensorProto.FLOAT,
        expand_matrix.shape,
        expand_matrix.flatten().tolist()
    )
    initializers.append(expand_weight_tensor)
    
    # Add repeats constant for error tiling
    repeats = helper.make_tensor(
        "repeats",
        onnx.TensorProto.INT64,
        [2],  # Shape is [2] for repeating [1, 32] input
        [1, 8]  # Repeat first dim 1x, second dim 8x to get [1, 256]
    )
    initializers.append(repeats)
    
    # Tile error for use in 256-dim operations
    tile_error = helper.make_node(
        "Tile",
        ["error", "repeats"],
        ["error_256"]
    )
    nodes.append(tile_error)
    
    # Initial expansion using GEMM
    expand_gemm = helper.make_node(
        "Gemm",
        ["input", "expand_weights", "error_256"],
        ["expand_add"],
        alpha=1.0,
        beta=1.0,
        transB=1  # Explicitly transpose B (weights)
    )
    nodes.append(expand_gemm)
    
    # Apply modulo 256 using Mod operator
    expand_mod = helper.make_node("Mod", ["expand_add", "const_256"], ["expand_final"], fmod=1)
    nodes.append(expand_mod)
    prev_output = "expand_final"
    
    # Middle layers using GEMM
    for i in range(num_rounds):
        weight_name = f"weights_{i}"
        m = middle_matrices[i]
        weight_tensor = helper.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT,
            m.shape,
            m.flatten().tolist()
        )
        initializers.append(weight_tensor)
        
        # GEMM + Modulo sequence
        gemm = helper.make_node(
            "Gemm",
            [prev_output, weight_name, "error_256"],
            [f"gemm_{i}"],
            alpha=1.0,
            beta=1.0,
            transB=1  # Explicitly transpose B (weights)
        )
        
        # Apply modulo 256 
        mod = helper.make_node("Mod", [f"gemm_{i}", "const_256"], [f"hidden_{i}"], fmod=1)
        
        nodes.extend([gemm, mod])
        prev_output = f"hidden_{i}"
    
    # Final reduction
    reduce_weight_tensor = helper.make_tensor(
        "reduce_weights",
        onnx.TensorProto.FLOAT,
        reduce_matrix.shape,
        reduce_matrix.flatten().tolist()
    )
    initializers.append(reduce_weight_tensor)
    
    # Final GEMM with small_noise for reduction
    final_gemm = helper.make_node(
        "Gemm",
        [prev_output, "reduce_weights", "error"],
        ["final_gemm"],
        alpha=1.0,
        beta=1.0,
        transB=1  # Explicitly transpose B (weights)
    )
    
    # Apply final modulo 256
    final_mod = helper.make_node("Mod", ["final_gemm", "const_256"], ["output"], fmod=1)
    
    nodes.extend([final_gemm, final_mod])
    
    # Define output tensor
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, OUTPUT_SIZE])
    
    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name="TensHashFp32",
        inputs=[input_tensor, error_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    
    # Create model
    model = helper.make_model(
        graph, 
        producer_name="tens-hash",
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Print model structure
    print("\nModel Structure:")
    print("Inputs:")
    for input in model.graph.input:
        print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    
    print("\nNodes:")
    for i, node in enumerate(model.graph.node):
        print(f"  Node {i} - {node.op_type}:")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")
    
    print("\nOutputs:")
    for output in model.graph.output:
        print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
    
    onnx.save(model, "tens_hash_fp32.onnx")
    print("\nOptimized ONNX model saved as tens_hash_fp32.onnx")

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
