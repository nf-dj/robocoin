#!/usr/bin/env python3
import sys
import onnx
import onnx.helper as helper
import numpy as np
from Crypto.Cipher import ChaCha20

HIDDEN_SIZE = 256
INPUT_SIZE = HIDDEN_SIZE
OUTPUT_SIZE = HIDDEN_SIZE

def parse_seed(seed_hex):
    if len(seed_hex) != 64:
        raise ValueError("Seed must be exactly 64 hex characters (32 bytes)")
    return bytes.fromhex(seed_hex)

def crypto_stream_chacha20_xor(message, nonce, key):
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(message)

def generate_matrices(seed, num_rounds):
    total_size = num_rounds * HIDDEN_SIZE * HIDDEN_SIZE
    nonce = b'\x00' * 8
    zero_message = bytes(total_size)
    keystream = crypto_stream_chacha20_xor(zero_message, nonce, seed)
    data = np.frombuffer(keystream, dtype=np.uint8)
    
    middle_matrices = []
    pos = 0
    
    for r in range(num_rounds):
        middle_data = data[pos:pos + (HIDDEN_SIZE * HIDDEN_SIZE)]
        ternary = middle_data % 3 - 1
        m = ternary.reshape(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float16)
        middle_matrices.append(m)
        pos += (HIDDEN_SIZE * HIDDEN_SIZE)
    
    return middle_matrices

def main(seed_hex, num_rounds):
    seed = parse_seed(seed_hex)
    middle_matrices = generate_matrices(seed, num_rounds)
    
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT16, [1, HIDDEN_SIZE])
    error_tensor = helper.make_tensor_value_info("error", onnx.TensorProto.FLOAT16, [1, HIDDEN_SIZE])
    
    initializers = []
    nodes = []

    const_2 = helper.make_tensor("const_2", onnx.TensorProto.FLOAT16, [], [2.0])
    initializers.append(const_2)
    
    prev_output = "input"
    
    for i in range(num_rounds):
        weight_name = f"weights_{i}"
        m = middle_matrices[i]
        weight_tensor = helper.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT16,
            m.shape,
            m.flatten().tolist()
        )
        initializers.append(weight_tensor)
        
        gemm = helper.make_node(
            "Gemm",
            [prev_output, weight_name, "error"],
            [f"gemm_{i}"],
            alpha=1.0,
            beta=1.0,
            transB=1
        )
        
        output_name = "output" if i == num_rounds-1 else f"hidden_{i}"
        mod = helper.make_node("Mod", [f"gemm_{i}", "const_2"], [output_name], fmod=1)
        
        nodes.extend([gemm, mod])
        prev_output = output_name if i < num_rounds-1 else None
    
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [1, HIDDEN_SIZE])
    
    graph = helper.make_graph(
        nodes=nodes,
        name="TernaryHashFp16",
        inputs=[input_tensor, error_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    
    model = helper.make_model(
        graph, 
        producer_name="ternary-hash",
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    onnx.save(model, "tens_hash_fp16.onnx")

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

    main(seed_hex, num_rounds)
