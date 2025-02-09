#!/usr/bin/env python3
import sys
import numpy as np
from Crypto.Cipher import ChaCha20
import hashlib

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)[::-1]

def bytes_to_binary_vector(b):
    bits = ''.join([format(x, '08b') for x in b])
    return np.array([int(bit) for bit in bits])

def binary_vector_to_bytes(vec):
    bits = ''.join(str(int(bit)) for bit in vec)
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

def generate_ternary_matrix_from_seed(seed):
    input_size, output_size = 256, 256
    pos_count = neg_count = 32
    total_nonzero = pos_count + neg_count
    
    # Generate all random values at once using ChaCha20
    nonce = b'\x00' * 8  # 8-byte zero nonce for PyCrypto ChaCha20
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    total_rand_vals = input_size * output_size
    rand_bytes = cipher.encrypt(b'\x00' * (total_rand_vals * 4))
    rand_vals = np.frombuffer(rand_bytes, dtype=np.uint32).reshape(input_size, output_size)
    
    # Initialize the matrix and column sums
    A = np.zeros((input_size, output_size), dtype=np.int8)
    col_sums = np.zeros(output_size, dtype=np.int32)
    
    # Pre-generate sign array
    base_signs = np.array([1] * pos_count + [-1] * neg_count)
    
    # Process each row
    for i in range(input_size):
        # Sort indices based on random values
        chosen_indices = np.argsort(rand_vals[i])[:total_nonzero]
        
        # Place signs at sorted positions and update column sums
        A[i, chosen_indices] = base_signs
        col_sums += A[i]
    
    # Verify the matrix
    for i in range(input_size):
        pos_count_actual = np.sum(A[i] == 1)
        neg_count_actual = np.sum(A[i] == -1)
        if pos_count_actual != pos_count or neg_count_actual != neg_count:
            raise ValueError(f"Row {i} has {pos_count_actual} +1s and {neg_count_actual} -1s (expected 32 each)")
    
    return A

def apply_matrix_and_threshold(binary_vector, noise_vector, matrix, bias):
    result = np.matmul(binary_vector, matrix)
    result = 2 * result + bias + noise_vector
    return np.where(result > 0, 1, 0)

def main():
    if len(sys.argv) != 3:
        print("Usage: tens_hash_np.py <32-byte-hex-seed> <32-byte-hex-input>")
        sys.exit(1)
        
    seed = hex_to_bytes(sys.argv[1])
    input_bytes = hex_to_bytes(sys.argv[2])
    
    if len(seed) != 32 or len(input_bytes) != 32:
        print("Error: Both seed and input must be 32 bytes (64 hex chars)")
        sys.exit(1)
        
    matrix = generate_ternary_matrix_from_seed(seed)
    print("weights:", matrix, file=sys.stderr)
    bias = -np.sum(matrix, axis=0) 
    print("bias:", bias, file=sys.stderr)

    input_hash_bytes = hashlib.sha256(input_bytes).digest()
    noise_bytes = hashlib.sha256(input_hash_bytes).digest()
    
    binary_vector = bytes_to_binary_vector(input_hash_bytes)
    print("input:", binary_vector, file=sys.stderr)
    noise_vector = bytes_to_binary_vector(noise_bytes)
    print("noise:", noise_vector, file=sys.stderr)
    
    output_vector = binary_vector
    for _ in range(1):
        output_vector = apply_matrix_and_threshold(output_vector, noise_vector, matrix, bias)

    print("output:", output_vector, file=sys.stderr)
    output_bytes = binary_vector_to_bytes(output_vector)

    print(output_bytes[::-1].hex())

if __name__ == "__main__":
    main()
