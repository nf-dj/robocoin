#!/usr/bin/env python3
import sys
import numpy as np
from Crypto.Cipher import ChaCha20
import hashlib

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def bytes_to_binary_vector(b):
    bits = ''.join([format(x, '08b') for x in b])
    return np.array([int(bit) for bit in bits])

def binary_vector_to_bytes(vec):
    bits = ''.join(str(int(bit)) for bit in vec)
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

def generate_ternary_matrix_from_seed(seed):
    input_size, output_size = 256, 256
    A = np.zeros((input_size, output_size), dtype=int)
    pos_count = neg_count = 32

    for i in range(input_size):
        nonce = i.to_bytes(8, 'big')
        cipher = ChaCha20.new(key=seed, nonce=nonce)
        
        rand_bytes = cipher.encrypt(b'\x00' * (output_size * 4))
        rand_ints = np.frombuffer(rand_bytes, dtype=np.uint32)
        chosen_indices = np.argsort(rand_ints)[:64]
        
        rand_bytes_shuffle = cipher.encrypt(b'\x00' * (64 * 4))
        shuffle_ints = np.frombuffer(rand_bytes_shuffle, dtype=np.uint32)
        shuffle_perm = np.argsort(shuffle_ints)
        sign_vector = np.array([1] * pos_count + [-1] * neg_count)
        sign_vector = sign_vector[shuffle_perm]
        
        A[i, chosen_indices] = sign_vector
    return A

def apply_matrix_and_threshold(binary_vector, noise_vector, ternary_matrix):
    binary_vectors = np.vstack([binary_vector, noise_vector])
    result = np.matmul(binary_vectors, ternary_matrix)
    bias = -np.sum(ternary_matrix, axis=0)
    result = 2 * result + bias + noise_vector
    return np.where(result[0] > 0, 1, 0)

def main():
    if len(sys.argv) != 3:
        print("Usage: tens_hash_np.py <32-byte-hex-seed> <32-byte-hex-input>")
        sys.exit(1)
        
    seed = hex_to_bytes(sys.argv[1])
    input_bytes = hex_to_bytes(sys.argv[2])
    
    if len(seed) != 32 or len(input_bytes) != 32:
        print("Error: Both seed and input must be 32 bytes (64 hex chars)")
        sys.exit(1)
        
    input_hash_bytes = hashlib.sha256(input_bytes).digest()
    noise_bytes = hashlib.sha256(input_hash_bytes).digest()
    
    binary_vector = bytes_to_binary_vector(input_hash_bytes)
    noise_vector = bytes_to_binary_vector(noise_bytes)
    ternary_matrix = generate_ternary_matrix_from_seed(seed)
    
    output_vector = binary_vector
    for _ in range(64):
        output_vector = apply_matrix_and_threshold(output_vector, noise_vector, ternary_matrix)
    output_bytes = binary_vector_to_bytes(output_vector)
    
    print(output_bytes.hex())

if __name__ == "__main__":
    main()
