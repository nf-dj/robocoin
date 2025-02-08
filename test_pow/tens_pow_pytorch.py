#!/usr/bin/env python3
import sys
import torch
import ctypes
import numpy as np
import time
import threading
from Crypto.Cipher import ChaCha20

# Constants
OPS_PER_HASH = 256 * 256 * 2 * 64  # Matrix multiply size * rounds
BATCH_SIZE = 8192  # Increased batch size

progress_data = {
    "attempts": 0,
    "best_bits": 0,
    "start_time": time.time(),
    "stop": False
}

def count_leading_zero_bits(hash_bytes):
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            for bit in range(7, -1, -1):
                if (byte >> bit) & 1:
                    return count
                count += 1
    return count

class HashVectorGenerator:
    def __init__(self, batch_size):
        self.libnoise = ctypes.CDLL("./libnoise.so")
        self.libnoise.compute_binary_and_noise_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        self.libnoise.compute_binary_and_noise_batch.restype = None

        self.batch_size = batch_size
        self.binary_buffer = np.zeros((batch_size, 256), dtype=np.float32)
        self.noise_buffer = np.zeros((batch_size, 256), dtype=np.float32)
        self.binary_buffer_torch = torch.zeros((batch_size, 256), dtype=torch.float32)
        self.noise_buffer_torch = torch.zeros((batch_size, 256), dtype=torch.float32)
        
        self.binary_ptr = self.binary_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.noise_ptr = self.noise_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def generate(self, input_batch, device):
        input_flat = input_batch.flatten()
        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        self.libnoise.compute_binary_and_noise_batch(
            input_ptr, 
            self.binary_ptr, 
            self.noise_ptr, 
            self.batch_size
        )
        
        self.binary_buffer_torch.copy_(torch.from_numpy(self.binary_buffer))
        self.noise_buffer_torch.copy_(torch.from_numpy(self.noise_buffer))
        
        return (
            self.binary_buffer_torch.to(device),
            self.noise_buffer_torch.to(device)
        )

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def generate_ternary_matrix_from_seed(seed, device):
    input_size, output_size = 256, 256
    A = torch.zeros((input_size, output_size), dtype=torch.float32, device=device)
    pos_count = neg_count = 32

    for i in range(input_size):
        nonce = i.to_bytes(8, 'big')
        cipher = ChaCha20.new(key=seed, nonce=nonce)
        
        rand_bytes = cipher.encrypt(b'\x00' * (output_size * 4))
        rand_ints = np.frombuffer(rand_bytes, dtype=np.int32)
        chosen_indices = np.argsort(rand_ints)[:64]
        
        rand_bytes_shuffle = cipher.encrypt(b'\x00' * (64 * 4))
        shuffle_ints = np.frombuffer(rand_bytes_shuffle, dtype=np.int32)
        shuffle_perm = np.argsort(shuffle_ints)
        sign_vector = torch.tensor([1] * pos_count + [-1] * neg_count, dtype=torch.float32, device=device)
        sign_vector = sign_vector[torch.tensor(shuffle_perm, dtype=torch.long, device=device)]
        
        A[i, torch.tensor(chosen_indices, dtype=torch.long, device=device)] = sign_vector
    return A

def apply_matrix_rounds(binary_vectors, ternary_matrix, bias_plus_noise):
    # Process all inputs in parallel through all rounds
    batch_size = binary_vectors.shape[0]
    outputs = binary_vectors
    
    for _ in range(64):
        results = torch.matmul(outputs, ternary_matrix)
        results = 2 * results + bias_plus_noise
        outputs = (results > 0).float()
    
    return outputs

def binary_vectors_to_bytes(vectors):
    bits = vectors.cpu().numpy()
    batch_size = bits.shape[0]
    results = []
    for i in range(batch_size):
        bits_str = ''.join(str(int(bit)) for bit in bits[i])
        byte_vals = [int(bits_str[i:i+8], 2) for i in range(0, 256, 8)]
        results.append(bytes(byte_vals))
    return results

def print_progress():
    while not progress_data["stop"]:
        now = time.time()
        total_time = now - progress_data["start_time"]
        attempts = progress_data["attempts"]
        best_bits = progress_data["best_bits"]
        hr = attempts / total_time if total_time > 0 else 0
        tops = (hr * OPS_PER_HASH) / 1e12
        status = ("  {:4.0f}s    {:7.0f} h/s    {:10.6f} TOPS    Total: {:12d}    Best Bits: {:3d}"
                  "\r").format(total_time, hr, tops, attempts, best_bits)
        print(status, end="", flush=True)
        time.sleep(1)

def find_pow(ternary_matrix, target_bytes, batch_size=BATCH_SIZE, device='mps'):
    target = int.from_bytes(target_bytes, 'big')
    vector_gen = HashVectorGenerator(batch_size)
    bias = -torch.sum(ternary_matrix, dim=0)
    
    progress_thread = threading.Thread(target=print_progress, daemon=True)
    progress_thread.start()
    
    try:
        while True:
            input_batch = torch.randint(0, 256, (batch_size, 32), dtype=torch.uint8)
            binary_vectors, noise_vectors = vector_gen.generate(input_batch.numpy(), device)
            
            # Add bias to noise vectors once for all inputs
            bias_plus_noise = bias.unsqueeze(0) + noise_vectors
            
            # Process entire batch through all rounds
            output_vectors = apply_matrix_rounds(binary_vectors, ternary_matrix, bias_plus_noise)
            
            # Convert outputs to bytes
            output_bytes_list = binary_vectors_to_bytes(output_vectors)
            
            # Update progress
            progress_data["attempts"] += batch_size
            
            # Check outputs
            for i, output_bytes in enumerate(output_bytes_list):
                zeros = count_leading_zero_bits(output_bytes)
                if zeros > progress_data["best_bits"]:
                    progress_data["best_bits"] = zeros
                    
                output_int = int.from_bytes(output_bytes, 'big')
                if output_int < target:
                    progress_data["stop"] = True
                    progress_thread.join()
                    print("\nSolution found!")
                    return input_batch[i].numpy().tobytes().hex()
                    
    except KeyboardInterrupt:
        progress_data["stop"] = True
        progress_thread.join()
        print("\nMining stopped by user")
        sys.exit(0)

def main():
    if len(sys.argv) != 3:
        print("Usage: tens_pow_pytorch.py <32-byte-hex-seed> <32-byte-hex-target>")
        sys.exit(1)
        
    seed = hex_to_bytes(sys.argv[1])
    target_bytes = hex_to_bytes(sys.argv[2])
    
    if len(seed) != 32 or len(target_bytes) != 32:
        print("Error: Both seed and target must be 32 bytes (64 hex chars)")
        sys.exit(1)

    device = torch.device('mps')
    print(f"Mining with PyTorch on device: {device}")
    print(f"Seed: {seed.hex()}")
    print(f"Target: {target_bytes.hex()}")
    print(f"Batch size: {BATCH_SIZE}")
    
    ternary_matrix = generate_ternary_matrix_from_seed(seed, device)
    solution = find_pow(ternary_matrix, target_bytes)
    print(f"Solution: {solution}")

if __name__ == "__main__":
    main()
