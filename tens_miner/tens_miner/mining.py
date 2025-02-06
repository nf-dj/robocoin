#!/usr/bin/env python3
import struct
import ctypes
import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20

# Constants
IN_SIZE = 32         # 32 bytes input
BITS = IN_SIZE * 8   # 256 bits
HIDDEN = 256         # state size (256 bits)
ROUNDS = 64
BATCH_SIZE = 1024    # adjustable
OPS_PER_HASH = 256 * 256 * 2 * 64  # 8,388,608 operations per hash

class NoiseGenerator:
    def __init__(self, batch_size):
        self.libnoise = ctypes.CDLL("./libnoise.so")
        self.libnoise.compute_noise_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        self.libnoise.compute_noise_batch.restype = None

        self.batch_size = batch_size
        self.noise_buffer = np.zeros((batch_size, HIDDEN), dtype=np.float32)
        self.noise_buffer_torch = torch.zeros((batch_size, HIDDEN), dtype=torch.float32)
        self.noise_ptr = self.noise_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def generate(self, nonces_np, device):
        nonces_flat = nonces_np.flatten()
        nonces_ptr = nonces_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.libnoise.compute_noise_batch(nonces_ptr, self.noise_ptr, self.batch_size)
        self.noise_buffer_torch.copy_(torch.from_numpy(self.noise_buffer))
        return self.noise_buffer_torch.to(device)

def nonces_from_counter(start, count):
    """Generate nonces from a starting counter."""
    nonces = []
    for counter in range(start, start + count):
        nonce = bytearray(IN_SIZE)
        nonce[0:8] = struct.pack("<Q", counter)
        nonces.append(bytes(nonce))
    return torch.tensor(list(nonces), dtype=torch.uint8)

class TensHashCore(nn.Module):
    def __init__(self, seed_hex, device, batch_size):
        super().__init__()
        self.device = torch.device(device)
        self.seed = bytes.fromhex(seed_hex)[::-1]  # Note: Reverse bytes for correct endianness
        
        self.noise_gen = NoiseGenerator(batch_size)
        
        # Initialize matrices from seed
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0' * 8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        # Pre-generate all matrices and transfer to device once
        self.matrices = torch.tensor(
            [[(random_bytes[r*HIDDEN*HIDDEN + i] % 3) - 1 
              for i in range(HIDDEN*HIDDEN)] 
             for r in range(ROUNDS)],
            dtype=torch.float32, device=self.device
        ).reshape(ROUNDS, HIDDEN, HIDDEN)
        
        # Pre-allocate state tensor
        self.state = torch.zeros((batch_size, BITS), dtype=torch.float32, device=device)
        
    def bytes_to_bits(self, inp):
        """Convert bytes to bits tensor efficiently."""
        inp_cpu = inp.cpu().numpy()
        bits_np = np.unpackbits(inp_cpu, axis=1)
        bits_np = bits_np.reshape(-1, IN_SIZE, 8)
        bits_np = np.flip(bits_np, axis=2)
        bits_np = bits_np.reshape(-1, BITS).astype(np.float32)
        self.state.copy_(torch.from_numpy(bits_np))
        return self.state
    
    def bits_to_bytes(self, bits):
        """Convert bits tensor back to bytes efficiently."""
        B = bits.shape[0]
        bits = bits.reshape(B, IN_SIZE, 8)
        bits = torch.flip(bits, dims=[2])
        bits_cpu = bits.cpu().numpy().astype(np.uint8)
        packed_np = np.packbits(bits_cpu, axis=2)
        packed_np = packed_np.reshape(B, IN_SIZE)
        return torch.from_numpy(packed_np).to(self.device)
    
    def forward_batch(self, nonce_batch):
        """Compute hashes for a batch of nonces."""
        # Convert input to bits and use pre-allocated state
        state = self.bytes_to_bits(nonce_batch.to(self.device))
        
        # Generate noise once for all rounds
        nonce_np = nonce_batch.cpu().numpy()
        noise_tensor = self.noise_gen.generate(nonce_np, self.device)
        
        # Process all rounds
        for r in range(ROUNDS):
            state = torch.matmul(state, self.matrices[r].t()) + noise_tensor
            state = torch.remainder(torch.floor(state), 2.0)
        
        # Convert back to bytes
        out = self.bits_to_bytes(state)
        return out