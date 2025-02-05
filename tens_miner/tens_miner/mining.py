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

# --- Load the C shared library for noise generation ---
libnoise = ctypes.CDLL("./libnoise.so")
libnoise.compute_noise_batch.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
libnoise.compute_noise_batch.restype = None

def compute_noise_batch_c(nonces_np):
    """Call the C function to compute noise for a batch."""
    batch_size = nonces_np.shape[0]
    noise_out = np.empty((batch_size, 256), dtype=np.float32)
    nonces_flat = nonces_np.flatten()
    noise_flat = noise_out.flatten()
    nonces_ptr = nonces_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    noise_ptr = noise_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    libnoise.compute_noise_batch(nonces_ptr, noise_ptr, batch_size)
    return noise_flat.reshape((batch_size, 256))

def nonces_from_counter(start, count):
    """Generate nonces from a starting counter."""
    nonces = []
    for counter in range(start, start + count):
        nonce = bytearray(IN_SIZE)
        nonce[0:8] = struct.pack("<Q", counter)
        nonces.append(bytes(nonce))
    return torch.tensor(list(nonces), dtype=torch.uint8)

class TensCoinMiner(nn.Module):  # Updated class name
    def __init__(self, seed_hex, device):
        super().__init__()
        self.device = torch.device(device)
        self.seed = bytes.fromhex(seed_hex)
        
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0' * 8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        mats = []
        for r in range(ROUNDS):
            start_idx = r * HIDDEN * HIDDEN
            mat = torch.tensor(
                [(random_bytes[start_idx + i] % 3) - 1 for i in range(HIDDEN * HIDDEN)],
                dtype=torch.float32, device=self.device
            ).reshape(HIDDEN, HIDDEN)
            mats.append(mat)
        self.matrices = torch.stack(mats, dim=0)
    
    def bytes_to_bits(self, inp):
        """Convert bytes to bits tensor."""
        inp_cpu = inp.cpu().numpy()
        bits_np = np.unpackbits(inp_cpu, axis=1)
        bits_np = bits_np.reshape(-1, IN_SIZE, 8)
        bits_np = np.flip(bits_np, axis=2)
        bits_np = bits_np.reshape(-1, BITS).astype(np.float32)
        return torch.from_numpy(bits_np).to(self.device)
    
    def bits_to_bytes(self, bits):
        """Convert bits tensor back to bytes."""
        B = bits.shape[0]
        bits = bits.reshape(B, IN_SIZE, 8)
        bits = torch.flip(bits, dims=[2])
        bits_cpu = bits.cpu().numpy().astype(np.uint8)
        packed_np = np.packbits(bits_cpu, axis=2)
        packed_np = packed_np.reshape(B, IN_SIZE)
        return torch.from_numpy(packed_np).to(self.device)
    
    def forward_batch(self, nonce_batch):
        """Compute hashes for a batch of nonces."""
        B = nonce_batch.shape[0]
        state = self.bytes_to_bits(nonce_batch.to(self.device))
        
        # Generate noise via C
        nonce_np = nonce_batch.cpu().numpy()
        noise_np = compute_noise_batch_c(nonce_np)
        noise_tensor = torch.from_numpy(noise_np).to(self.device)
        
        # Process rounds
        for r in range(ROUNDS):
            state = torch.fmod(torch.matmul(state, self.matrices[r].t()) + noise_tensor, 2.0)
        
        return self.bits_to_bytes(state)
