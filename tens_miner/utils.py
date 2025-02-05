"""Utility functions for TensHash mining."""
import struct
import torch
import numpy as np
import ctypes
from .constants import IN_SIZE

def count_leading_zero_bits(hash_bytes):
    """Count the number of leading zero bits in the hash (big-endian display)."""
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

def print_hex_le(hash_bytes):
    """Return a hex string for the bytes in reverse order (big-endian display)."""
    return "".join("{:02x}".format(b) for b in hash_bytes[::-1])

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
