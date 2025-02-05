#!/usr/bin/env python3
import argparse, time, struct, sys, threading, ctypes
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

# Global variables for progress reporting.
progress_data = {
    "attempts": 0,
    "best_zero_bits": 0,
    "start_time": time.time(),
    "stop": False
}

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
# Make sure libnoise.so is in the same directory.
libnoise = ctypes.CDLL("./libnoise.so")
# Set up the C function signature.
# Signature: void compute_noise_batch(const uint8_t *nonces, float *noise_out, int batch_size)
libnoise.compute_noise_batch.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
libnoise.compute_noise_batch.restype = None

def compute_noise_batch_c(nonces_np):
    """
    Call the C function to compute noise for a batch.
    nonces_np: a NumPy array of shape (batch_size, 32) of type np.uint8.
    Returns a NumPy array of shape (batch_size, 256) with dtype np.float32.
    """
    batch_size = nonces_np.shape[0]
    noise_out = np.empty((batch_size, 256), dtype=np.float32)
    nonces_flat = nonces_np.flatten()
    noise_flat = noise_out.flatten()
    nonces_ptr = nonces_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    noise_ptr = noise_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    libnoise.compute_noise_batch(nonces_ptr, noise_ptr, batch_size)
    return noise_flat.reshape((batch_size, 256))

def nonces_from_counter(start, count):
    """
    Generate a tensor of shape (count, IN_SIZE) of nonce bytes.
    Only the first 8 bytes are set from the counter (little-endian).
    Returns a torch.uint8 tensor (on CPU).
    """
    nonces = []
    for counter in range(start, start + count):
        nonce = bytearray(IN_SIZE)
        nonce[0:8] = struct.pack("<Q", counter)
        nonces.append(bytes(nonce))
    return torch.tensor(list(nonces), dtype=torch.uint8)

class TensHashMiner(nn.Module):
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
        self.matrices = torch.stack(mats, dim=0)  # shape: (ROUNDS, HIDDEN, HIDDEN)
        
    def bytes_to_bits(self, inp):
        """
        Convert a batch of inputs (tensor of shape (B, IN_SIZE), dtype=uint8)
        to a tensor of shape (B, BITS) of float32 bits.
        Uses numpy's unpackbits and flips each byte to get LSB-first order.
        """
        inp_cpu = inp.cpu().numpy()
        bits_np = np.unpackbits(inp_cpu, axis=1)
        bits_np = bits_np.reshape(-1, IN_SIZE, 8)
        bits_np = np.flip(bits_np, axis=2)
        bits_np = bits_np.reshape(-1, BITS).astype(np.float32)
        return torch.from_numpy(bits_np).to(self.device)
    
    def bits_to_bytes(self, bits):
        """
        Convert a batch of bit tensors (shape: (B, BITS)) back to bytes.
        """
        B = bits.shape[0]
        bits = bits.reshape(B, IN_SIZE, 8)
        bits = torch.flip(bits, dims=[2])
        bits_cpu = bits.cpu().numpy().astype(np.uint8)
        packed_np = np.packbits(bits_cpu, axis=2)
        packed_np = packed_np.reshape(B, IN_SIZE)
        return torch.from_numpy(packed_np).to(self.device)
    
    def forward_batch(self, nonce_batch):
        """
        Compute hashes for a batch of nonce inputs (tensor of shape (B, IN_SIZE), dtype=uint8).
        Returns a tuple: (out, noise_time, inference_time) where:
          - out is a tensor of shape (B, IN_SIZE) containing the hash bytes,
          - noise_time is the time spent generating noise vectors via the C function,
          - inference_time is the time spent on the 64 rounds of matrix multiplications.
          
        For each nonce, we compute a 256-element noise vector via C, and then in every round we add
        that same noise vector (broadcast automatically) as bias.
        """
        B = nonce_batch.shape[0]
        state = self.bytes_to_bits(nonce_batch.to(self.device))
        
        # --- Time noise generation via C ---
        noise_start = time.time()
        nonce_np = nonce_batch.cpu().numpy()  # shape: (B, 32)
        noise_np = compute_noise_batch_c(nonce_np)  # shape: (B, 256)
        noise_tensor = torch.from_numpy(noise_np).to(self.device)
        noise_time = time.time() - noise_start
        
        # --- Time inference (matrix multiplications) ---
        inference_start = time.time()
        for r in range(ROUNDS):
            state = torch.fmod(torch.matmul(state, self.matrices[r].t()) + noise_tensor, 2.0)
        inference_time = time.time() - inference_start
        
        out = self.bits_to_bytes(state)
        return out, noise_time, inference_time

def progress_printer():
    """Print a single status line (using carriage return) every second until signaled to stop."""
    while not progress_data["stop"]:
        now = time.time()
        total_time = now - progress_data["start_time"]
        attempts = progress_data["attempts"]
        best = progress_data["best_zero_bits"]
        hr = attempts / total_time if total_time > 0 else 0
        tops = (hr * OPS_PER_HASH) / 1e12  # TOPS in trillions of ops per second
        status = ("  {:4.0f}s    {:7.0f} h/s    {:10.6f} TOPS    Total: {:12d}    Best Bits: {:3d}"
                  "\r").format(total_time, hr, tops, attempts, best)
        print(status, end="", flush=True)
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="TensHash Miner (PyTorch with C noise generation)")
    parser.add_argument("seed", help="64 hex character seed")
    parser.add_argument("difficulty", type=int, help="Number of leading 0 bits required (1-256)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of nonces per batch")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda, mps, cpu)")
    args = parser.parse_args()
    
    if len(args.seed) != 64:
        sys.exit("Seed must be 64 hex characters")
    if args.difficulty < 1 or args.difficulty > 256:
        sys.exit("Difficulty must be between 1 and 256")
    
    device = args.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA not available; using MPS instead.")
            device = "mps"
        else:
            print("CUDA not available; using CPU instead.")
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; using CPU instead.")
        device = "cpu"
    
    print("Mining with PyTorch on device:", device)
    print("  Seed:", args.seed)
    print("  Difficulty: {} leading 0 bits".format(args.difficulty))
    
    miner = TensHashMiner(args.seed, device)
    miner.eval()
    
    # Start the progress printer thread.
    progress_thread = threading.Thread(target=progress_printer, daemon=True)
    progress_thread.start()
    
    attempts = 0
    solution_found = False
    solution_nonce = None
    solution_hash = None
    
    while not solution_found:
        nonce_batch = nonces_from_counter(attempts, args.batch_size)
        attempts += args.batch_size
        progress_data["attempts"] = attempts  # update global counter
        
        with torch.no_grad():
            out_batch, noise_t, infer_t = miner.forward_batch(nonce_batch)
        # (Optional) Print per-batch timing info on a new line:
        #print("\nBatch noise time: {:.6f}s, inference time: {:.6f}s".format(noise_t, infer_t))
        
        out_batch_cpu = out_batch.cpu().numpy()
        nonce_batch_cpu = nonce_batch.cpu().numpy()
        
        for i in range(args.batch_size):
            hash_bytes = bytes(out_batch_cpu[i].tolist())
            # For display, reverse the bytes (big-endian)
            hash_disp = bytes(out_batch_cpu[i].tolist())[::-1]
            zeros = count_leading_zero_bits(hash_disp)
            if zeros > progress_data["best_zero_bits"]:
                progress_data["best_zero_bits"] = zeros
            if zeros >= args.difficulty:
                solution_found = True
                nonce_bytes = bytes(nonce_batch_cpu[i].tolist())[::-1]
                solution_nonce = nonce_bytes
                solution_hash = hash_disp
                break
    
    progress_data["stop"] = True
    progress_thread.join()
    
    total_time = time.time() - progress_data["start_time"]
    print("\n\nSolution found!")
    print("Nonce:", print_hex_le(solution_nonce))
    print("Hash: ", print_hex_le(solution_hash))
    print("Stats:")
    print("  Time: {:.1f} seconds".format(total_time))
    print("  Total hashes: {}".format(attempts))
    print("  Avg hash rate: {:.1f} h/s".format(attempts / total_time))
    tops_overall = ((attempts / total_time) * OPS_PER_HASH) / 1e12
    print("  Avg TOPS: {:.6f}".format(tops_overall))

if __name__ == "__main__":
    main()

