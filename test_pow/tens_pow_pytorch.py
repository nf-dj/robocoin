#!/usr/bin/env python3
import argparse, time, struct, sys, threading
from hashlib import sha256
import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20

# Constants
IN_SIZE = 32         
BITS = IN_SIZE * 8   
HIDDEN = 256         
ROUNDS = 64
BATCH_SIZE = 1024    
OPS_PER_HASH = 256 * 256 * 2 * 64  

progress_data = {
    "attempts": 0,
    "best_zero_bits": 0,
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

def print_hex(hash_bytes):
    return "".join("{:02x}".format(b) for b in hash_bytes)

def generate_noise_batch(nonces):
    B = nonces.shape[0]
    noise = np.zeros((B, ROUNDS * HIDDEN), dtype=np.float32)
    nonces_bytes = [bytes(n.tolist()) for n in nonces]
    
    for b, nonce in enumerate(nonces_bytes):
        input_hash = sha256(nonce).digest()
        for i in range(ROUNDS * HIDDEN):
            byte_idx = i % 32
            bit_idx = i % 8
            noise[b, i] = float((input_hash[byte_idx] >> bit_idx) & 1)
    return noise.reshape(B, ROUNDS, HIDDEN)

def nonces_from_counter(start, count):
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
        self.seed = bytes.fromhex(seed_hex)[::-1]
        
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0' * 8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        self.matrices = torch.tensor(
            [[(random_bytes[r*HIDDEN*HIDDEN + i] % 3) - 1 
              for i in range(HIDDEN*HIDDEN)] 
             for r in range(ROUNDS)],
            dtype=torch.float32, device=self.device
        ).reshape(ROUNDS, HIDDEN, HIDDEN)
        
    def bytes_to_bits(self, inp):
        inp_cpu = inp.cpu().numpy()
        bits_np = np.unpackbits(inp_cpu, axis=1)
        bits_np = bits_np.reshape(-1, IN_SIZE, 8)
        bits_np = np.flip(bits_np, axis=2)
        bits_np = bits_np.reshape(-1, BITS).astype(np.float32)
        return torch.from_numpy(bits_np).to(self.device)
    
    def bits_to_bytes(self, bits):
        B = bits.shape[0]
        bits = bits.reshape(B, IN_SIZE, 8)
        bits = torch.flip(bits, dims=[2])
        bits_cpu = bits.cpu().numpy().astype(np.uint8)
        packed_np = np.packbits(bits_cpu, axis=2)
        packed_np = packed_np.reshape(B, IN_SIZE)
        return torch.from_numpy(packed_np).to(self.device)
    
    def forward_batch(self, nonce_batch):
        B = nonce_batch.shape[0]
        state = self.bytes_to_bits(nonce_batch.to(self.device))
        
        # Noise generation
        noise_start = time.time()
        nonce_np = nonce_batch.cpu().numpy()
        noise_np = generate_noise_batch(nonce_np)  # shape: (B, ROUNDS, HIDDEN)
        noise_tensor = torch.from_numpy(noise_np).to(self.device)
        noise_time = time.time() - noise_start
        
        # Matrix multiplications
        inference_start = time.time()
        for r in range(ROUNDS):
            state = torch.matmul(state, self.matrices[r].t()) + noise_tensor[:, r]
            state = torch.remainder(torch.floor(state), 2.0)
        inference_time = time.time() - inference_start
        
        out = self.bits_to_bytes(state)
        return out, noise_time, inference_time

def progress_printer():
    while not progress_data["stop"]:
        now = time.time()
        total_time = now - progress_data["start_time"]
        attempts = progress_data["attempts"]
        best = progress_data["best_zero_bits"]
        hr = attempts / total_time if total_time > 0 else 0
        tops = (hr * OPS_PER_HASH) / 1e12
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
    
    progress_thread = threading.Thread(target=progress_printer, daemon=True)
    progress_thread.start()
    
    attempts = 0
    solution_found = False
    solution_nonce = None
    solution_hash = None
    
    while not solution_found:
        nonce_batch = nonces_from_counter(attempts, args.batch_size)
        attempts += args.batch_size
        progress_data["attempts"] = attempts
        
        with torch.no_grad():
            out_batch, noise_t, infer_t = miner.forward_batch(nonce_batch)
        
        out_batch_cpu = out_batch.cpu().numpy()
        nonce_batch_cpu = nonce_batch.cpu().numpy()
        
        for i in range(args.batch_size):
            hash_bytes = bytes(out_batch_cpu[i].tolist())
            hash_disp = bytes(out_batch_cpu[i].tolist())[::-1]
            zeros = count_leading_zero_bits(hash_disp)
            if zeros > progress_data["best_zero_bits"]:
                progress_data["best_zero_bits"] = zeros
            if zeros >= args.difficulty:
                solution_found = True
                nonce_bytes = bytes(nonce_batch_cpu[i].tolist())
                solution_nonce = nonce_bytes[::-1]
                solution_hash = hash_disp
                break
    
    progress_data["stop"] = True
    progress_thread.join()
    
    total_time = time.time() - progress_data["start_time"]
    print("\n\nSolution found!")
    print("Nonce:", print_hex(solution_nonce))
    print("Hash: ", print_hex(solution_hash))
    print("Stats:")
    print("  Time: {:.1f} seconds".format(total_time))
    print("  Total hashes: {}".format(attempts))
    print("  Avg hash rate: {:.1f} h/s".format(attempts / total_time))
    tops_overall = ((attempts / total_time) * OPS_PER_HASH) / 1e12
    print("  Avg TOPS: {:.6f}".format(tops_overall))

if __name__ == "__main__":
    main()