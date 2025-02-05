import torch
import torch.nn as nn
from hashlib import sha256
from Crypto.Cipher import ChaCha20
import argparse
import numpy as np

IN_SIZE = 32
HIDDEN = 256
ROUNDS = 64

class TensHash(nn.Module):
    def __init__(self, seed_hex, device="mps"):
        super().__init__()
        self.device = torch.device(device)
        self.seed = bytes.fromhex(seed_hex)[::-1]
        
        total_size = ROUNDS * HIDDEN * HIDDEN
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0'*8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        # Store as single tensor for efficiency
        self.matrices = torch.tensor(
            [[(random_bytes[r*HIDDEN*HIDDEN + i] % 3) - 1 
              for i in range(HIDDEN*HIDDEN)] 
             for r in range(ROUNDS)],
            dtype=torch.float32, device=self.device
        ).reshape(ROUNDS, HIDDEN, HIDDEN)

    def bytes_to_bits(self, input_bytes):
        input_array = np.frombuffer(input_bytes, dtype=np.uint8)
        bits = np.zeros(IN_SIZE * 8, dtype=np.float32)
        for byte_idx in range(IN_SIZE):
            for bit_idx in range(8):
                bits[byte_idx * 8 + bit_idx] = float((input_array[byte_idx] >> bit_idx) & 1)
        return torch.from_numpy(bits).to(self.device)

    def bits_to_bytes(self, bits):
        output = bytearray(IN_SIZE)
        bits_cpu = bits.cpu().numpy()
        for byte_idx in range(IN_SIZE):
            byte = 0
            for bit_idx in range(8):
                if bits_cpu[byte_idx * 8 + bit_idx] > 0.5:
                    byte |= 1 << bit_idx
            output[byte_idx] = byte
        return bytes(output)

    def generate_noise(self, input_bytes):
        input_hash = sha256(input_bytes).digest()
        noise = np.zeros(ROUNDS * HIDDEN, dtype=np.float32)
        for i in range(ROUNDS * HIDDEN):
            byte_idx = i % 32
            bit_idx = i % 8
            noise[i] = float((input_hash[byte_idx] >> bit_idx) & 1)
        return torch.from_numpy(noise).reshape(ROUNDS, HIDDEN).to(self.device)

    def hash(self, input_bytes):
        state = self.bytes_to_bits(input_bytes)
        noise = self.generate_noise(input_bytes)
        
        for r in range(ROUNDS):
            # Use batched matrix multiply for speed but match C's behavior
            state = torch.matmul(state, self.matrices[r].t()) + noise[r]
            state = torch.remainder(torch.floor(state), 2.0)
        
        return self.bits_to_bytes(state)

def main():
    parser = argparse.ArgumentParser(description='TensHash PyTorch Implementation')
    parser.add_argument('seed', help='64-character hex seed')
    parser.add_argument('input', help='64-character hex input')
    args = parser.parse_args()
    
    if len(args.seed) != 64:
        parser.error("Seed must be 64 hex characters")
    if len(args.input) != 64:
        parser.error("Input must be 64 hex characters")
    
    seed_hex = args.seed
    input_bytes = bytes.fromhex(args.input)[::-1]
    
    hasher = TensHash(seed_hex)
    output = hasher.hash(input_bytes)
    print(output[::-1].hex())

if __name__ == "__main__":
    main()