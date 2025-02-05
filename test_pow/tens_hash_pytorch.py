import torch
import torch.nn as nn
from hashlib import sha256
from Crypto.Cipher import ChaCha20
import argparse

IN_SIZE = 32
HIDDEN = 256
ROUNDS = 64

class TensHash(nn.Module):
    def __init__(self, seed_hex, device="mps"):
        super().__init__()
        # Choose device: prefer MPS if available and requested, otherwise CUDA, else CPU.
        self.device = torch.device(device if torch.backends.mps.is_available() and device == "mps" 
                                   else "cuda" if torch.cuda.is_available() else "cpu")
        # Parse the seed from hex (user provides big-endian),
        # then reverse it to store as little-endian (to mimic the C code behavior).
        self.seed = bytes.fromhex(seed_hex)[::-1]
        
        # Generate matrices using ChaCha20.
        total_size = ROUNDS * HIDDEN * HIDDEN
        # ChaCha20 in PyCryptodome requires an 8-byte nonce.
        cipher = ChaCha20.new(key=self.seed, nonce=b'\0'*8)
        random_bytes = cipher.encrypt(b'\0' * total_size)
        
        # Create and store matrices.
        # Note: We use float32 here to match the IMPL_FP32 mode in your C code.
        self.matrices = nn.ParameterList([
            nn.Parameter(
                torch.tensor(
                    [(random_bytes[r*HIDDEN*HIDDEN + i] % 3) - 1 for i in range(HIDDEN*HIDDEN)],
                    dtype=torch.float32, device=self.device
                ).reshape(HIDDEN, HIDDEN),
                requires_grad=False
            ) for r in range(ROUNDS)
        ])

    def bytes_to_bits(self, input_bytes):
        """Convert bytes to a bit tensor.
           Assumes input_bytes is provided in little-endian order.
        """
        # Create a tensor of the bytes.
        input_tensor = torch.tensor(list(input_bytes), device=self.device)
        bits = torch.zeros(IN_SIZE * 8, dtype=torch.float32, device=self.device)
        
        # Convert each byte to its 8 bits (LSB is bit0).
        for byte_idx in range(IN_SIZE):
            for bit_idx in range(8):
                bits[byte_idx * 8 + bit_idx] = (input_tensor[byte_idx] >> bit_idx) & 1
        
        return bits

    def bits_to_bytes(self, bits):
        """Convert bit tensor back to bytes.
           Produces bytes in little-endian order.
        """
        output = bytearray(IN_SIZE)
        bits_cpu = bits.cpu()
        
        for byte_idx in range(IN_SIZE):
            byte = 0
            for bit_idx in range(8):
                if bits_cpu[byte_idx * 8 + bit_idx] > 0.5:
                    byte |= 1 << bit_idx
            output[byte_idx] = byte
            
        return bytes(output)

    def generate_noise(self, input_bytes):
        """Generate noise from input using SHA256."""
        input_hash = sha256(input_bytes).digest()
        noise = torch.zeros(ROUNDS, HIDDEN, dtype=torch.float32, device=self.device)
        
        for r in range(ROUNDS):
            for h in range(HIDDEN):
                idx = (r * HIDDEN + h)
                byte_idx = idx % 32
                bit_idx = idx % 8
                noise[r, h] = (input_hash[byte_idx] >> bit_idx) & 1
                
        return noise

    def hash(self, input_bytes):
        # Convert input bytes (assumed little-endian internally) to bits.
        state = self.bytes_to_bits(input_bytes)
        
        # Generate noise from the original input.
        noise = self.generate_noise(input_bytes)
        
        # Apply the 64 rounds of matrix multiplication mod 2.
        for r in range(ROUNDS):
            state = torch.fmod(
                torch.matmul(self.matrices[r], state) + noise[r],
                2.0
            )
        
        # Convert the final state (bits) back to bytes (little-endian).
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
    
    # For consistency with the C code:
    #   - Parse the provided hex (which is big-endian) and then convert it to little-endian.
    seed_hex = args.seed
    input_bytes = bytes.fromhex(args.input)[::-1]
    
    # Initialize the hash function.
    hasher = TensHash(seed_hex)
    
    # Compute the hash.
    output = hasher.hash(input_bytes)
    
    # Display the output in big-endian order by reversing the internal little-endian representation.
    print(output[::-1].hex())

if __name__ == "__main__":
    main()

