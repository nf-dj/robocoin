#!/usr/bin/env python3
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import subprocess
from Crypto.Cipher import ChaCha20

BATCH_SIZE = 128  # Match the batch size with Objective-C code

class PowModel(nn.Module):
    def __init__(self, seed):
        super(PowModel, self).__init__()
        self.input_size = 256
        self.output_size = 256
        self.rounds = 64
        
        # Generate the ternary matrix as a parameter
        ternary_matrix = self.generate_ternary_matrix_from_seed(seed)
        self.register_parameter('ternary_matrix', 
                                nn.Parameter(ternary_matrix.float(), requires_grad=False))
        
        # Register bias as a parameter
        bias = -torch.sum(ternary_matrix, dim=0)
        self.register_parameter('bias', 
                                nn.Parameter(bias.float(), requires_grad=False))

    def generate_ternary_matrix_from_seed(self, seed):
        A = torch.zeros((self.input_size, self.output_size), dtype=torch.float32)
        pos_count = neg_count = 32

        for i in range(self.input_size):
            nonce = i.to_bytes(8, 'big')
            cipher = ChaCha20.new(key=seed, nonce=nonce)
            
            rand_bytes = cipher.encrypt(b'\x00' * (self.output_size * 4))
            rand_ints = np.frombuffer(rand_bytes, dtype=np.int32)
            chosen_indices = np.argsort(rand_ints)[:64]
            
            rand_bytes_shuffle = cipher.encrypt(b'\x00' * (64 * 4))
            shuffle_ints = np.frombuffer(rand_bytes_shuffle, dtype=np.int32)
            shuffle_perm = np.argsort(shuffle_ints)
            
            # Create tensors with float32 dtype
            sign_vector = torch.tensor([1.0] * pos_count + [-1.0] * neg_count, dtype=torch.float32)
            chosen_indices_tensor = torch.tensor(chosen_indices, dtype=torch.long)
            shuffle_perm_tensor = torch.tensor(shuffle_perm, dtype=torch.long)
            
            # Apply the permutation and assignment
            sign_vector = sign_vector[shuffle_perm_tensor]
            A[i].index_put_((chosen_indices_tensor,), sign_vector)
        
        return A

    def forward(self, x, noise):
        # Ensure inputs are FP16
        x = x.half()
        noise = noise.half()
        outputs = x
        
        # Add bias to noise (cast bias to FP16)
        bias_plus_noise = self.bias.unsqueeze(0).half() + noise
        
        # Apply rounds using FP16 operations
        for _ in range(self.rounds):
            results = torch.matmul(outputs, self.ternary_matrix.half())
            results = 2.0 * results + bias_plus_noise
            outputs = (results > 0).half()
        return outputs

def export_model(seed_hex):
    # Convert hex seed to bytes
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise ValueError("Seed must be 32 bytes (64 hex chars)")

    # Create and initialize the model
    model = PowModel(seed)
    model.eval()
    
    # Convert model to FP16 so that all parameters and ops use half precision
    model = model.half()

    # Create example inputs with FP16 dtype and proper batch dimension.
    # Use non-zero (random) inputs for tracing!
    example_binary = torch.randint(0, 2, (BATCH_SIZE, 256), dtype=torch.float16)
    example_noise = torch.randint(0, 2, (BATCH_SIZE, 256), dtype=torch.float16)

    # Trace the model with FP16 inputs
    traced_model = torch.jit.trace(model, (example_binary, example_noise))

    # Define input types with batch dimension and FP16 dtype
    input_binary = ct.TensorType(shape=(BATCH_SIZE, 256), dtype=np.float16, name="binary_input")
    input_noise = ct.TensorType(shape=(BATCH_SIZE, 256), dtype=np.float16, name="noise_input")

    # Convert to Core ML with FP16 compute precision and minimum deployment target iOS16
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_binary, input_noise],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16
    )

    # Save the model as an ML Program (.mlpackage is required)
    mlmodel_filename = "PowModel.mlpackage"
    coreml_model.save(mlmodel_filename)
    print(f"Core ML model saved as {mlmodel_filename}")

    # Compile the model using coremlc
    compile_cmd = ["xcrun", "coremlc", "compile", mlmodel_filename, "."]
    try:
        subprocess.run(compile_cmd, check=True)
        print("Model compiled successfully into PowModel.mlmodelc")
    except subprocess.CalledProcessError as e:
        print("Error during model compilation:", e)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: export_pow_coreml.py <32-byte-hex-seed>")
        sys.exit(1)
    
    export_model(sys.argv[1])

