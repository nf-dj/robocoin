#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
import hashlib

# Load the ONNX model
MODEL_PATH = "tens_hash_fp32.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def hex_to_float32_array(hex_str):
    """Convert a 32-byte hex string to a NumPy float32 array with shape [1,32]."""
    if len(hex_str) != 64:
        raise ValueError("Hex input must be exactly 64 characters (32 bytes).")
    byte_array = bytes.fromhex(hex_str)
    return np.frombuffer(byte_array, dtype=np.uint8).astype(np.float32).reshape(1, 32)

def sha256_to_error(input_array):
    """Compute error vector from input using SHA-256.
    Returns array with shape [1, 32] for error input
    """
    # Get the SHA-256 digest of the input
    input_bytes = input_array.astype(np.uint8).tobytes()
    digest = hashlib.sha256(input_bytes).digest()
    
    # Error is first 32 bytes shaped to [1, 32]
    error = np.array([digest[i] for i in range(32)], dtype=np.float32).reshape(1, 32)
    
    return error

def float32_array_to_hex(float32_array):
    """Convert a NumPy float32 array to a 32-byte hex string."""
    uint8_array = float32_array.astype(np.uint8)
    return uint8_array.tobytes().hex()

def run_model(hex_input):
    """Run the ONNX model with a hex input string and return the hex output."""
    # Convert hex string to float32 array (shape [1,32])
    input_array = hex_to_float32_array(hex_input)
    
    # Compute error vector from input
    error = sha256_to_error(input_array)
    
    print("Input shapes:")
    print(f"  input: {input_array.shape}")
    print(f"  error: {error.shape}")

    # Run inference
    output = session.run(None, {
        "input": input_array,
        "error": error
    })[0]
    
    # Convert output to hex
    return float32_array_to_hex(output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_tens_hash_fp32.py <32-byte-hex>")
    
    hex_input = sys.argv[1].strip().lower()
    try:
        print(run_model(hex_input))
    except Exception as e:
        sys.exit("Error: " + str(e))