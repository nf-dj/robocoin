#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
import hashlib

# Update the model path to your INT8 model file
MODEL_PATH = "tens_hash_int8.onnx"

# Create the inference session using the CPU provider (or adjust as needed)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def hex_to_int8_array(hex_str):
    """Convert a 32-byte hex string to a NumPy int8 array with shape [1,32]."""
    if len(hex_str) != 64:
        raise ValueError("Hex input must be exactly 64 characters (32 bytes).")
    byte_array = bytes.fromhex(hex_str)
    # Convert from uint8 to int8 (two's complement arithmetic)
    return np.frombuffer(byte_array, dtype=np.uint8).astype(np.int8).reshape(1, 32)

def sha256_to_error(input_array):
    """Compute error vector from input using SHA-256.
    Returns an int8 array with shape [1, 32].
    """
    # Get the SHA-256 digest of the input (using uint8 bytes)
    input_bytes = input_array.astype(np.uint8).tobytes()
    digest = hashlib.sha256(input_bytes).digest()
    # Convert the first 32 bytes of the digest to int8
    return np.frombuffer(digest[:32], dtype=np.uint8).astype(np.int8).reshape(1, 32)

def int8_array_to_hex(int8_array):
    """Convert a NumPy int8 array to a 32-byte hex string."""
    # Convert to uint8 so that the bytes are displayed as 0-255
    uint8_array = int8_array.astype(np.uint8)
    return uint8_array.tobytes().hex()

def run_model(hex_input):
    """Run the ONNX INT8 model with a hex input string and return the hex output."""
    # Convert hex string to int8 array (shape [1,32])
    input_array = hex_to_int8_array(hex_input)
    
    # Compute error vector from input
    error = sha256_to_error(input_array)
    
    print("Input shapes:")
    print(f"  input: {input_array.shape}")
    print(f"  error: {error.shape}")

    # Run inference. The model is assumed to expect inputs named "input" and "error".
    output = session.run(None, {
        "input": input_array,
        "error": error
    })[0]
    
    # Convert the int8 output to hex
    return int8_array_to_hex(output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_tens_hash_int8.py <32-byte-hex>")
    
    hex_input = sys.argv[1].strip().lower()
    try:
        print(run_model(hex_input))
    except Exception as e:
        sys.exit("Error: " + str(e))
