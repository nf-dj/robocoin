#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
import hashlib

# Update the model path to your UINT8 model file
MODEL_PATH = "tens_hash_uint8.onnx"

# Create the inference session using the CPU provider (or adjust as needed)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def hex_to_uint8_array(hex_str):
    """Convert a 32-byte hex string to a NumPy uint8 array with shape [1,32]."""
    if len(hex_str) != 64:
        raise ValueError("Hex input must be exactly 64 characters (32 bytes).")
    byte_array = bytes.fromhex(hex_str)
    return np.frombuffer(byte_array, dtype=np.uint8).reshape(1, 32)

def sha256_to_error(input_array):
    """Compute error vector from input using SHA-256.
    Returns a uint8 array with shape [1, 32].
    """
    # Get the SHA-256 digest of the input (using uint8 bytes)
    input_bytes = input_array.tobytes()
    digest = hashlib.sha256(input_bytes).digest()
    return np.frombuffer(digest[:32], dtype=np.uint8).reshape(1, 32)

def uint8_array_to_hex(uint8_array):
    """Convert a NumPy uint8 array to a 32-byte hex string."""
    return uint8_array.tobytes().hex()

def run_model(hex_input):
    """Run the ONNX UINT8 model with a hex input string and return the hex output."""
    # Convert hex string to uint8 array (shape [1,32])
    input_array = hex_to_uint8_array(hex_input)
    
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
    
    # Convert the uint8 output to hex
    return uint8_array_to_hex(output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_tens_hash_uint8.py <32-byte-hex>")
    
    hex_input = sys.argv[1].strip().lower()
    try:
        print(run_model(hex_input))
    except Exception as e:
        sys.exit("Error: " + str(e))

