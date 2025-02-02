#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
import hashlib

# Load the ONNX model
MODEL_PATH = "tens_hash.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def hex_to_int8_array(hex_str):
    """Convert a 32-byte hex string to a NumPy int8 array with shape [1,32]."""
    if len(hex_str) != 64:
        raise ValueError("Hex input must be exactly 64 characters (32 bytes).")
    byte_array = bytes.fromhex(hex_str)
    return np.frombuffer(byte_array, dtype=np.int8).reshape(1, 32)

def int8_array_to_hex(int8_array):
    """Convert a NumPy int8 array to a 32-byte hex string."""
    byte_array = int8_array.astype(np.uint8).tobytes()
    return byte_array.hex()

def compute_error_vector(input_array):
    """
    Compute the error vector from the input.
    This is done by taking SHA-256 of the input bytes and using the 32-byte digest.
    The result is then interpreted as an int8 array with shape [1,32].
    """
    # Compute SHA-256 digest of the input bytes
    digest = hashlib.sha256(input_array.tobytes()).digest()
    # Convert digest to int8 array with shape [1,32]
    error = np.frombuffer(digest, dtype=np.int8).reshape(1, 32)
    return error

def run_model(hex_input):
    """Run the ONNX model with a hex input string and return the hex output."""
    # Convert hex string to int8 array (shape [1,32])
    input_array = hex_to_int8_array(hex_input)
    # Compute error vector from the input using SHA-256
    error_array = compute_error_vector(input_array)
    # Run inference
    output = session.run(None, {"input": input_array, "error": error_array})[0]
    # Convert output to hex
    return int8_array_to_hex(output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_tens_hash.py <32-byte-hex>")
    
    hex_input = sys.argv[1].strip().lower()
    try:
        print(run_model(hex_input))
    except Exception as e:
        sys.exit("Error: " + str(e))
