#!/usr/bin/env python3
import numpy as np
import coremltools as ct

def binary_to_hex(binary_array):
    # Convert binary array to bytes then to hex
    bits = ''.join(str(int(bit)) for bit in binary_array)
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        bytes_list.append(int(byte, 2))
    return bytes(bytes_list).hex()

def main():
    print("Loading model...")
    model = ct.models.MLModel("tens_hash.mlmodel")
    
    print("\nModel inputs:", model.input_description)
    print("Model outputs:", model.output_description)
    
    # Create test input - all zeros
    input_vector = np.zeros(256, dtype=np.float32)
    noise_vector = np.zeros(256, dtype=np.float32)
    
    print("\nTesting with zero vectors...")
    input_data = {
        "input": input_vector,
        "noise": noise_vector
    }
    
    try:
        prediction = model.predict(input_data)
        print("\nPrediction successful!")
        print("Output shape:", prediction['output'].shape)
        print("Output hex:", binary_to_hex(prediction['output']))
    except Exception as e:
        print("\nPrediction failed:", str(e))
        
if __name__ == "__main__":
    main()
