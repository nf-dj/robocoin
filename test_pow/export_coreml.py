#!/usr/bin/env python3
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes

def create_model():
    # Test case: 4x4 matrix
    input_features = [('input', datatypes.Array(1, 4))]
    output_features = [('output', datatypes.Array(1, 4))]
    
    # Create a simple test ternary matrix
    matrix = np.array([
        [1, -1, 0, 1],
        [-1, 1, 1, 0],
        [0, 1, -1, -1],
        [1, 0, -1, 1]
    ], dtype=np.float32)
    
    print("Matrix used:")
    print(matrix)
    print("\nMatrix shape:", matrix.shape)
    print("Matrix dtype:", matrix.dtype)
    print("Matrix min/max:", np.min(matrix), np.max(matrix))
    
    # Look at model params
    builder = NeuralNetworkBuilder(
        input_features,
        output_features,
        disable_rank5_shape_mapping=True
    )
    
    # Just one matrix multiply
    builder.add_inner_product(
        name='matmul',
        input_name='input',
        output_name='output',
        input_channels=4,
        output_channels=4,
        W=matrix,
        b=None,
        has_bias=False
    )
    
    # Print spec details
    print("\nLayer spec:")
    layer = builder.spec.neuralNetwork.layers[0]
    print("Layer type:", layer.innerProduct)
    print("Layer weights shape:", layer.innerProduct.weights.floatValue)
    
    return builder

def main():
    try:
        print("Creating model...")
        builder = create_model()
        
        spec = builder.spec
        model = ct.models.MLModel(spec)
        
        print("\nSaving model...")
        model.save("tens_hash.mlmodel")
        
        print("Compiling model...")
        import os
        os.system(f"xcrun coremlc compile tens_hash.mlmodel .")
        print("Model compiled successfully!")
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()