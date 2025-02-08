#!/usr/bin/env python3
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes

def create_model():
    # Use 256x256 matrix
    input_features = [('input', datatypes.Array(1, 256))]
    output_features = [('output', datatypes.Array(1, 256))]
    
    # Create a random ternary matrix
    matrix = np.random.choice([-1, 0, 1], size=(256, 256), p=[0.25, 0.5, 0.25]).astype(np.float32)
    print(f"Matrix shape: {matrix.shape}")
    
    builder = NeuralNetworkBuilder(
        input_features,
        output_features,
        disable_rank5_shape_mapping=True
    )
    
    # Reshape input 
    builder.add_reshape(
        name="reshape_input",
        input_name="input",
        output_name="input_reshaped",
        target_shape=(1, 1, 1, 1, 256),  # Rank 5 as required
        mode=0
    )
    
    # Load matrix weights
    matrix_name = 'matrix_const'
    builder.add_load_constant(
        name=matrix_name,
        output_name=matrix_name,
        constant_value=matrix,
        shape=[256, 256]
    )
    
    # Reshape matrix to rank 5
    builder.add_reshape(
        name="reshape_matrix",
        input_name=matrix_name,
        output_name="matrix_reshaped",
        target_shape=(1, 1, 1, 256, 256),  # Rank 5
        mode=0
    )
    
    # Reshape input for multiply
    builder.add_reshape(
        name="reshape_input_2",
        input_name="input_reshaped",
        output_name="input_for_multiply",
        target_shape=(1, 1, 1, 1, 256),
        mode=0
    )
    
    # Elementwise multiply
    builder.add_multiply_broadcastable(
        name="multiply",
        input_names=["input_for_multiply", "matrix_reshaped"],
        output_name="multiply_out"
    )
    
    # Sum along last dimension
    builder.add_reduce_sum(
        name="sum",
        input_name="multiply_out",
        output_name="sum_out",
        axes=[4],  # Sum along last dimension (now index 4 due to rank 5)
        keepdims=True
    )
    
    # Final reshape back to rank 2
    builder.add_reshape(
        name="reshape_output",
        input_name="sum_out",
        output_name="output",
        target_shape=(1, 256),
        mode=0
    )
    
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