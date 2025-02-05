import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes
import numpy as np

# Configuration
input_dim = 256
num_layers = 64
dtype = np.float16

def create_model():
    # Create input and output features
    input_features = [('input', datatypes.Array(input_dim))]
    output_features = [('output', datatypes.Array(input_dim))]
    
    builder = NeuralNetworkBuilder(
        input_features,
        output_features,
        disable_rank5_shape_mapping=True
    )

    # Initialize hidden layer input name
    previous_output = "input"

    # Create 64-layer GEMM sequence
    for layer_idx in range(num_layers):
        # Generate random weights/bias for GEMM (256x256 matrix)
        weights = np.random.randn(input_dim, input_dim).astype(dtype)
        bias = np.random.randn(input_dim).astype(dtype)

        # GEMM Layer (256x256 matrix multiply)
        gemm_name = f"gemm_{layer_idx}"
        gemm_output = f"{gemm_name}_out" if layer_idx < num_layers - 1 else "output"
        
        builder.add_inner_product(
            name=gemm_name,
            input_name=previous_output,
            output_name=gemm_output,
            input_channels=input_dim,
            output_channels=input_dim,
            W=weights,
            b=bias,
            has_bias=True
        )
        
        # Set input for next layer
        previous_output = gemm_output

    return builder

if __name__ == "__main__":
    try:
        # Create model architecture
        builder = create_model()
        
        # Configure model metadata
        spec = builder.spec
        spec.description.metadata.author = "Your Name"
        spec.description.metadata.shortDescription = (
            f"64-layer 256x256 GEMM network "
            f"(ANE-optimized, fp16)"
        )
        spec.specificationVersion = 5  # Core ML 5 (iOS 16+)
        
        # Create model with ANE optimizations
        model = ct.models.MLModel(
            spec,
            compute_units=ct.ComputeUnit.ALL
        )
        
        model.save("64x256gemm_ANE_optimized.mlpackage")
        print("Model saved successfully!")
        
        # Test the model
        test_input = np.random.randn(input_dim).astype(np.float16)
        result = model.predict({'input': test_input})
        print("Model test successful!")
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        import traceback
        traceback.print_exc()
