import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Define random weights and biases
weights = np.random.rand(256, 256).astype(np.float32)
bias = np.random.rand(1, 256).astype(np.float32)

# Define input specifications with a fixed batch size
input_specs = [
    mb.TensorSpec(shape=(1, 256)),  # Fixed batch size of 1
]

# Define the MIL program with embedded constants
@mb.program(input_specs=input_specs)
def matmul_scaled_bias_clamped_relu_prog(input):
    for _ in range(64):
        input = mb.matmul(x=input, y=weights)  # Matrix Multiplication with constant weights
        input = mb.mul(x=input, y=2.0)         # Multiply by 2
        input = mb.add(x=input, y=bias)        # Add constant bias
        input = mb.relu(x=input)               # Apply ReLU
        input = mb.clip(x=input, alpha=0.0, beta=1.0)  # Clamp to [0, 1]
    return input

# Convert to Core ML model with FP16 precision
mlmodel = ct.convert(
    matmul_scaled_bias_clamped_relu_prog,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    outputs=[ct.TensorType(name="output")],  # Specify the output name
)

# Save the model
mlmodel.save("test_coreml.mlpackage")
print("Model saved successfully")

