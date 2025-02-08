import coremltools as ct
import numpy as np

# Load the Core ML model
model = ct.models.MLModel('test_coreml.mlpackage')

# Display model input and output names
print("Model input names:", model.input_description)
print("Model output names:", model.output_description)

# Generate random input data with the appropriate shape
input_data = np.random.rand(1024, 256).astype(np.float32)  # Shape: (batch_size, feature_size)
bias_data = np.random.rand(1024, 256).astype(np.float32)   # Shape: (batch_size, feature_size)

# Prepare the input dictionary
input_dict = {'input': input_data, 'bias': bias_data}

# Perform inference
predictions = model.predict(input_dict)

# Retrieve and display the output
output = predictions['clip_63']
print("Model output:", output)

