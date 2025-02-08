import coremltools as ct
import numpy as np

# Load the Core ML model
model = ct.models.MLModel('test_coreml.mlpackage')

# Display model input and output names
print("Model input names:", model.input_description)
print("Model output names:", model.output_description)

# Generate a random input array with the appropriate shape
# Ensure the input data type matches the model's expected input type
input_data = np.random.rand(1, 256).astype(np.float32)

# Prepare the input dictionary
input_dict = {'input': input_data}

# Perform inference
predictions = model.predict(input_dict)

# Retrieve and display the output
output = predictions['clip_63']
print("Model output:", output)

