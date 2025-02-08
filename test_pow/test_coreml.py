#!/usr/bin/env python3
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import subprocess

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Create a linear layer with 4 input features and 4 output features, no bias.
        self.linear = nn.Linear(4, 4, bias=False)
        # Define a 4x4 ternary weight matrix.
        matrix = torch.tensor([
            [ 1.0,  0.0, -1.0,  1.0],
            [ 0.0, -1.0,  1.0,  0.0],
            [-1.0,  1.0,  0.0, -1.0],
            [ 1.0,  0.0,  1.0, -1.0]
        ], dtype=torch.float32)
        # Add a small epsilon to force full-precision behavior.
        matrix += 1e-6  
        with torch.no_grad():
            self.linear.weight.copy_(matrix)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    # Instantiate the model and set it to evaluation mode.
    model = SimpleModel()
    model.eval()

    # Create an example input tensor of shape [1, 4].
    example_input = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    # Trace the model using TorchScript.
    traced_model = torch.jit.trace(model, example_input)

    # Define the input type with a name that matches your expected feature name.
    # Do NOT specify outputs; they will be inferred automatically.
    input_type = ct.TensorType(shape=example_input.shape, name="input")

    # Convert the traced PyTorch model to Core ML.
    # Specify source="pytorch" and set the minimum deployment target.
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_type],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS14
    )

    # Save the Core ML model.
    mlmodel_filename = "SimpleModel.mlmodel"
    coreml_model.save(mlmodel_filename)
    print(f"Core ML model saved as {mlmodel_filename}")

    # Compile the .mlmodel into a .mlmodelc folder using the Core ML compiler.
    compile_cmd = ["xcrun", "coremlc", "compile", mlmodel_filename, "."]
    try:
        subprocess.run(compile_cmd, check=True)
        print("Model compiled successfully into SimpleModel.mlmodelc")
    except subprocess.CalledProcessError as e:
        print("Error during model compilation:", e)

