import torch
import torch.nn as nn
import time
import os

# Use MPS (Metal Performance Shaders) if available, otherwise fallback to CPU
use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")

# Allow FP16 execution but fallback to FP32 if error occurs
use_fp16 = True  # Try FP16 first

# Print device info
print(f"🔹 Using device: {device} (FP16: {use_fp16})")
if not use_mps:
    print("⚠️ Warning: MPS not available! Running on CPU, which will be much slower.")

class TernaryLinear(nn.Module):
    """Custom Linear Layer with Ternary Weights (-1, 0, 1) and Binary Bias (0,1)."""
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(self.init_ternary_weights(), requires_grad=False)
        self.bias = nn.Parameter(self.init_binary_bias(), requires_grad=False)

    def init_ternary_weights(self):
        """Initialize weights randomly with values {-1, 0, 1}."""
        dtype = torch.float16 if use_fp16 else torch.float32
        return torch.randint(-1, 2, (self.out_features, self.in_features), dtype=dtype).to(device)

    def init_binary_bias(self):
        """Initialize bias randomly with values {0, 1}."""
        dtype = torch.float16 if use_fp16 else torch.float32
        return torch.randint(0, 2, (self.out_features,), dtype=dtype).to(device)

    def forward(self, x):
        try:
            result = torch.matmul(x, self.weight.T) + self.bias  # GEMM + Bias
            result = torch.fmod(result, 2.0)  # Apply mod 2
        except RuntimeError as e:
            if "trunc_divide op with float16 input" in str(e):
                print("⚠️ FP16 failed! Falling back to FP32.")
                global use_fp16
                use_fp16 = False
                self.weight = nn.Parameter(self.init_ternary_weights(), requires_grad=False)
                self.bias = nn.Parameter(self.init_binary_bias(), requires_grad=False)
                x = x.to(torch.float32)  # Convert input to FP32
                result = torch.matmul(x, self.weight.T) + self.bias
                result = torch.fmod(result, 2.0)  # Apply mod 2 in FP32
        return result

class GEMMMod2Model(nn.Module):
    def __init__(self):
        super(GEMMMod2Model, self).__init__()
        self.layers = nn.ModuleList([TernaryLinear(256, 256) for _ in range(64)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Initialize model and move to MPS/CPU
dtype = torch.float16 if use_fp16 else torch.float32
model = GEMMMod2Model().to(device).to(dtype)

# Print device info to verify tensors are correctly assigned
print(f"✅ Model is running on: {next(model.parameters()).device} (dtype: {dtype})")

# Warm-up to avoid MPS kernel compilation overhead
print("🔄 Warming up MPS...")
warmup_batch = torch.randint(0, 2, (16, 256), dtype=dtype).to(device)

try:
    for _ in range(100):  
        _ = model(warmup_batch)
except RuntimeError as e:
    if "trunc_divide op with float16 input" in str(e):
        print("⚠️ FP16 failed during warm-up! Switching to FP32.")
        use_fp16 = False
        dtype = torch.float32
        model = GEMMMod2Model().to(device).to(dtype)
        warmup_batch = warmup_batch.to(dtype)

# Compute number of operations per inference
operations_per_inference = 64 * (256 * 256) * 2  # Multiply-accumulate (MAC) operations

batch_size = 256
max_batch_size = 65536
start_time = time.perf_counter()

print("🚀 Starting continuous benchmark (Press Ctrl+C to stop)...")

try:
    while batch_size <= max_batch_size:
        # Create input tensor
        x_tensor = torch.randint(0, 2, (batch_size, 256), dtype=dtype).to(device)

        num_inferences = 0
        batch_start_time = time.perf_counter()

        while time.perf_counter() - batch_start_time < 10.0:  # Run for 10 seconds per batch size
            _ = model(x_tensor)
            num_inferences += batch_size  # Account for batch size

        end_time = time.perf_counter()
        elapsed_time = end_time - batch_start_time

        # Compute performance
        ips = num_inferences / elapsed_time  # Inferences per second
        total_operations = ips * operations_per_inference
        tops = total_operations / 1e12  # Convert to TeraOPS

        print(f"🟢 Batch Size: {batch_size} | ⚡ IPS: {ips:.2f} | Estimated TOPS: {tops:.3f}")

        # Double the batch size
        batch_size *= 2

except KeyboardInterrupt:
    print("\n🛑 Benchmark stopped.")

# Run CPU comparison
print("\n🔄 Running CPU comparison...")
device = torch.device("cpu")
model = GEMMMod2Model().to(device).to(dtype)

cpu_batch_size = 16  # Small batch for CPU
x_cpu = torch.randint(0, 2, (cpu_batch_size, 256), dtype=dtype).to(device)

num_inferences = 0
cpu_start_time = time.perf_counter()

while time.perf_counter() - cpu_start_time < 10.0:
    _ = model(x_cpu)
    num_inferences += cpu_batch_size

cpu_end_time = time.perf_counter()
cpu_elapsed_time = cpu_end_time - cpu_start_time
cpu_ips = num_inferences / cpu_elapsed_time
cpu_total_operations = cpu_ips * operations_per_inference
cpu_tops = cpu_total_operations / 1e12

print(f"🖥️ CPU Test | ⚡ IPS: {cpu_ips:.2f} | Estimated TOPS: {cpu_tops:.3f}")

