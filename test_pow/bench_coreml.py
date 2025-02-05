import coremltools as ct
import numpy as np
import time
from tqdm import tqdm

def calculate_tops(batch_size, input_dim, num_layers, time_taken):
    """Calculate TOPS (Trillion Operations Per Second)"""
    ops_per_gemm = 2 * input_dim * input_dim * input_dim  # multiply-adds
    total_ops = ops_per_gemm * num_layers * batch_size
    tops = (total_ops / time_taken) / 1e12
    return tops

def benchmark_ane_model(batch_sizes=[1, 8, 16, 32, 64, 128], num_iterations=100):
    # Load model with Neural Engine compute unit
    print("Loading model...")
    model = ct.models.MLModel('64x256gemm_ANE_optimized.mlpackage', compute_units=ct.ComputeUnit.ALL)
    
    input_dim = 256
    num_layers = 64
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create input vector with fp16 precision
        input_vector = np.random.randn(input_dim).astype(np.float16)
        
        # Warmup runs
        print("Warming up...")
        for _ in range(10):
            _ = model.predict({'input': input_vector})
        
        # Timed runs
        times = []
        print(f"Running {num_iterations} iterations...")
        for _ in tqdm(range(num_iterations)):
            batch_start_time = time.time()
            
            # Process batch sequentially
            for _ in range(batch_size):
                _ = model.predict({'input': input_vector})
            
            batch_time = time.time() - batch_start_time
            times.append(batch_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        tops = calculate_tops(batch_size, input_dim, num_layers, avg_time)
        
        result = {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'std_time': std_time,
            'tops': tops,
            'samples_per_sec': batch_size / avg_time
        }
        results.append(result)
        
        print(f"\nResults for batch size {batch_size}:")
        print(f"Average time per batch: {avg_time*1000:.2f} ms")
        print(f"Standard deviation: {std_time*1000:.2f} ms")
        print(f"TOPS: {tops:.2f}")
        print(f"Samples per second: {batch_size/avg_time:.2f}")
    
    # Print summary
    print("\nSummary:")
    print("Batch Size | Time (ms) | TOPS | Samples/sec")
    print("-" * 50)
    for r in results:
        print(f"{r['batch_size']:^10d} | {r['avg_time']*1000:^9.2f} | {r['tops']:^4.2f} | {r['samples_per_sec']:^11.2f}")
    
    # Find optimal batch size
    max_tops_result = max(results, key=lambda x: x['tops'])
    print(f"\nOptimal batch size: {max_tops_result['batch_size']}")
    print(f"Maximum TOPS: {max_tops_result['tops']:.2f}")
    print(f"Maximum throughput: {max_tops_result['samples_per_sec']:.2f} samples/sec")

if __name__ == "__main__":
    try:
        benchmark_ane_model()
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        print("Detailed error info:")
        import traceback
        traceback.print_exc()
