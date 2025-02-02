#!/usr/bin/env python3
import subprocess
import random
import sys
from typing import Tuple

def generate_random_hex(length: int) -> str:
    """Generate a random hex string of specified length."""
    return ''.join(random.choice('0123456789abcdef') for _ in range(length))

def run_hash(seed: str, input_hex: str, impl_type: int) -> str:
    """Run the hash binary with given parameters and return output."""
    cmd = ['./tens_hash', seed, input_hex, str(impl_type)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

def generate_test_case() -> Tuple[str, str]:
    """Generate a random test case (seed and input)."""
    seed = generate_random_hex(64)
    input_hex = generate_random_hex(64)
    return seed, input_hex

def compare_implementations(seed: str, input_hex: str) -> bool:
    """Compare results from both implementations."""
    int_result = run_hash(seed, input_hex, 0)
    fp_result = run_hash(seed, input_hex, 1)
    
    match = int_result == fp_result
    if not match:
        print(f"\nMismatch found!")
        print(f"Seed:  {seed}")
        print(f"Input: {input_hex}")
        print(f"Int8:  {int_result}")
        print(f"FP32:  {fp_result}")
    
    return match

def main():
    num_tests = 1000
    matches = 0
    
    print(f"Running {num_tests} test cases...")
    
    for i in range(num_tests):
        seed, input_hex = generate_test_case()
        if compare_implementations(seed, input_hex):
            matches += 1
        
        progress = (i + 1) / num_tests * 100
        print(f"\rProgress: {progress:.1f}% ({matches}/{i+1} matches)", end='', flush=True)
    
    print("\n")
    if matches == num_tests:
        print("✅ All implementations match!")
    else:
        print(f"❌ Found differences: {matches}/{num_tests} matches")

if __name__ == "__main__":
    main()
