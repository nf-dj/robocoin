import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # for progress bar
from Crypto.Cipher import ChaCha20

# --------------------------------------
# Hadamard Matrix Generation and Permutation using ChaCha20
# --------------------------------------

def hadamard_matrix(n):
    """Generate an n x n Hadamard matrix (n must be a power of 2)."""
    if n == 1:
        return np.array([[1]])
    H_n_2 = hadamard_matrix(n // 2)
    return np.block([[H_n_2, H_n_2], [H_n_2, -H_n_2]])

def init_permuted_hadamard(n, seed):
    """
    Generate an n x n Hadamard matrix and permute its rows and columns once.
    The permutation is derived from a 32-byte seed using ChaCha20.
    
    Parameters:
        n (int): Size of the Hadamard matrix (must be a power of 2).
        seed (bytes): A 32-byte seed.
        
    Returns:
        np.ndarray: The permuted Hadamard matrix.
    """
    if len(seed) != 32:
        raise ValueError("A 32-byte seed must be provided.")
    
    # Use a fixed nonce (8 bytes) so that the permutation is fully determined by the seed.
    nonce = b'\0' * 8
    cipher = ChaCha20.new(key=seed, nonce=nonce)
    
    # Generate permutation for rows:
    # Generate n random 64-bit integers (n * 8 bytes)
    row_rand_bytes = cipher.encrypt(b'\0' * (n * 8))
    row_rand_ints = np.frombuffer(row_rand_bytes, dtype=np.uint64)
    # Convert to floats in [0, 1)
    row_rand_floats = row_rand_ints / (2**64)
    row_perm = np.argsort(row_rand_floats)
    
    # Generate permutation for columns:
    col_rand_bytes = cipher.encrypt(b'\0' * (n * 8))
    col_rand_ints = np.frombuffer(col_rand_bytes, dtype=np.uint64)
    col_rand_floats = col_rand_ints / (2**64)
    col_perm = np.argsort(col_rand_floats)
    
    # Generate and permute the Hadamard matrix.
    H = hadamard_matrix(n)
    H = H[row_perm, :][:, col_perm]
    return H

# --------------------------------------
# Multi-Round Hadamard Transform
# --------------------------------------

def binary_hadamard_transform_single_round(v, H):
    """
    Apply one round of the Hadamard transform to a binary vector v using matrix H.
    
    Parameters:
        v (np.ndarray): 1D binary vector with elements 0 or 1.
        H (np.ndarray): Permuted Hadamard matrix.
        
    Returns:
        np.ndarray: Transformed binary vector.
    """
    # Convert {0,1} to {-1,1}
    v_signed = 2 * v - 1
    # Apply the Hadamard transform
    y = H @ v_signed
    # Allocate output vector
    y_binary = np.empty_like(y, dtype=int)
    # For y > 0, output 1; for y < 0, output 0.
    y_binary[y > 0] = 1
    y_binary[y < 0] = 0
    # For ties (y == 0), randomly assign 0 or 1.
    tie_indices = (y == 0)
    y_binary[tie_indices] = np.random.randint(0, 2, size=tie_indices.sum())
    return y_binary

def multi_round_transform(v, H, rounds=16):
    """
    Apply the Hadamard transform repeatedly for a specified number of rounds.
    
    Parameters:
        v (np.ndarray): Input binary vector.
        H (np.ndarray): Permuted Hadamard matrix.
        rounds (int): Number of rounds.
        
    Returns:
        np.ndarray: Final transformed binary vector.
    """
    result = v.copy()
    for _ in range(rounds):
        result = binary_hadamard_transform_single_round(result, H)
    return result

# --------------------------------------
# Randomness Testing Functions
# --------------------------------------

def randomness_tests(n_samples=20000, vector_size=256, rounds=16, seed=None):
    """
    Run the multi-round Hadamard transform on many random inputs and collect outputs.
    The Hadamard matrix is generated and permuted once using the provided 32-byte seed.
    
    Parameters:
        n_samples (int): Number of samples.
        vector_size (int): Size of each input vector (must be a power of 2).
        rounds (int): Number of rounds of the transform.
        seed (bytes): 32-byte seed for generating the permuted Hadamard matrix.
        
    Returns:
        tuple: (bit_means, bit_variances, outputs)
    """
    if seed is None or len(seed) != 32:
        raise ValueError("A 32-byte seed must be provided.")
    
    # Initialize a single permuted Hadamard matrix using the ChaCha20-based seed.
    H = init_permuted_hadamard(vector_size, seed)
    
    outputs = np.zeros((n_samples, vector_size), dtype=int)
    for i in tqdm(range(n_samples), desc="Processing samples"):
        # Generate a random binary vector of length vector_size.
        v = np.random.randint(0, 2, vector_size)
        # Apply multiple rounds of the transform.
        outputs[i] = multi_round_transform(v, H, rounds=rounds)
    
    bit_means = outputs.mean(axis=0)   # Expected ~0.5 per bit.
    bit_variances = outputs.var(axis=0)  # Expected ~0.25 per bit.
    return bit_means, bit_variances, outputs

def shannon_entropy(bits):
    """Calculate the Shannon entropy of a binary array."""
    p1 = np.mean(bits)
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1) if (p0 > 0 and p1 > 0) else 0

def chi_square_uniformity_test(bits):
    """
    Perform a chi-square test for uniformity on a binary sequence.
    
    Returns:
        tuple: (chi-square statistic, p-value)
    """
    counts = np.bincount(bits.astype(int), minlength=2)
    expected = np.array([len(bits) / 2, len(bits) / 2])
    return stats.chisquare(counts, expected)

def runs_test_for_sample(sample):
    """
    Perform a runs test on a binary sequence.
    
    Returns:
        tuple: (z-score, two-sided p-value)
    """
    n = len(sample)
    n1 = np.sum(sample)
    n0 = n - n1
    runs = 1 + np.sum(sample[1:] != sample[:-1])
    mu = (2 * n1 * n0 / n) + 1
    sigma = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))) if n > 1 else 0
    if sigma == 0:
        return 0, 1.0  # Constant sequence.
    z = (runs - mu) / sigma
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val

def autocorrelation(sequence, lag=1):
    """
    Compute the lag-1 autocorrelation of a binary sequence.
    
    Returns:
        float: Pearson correlation coefficient between the sequence and its lagged version.
    """
    n = len(sequence)
    if n <= lag:
        return 0
    seq = sequence - np.mean(sequence)
    std1 = np.std(seq[:-lag])
    std2 = np.std(seq[lag:])
    if std1 == 0 or std2 == 0:
        return 0
    return np.corrcoef(seq[:-lag], seq[lag:])[0, 1]

# --------------------------------------
# Run All Tests and Summarize Results
# --------------------------------------

# Define parameters
vector_size = 256   # Must be a power of 2.
rounds = 16
n_samples = 20000

# Define a valid 32-byte seed (for example, 32 ASCII characters)
seed = b'0123456789ABCDEF0123456789ABCDEF'  # 32 bytes

# Run the randomness tests using the multi-round transform.
bit_means, bit_variances, outputs = randomness_tests(
    n_samples=n_samples,
    vector_size=vector_size,
    rounds=rounds,
    seed=seed
)

# Compute Shannon entropy per bit position.
entropy_values = np.array([shannon_entropy(outputs[:, i]) for i in range(vector_size)])
avg_entropy = np.mean(entropy_values)  # Expected ~1

# Perform chi-square tests for each bit position.
chi2_results = [chi_square_uniformity_test(outputs[:, i]) for i in range(vector_size)]
chi2_values, p_values = zip(*chi2_results)
avg_chi2 = np.mean(chi2_values)
avg_chi_p_value = np.mean(p_values)  # Should be > 0.05 for uniformity

# Apply runs test to each sample.
runs_p_values = np.array([runs_test_for_sample(sample)[1] for sample in outputs])
fraction_failing_runs = np.mean(runs_p_values < 0.05)  # Expected ~5%

# Compute lag-1 autocorrelation for each sample.
autocorr_values = np.array([autocorrelation(sample, lag=1) for sample in outputs])
mean_autocorr = np.mean(autocorr_values)  # Expected to be near 0

# Summarize results.
summary_results = {
    "Average Bit Mean": np.mean(bit_means),
    "Average Bit Variance": np.mean(bit_variances),
    "Average Shannon Entropy": avg_entropy,
    "Average Chi-square Value": avg_chi2,
    "Average Chi-square p-value": avg_chi_p_value,
    "Fraction Failing Runs Test": fraction_failing_runs,
    "Mean Lag-1 Autocorrelation": mean_autocorr,
}

print("\nSummary of Randomness Tests:")
for key, value in summary_results.items():
    print(f"{key}: {value:.4f}")

# --------------------------------------
# Check Test Conditions
# --------------------------------------

passed = True

if not (0.49 < summary_results["Average Bit Mean"] < 0.51):
    print("[FAIL] Bit Mean is outside expected range (0.49 - 0.51)")
    passed = False

if not (0.24 < summary_results["Average Bit Variance"] < 0.26):
    print("[FAIL] Bit Variance is outside expected range (0.24 - 0.26)")
    passed = False

if not (0.98 < summary_results["Average Shannon Entropy"] < 1.02):
    print("[FAIL] Shannon Entropy is outside expected range (0.98 - 1.02)")
    passed = False

if not (summary_results["Average Chi-square p-value"] > 0.05):
    print("[FAIL] Chi-square p-value is too low (indicates non-uniform distribution)")
    passed = False

if not (0.03 < summary_results["Fraction Failing Runs Test"] < 0.07):
    print("[FAIL] Fraction failing runs test is outside expected range (0.03 - 0.07)")
    passed = False

if not (abs(summary_results["Mean Lag-1 Autocorrelation"]) < 0.05):
    print("[FAIL] Mean lag-1 autocorrelation is too high (indicates dependency between bits)")
    passed = False

if passed:
    print("\n[PASS] All randomness tests passed!")
else:
    print("\n[FAIL] Randomness tests failed.")

