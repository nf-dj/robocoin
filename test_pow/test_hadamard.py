import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # for progress bar

def hadamard_matrix(n):
    """Generate an n x n Hadamard matrix (n must be a power of 2)."""
    if n == 1:
        return np.array([[1]])
    H_n_2 = hadamard_matrix(n // 2)
    return np.block([[H_n_2, H_n_2], [H_n_2, -H_n_2]])

def binary_hadamard_transform(v, permute=True):
    """
    Apply the Hadamard transform to a binary vector {0,1} and return binary output.
    If permute is True, randomly permute the rows and columns of the Hadamard matrix.
    """
    n = len(v)
    H = hadamard_matrix(n)
    
    if permute:
        # Randomly permute the rows and columns
        row_perm = np.random.permutation(n)
        col_perm = np.random.permutation(n)
        H = H[row_perm, :][:, col_perm]

    # Convert {0,1} input to {-1,1}
    v_signed = 2 * v - 1
    
    # Apply the Hadamard transform
    y = H @ v_signed

    # Custom tie handling:
    # For y > 0, output 1; for y < 0, output 0; for y == 0, randomly choose 0 or 1.
    y_binary = np.empty_like(y, dtype=int)
    y_binary[y > 0] = 1
    y_binary[y < 0] = 0
    tie_indices = (y == 0)
    # Randomly assign 0 or 1 for tie values
    y_binary[tie_indices] = np.random.randint(0, 2, size=tie_indices.sum())
    
    return y_binary

def randomness_tests(n_samples=20000, vector_size=256, permute=True):
    """Run the Hadamard transform on many random inputs and collect outputs."""
    outputs = np.zeros((n_samples, vector_size), dtype=int)
    
    for i in tqdm(range(n_samples), desc="Processing samples"):
        v = np.random.randint(0, 2, vector_size)  # random {0,1} vector
        outputs[i] = binary_hadamard_transform(v, permute=permute)
    
    bit_means = outputs.mean(axis=0)      # per-bit average; expected ~0.5
    bit_variances = outputs.var(axis=0)     # per-bit variance; expected ~0.25
    return bit_means, bit_variances, outputs

# --------------------------
# Additional Randomness Tests
# --------------------------

def runs_test_for_sample(sample):
    """
    Perform a runs test on a binary sequence.
    Returns the z-score and two-sided p-value.
    """
    n = len(sample)
    n1 = np.sum(sample)
    n0 = n - n1
    # Count runs: one run for the first bit, plus one for each change
    runs = 1 + np.sum(sample[1:] != sample[:-1])
    # Expected number of runs
    mu = (2 * n1 * n0 / n) + 1
    # Variance of the number of runs
    sigma = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1)))
    if sigma == 0:
        return 0, 1.0  # extreme case: all bits are the same
    z = (runs - mu) / sigma
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided p-value
    return z, p_val

def autocorrelation(sequence, lag=1):
    """
    Compute lag-1 autocorrelation for a binary sequence.
    Returns the Pearson correlation coefficient between sequence[:-lag] and sequence[lag:].
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

# --------------------------
# Run Tests and Summarize Results
# --------------------------

# Run the base randomness tests
vector_size = 256
bit_means, bit_variances, outputs = randomness_tests(n_samples=20000, vector_size=vector_size, permute=True)

# Shannon entropy per bit position
def shannon_entropy(bits):
    p1 = np.mean(bits)
    p0 = 1 - p1
    # Avoid log2(0)
    return -p0 * np.log2(p0) - p1 * np.log2(p1) if p0 > 0 and p1 > 0 else 0

entropy_values = np.array([shannon_entropy(outputs[:, i]) for i in range(vector_size)])
avg_entropy = np.mean(entropy_values)  # Expected ~1

# Chi-square test for uniformity per bit
def chi_square_uniformity_test(bits):
    counts = np.bincount(bits.astype(int), minlength=2)
    expected = np.array([len(bits) / 2, len(bits) / 2])
    return stats.chisquare(counts, expected)

chi2_results = [chi_square_uniformity_test(outputs[:, i]) for i in range(vector_size)]
chi2_values, p_values = zip(*chi2_results)
avg_chi2 = np.mean(chi2_values)
avg_chi_p_value = np.mean(p_values)  # should be > 0.05

# Runs test across all samples
runs_p_values = np.array([runs_test_for_sample(sample)[1] for sample in outputs])
fraction_failing_runs = np.mean(runs_p_values < 0.05)  # expected ~0.05 (5%)

# Autocorrelation (lag-1) for all samples
autocorr_values = np.array([autocorrelation(sample, lag=1) for sample in outputs])
mean_autocorr = np.mean(autocorr_values)  # expected to be near 0

# --------------------------
# Summarize All Test Results
# --------------------------
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

# --------------------------
# Check Test Conditions
# --------------------------
passed = True

# Test for Bit Mean: expect near 0.5 (here, 0.49 - 0.51)
if not (0.49 < summary_results["Average Bit Mean"] < 0.51):
    print("[FAIL] Bit Mean is outside expected range (0.49 - 0.51)")
    passed = False

# Test for Bit Variance: expect near 0.25 (0.24 - 0.26)
if not (0.24 < summary_results["Average Bit Variance"] < 0.26):
    print("[FAIL] Bit Variance is outside expected range (0.24 - 0.26)")
    passed = False

# Test for Shannon Entropy: expect near 1 (0.98 - 1.02)
if not (0.98 < summary_results["Average Shannon Entropy"] < 1.02):
    print("[FAIL] Shannon Entropy is outside expected range (0.98 - 1.02)")
    passed = False

# Test for Chi-square p-value: expect > 0.05 (uniform distribution)
if not (summary_results["Average Chi-square p-value"] > 0.05):
    print("[FAIL] Chi-square p-value is too low (indicates non-uniform distribution)")
    passed = False

# Test for Runs Test: fraction failing should be near 5% (accept 3%-7%)
if not (0.03 < summary_results["Fraction Failing Runs Test"] < 0.07):
    print("[FAIL] Fraction failing runs test is outside expected range (0.03 - 0.07)")
    passed = False

# Test for Autocorrelation: mean absolute lag-1 autocorrelation should be very low (< 0.05)
if not (abs(summary_results["Mean Lag-1 Autocorrelation"]) < 0.05):
    print("[FAIL] Mean lag-1 autocorrelation is too high (indicates dependency between bits)")
    passed = False

if passed:
    print("\n[PASS] All randomness tests passed!")
else:
    print("\n[FAIL] Randomness tests failed.")

