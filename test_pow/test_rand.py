import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def generate_binary_vectors(n_vectors=1000, vector_size=256):
    """Generate random binary vectors."""
    return np.random.randint(0, 2, size=(n_vectors, vector_size))

def generate_ternary_matrix(input_size=256, output_size=256):
    """
    Generate a random ternary matrix with values -1, 0, and 1,
    where each row is balanced: exactly 32 entries of +1 and 32 entries of -1,
    and the rest are 0.

    Parameters:
      input_size (int): Number of rows in the matrix.
      output_size (int): Number of columns in the matrix (must be >= 64).

    Returns:
      np.ndarray: A matrix of shape (input_size, output_size) where each row
                  has 32 +1's, 32 -1's, and (output_size - 64) zeros.
    """
    if output_size < 64:
        raise ValueError("output_size must be at least 64 to have 32 +1's and 32 -1's per row")

    A = np.zeros((input_size, output_size), dtype=int)
    d = 64  # total nonzero entries per row (32 +1's and 32 -1's)
    pos_count = d // 2  # 32
    neg_count = d // 2  # 32

    for i in range(input_size):
        # Randomly choose d distinct column indices for nonzero entries.
        indices = np.random.choice(output_size, d, replace=False)
        # Create an array with 32 +1's and 32 -1's.
        values = np.array([1] * pos_count + [-1] * neg_count)
        # Shuffle the values so that the positions of +1's and -1's are random.
        np.random.shuffle(values)
        A[i, indices] = values

    return A

def apply_matrix_and_threshold(binary_vectors, ternary_matrix):
    """
    Multiply binary vectors by a ternary matrix, add a bias term and a random noise matrix,
    then use a simple threshold (x > 0 → 1, else 0).

    Steps:
      1. Compute the matrix product.
      2. Compute the bias vector as: bias = -0.5 * sum(ternary_matrix, axis=0) and add it.
      3. Add a random noise matrix. (Here we use noise drawn from {0, 1} per element.)
      4. Multiply the result by 2 (to amplify the signal) and add the noise.
      5. Apply thresholding: if result > 0 → 1; otherwise → 0.
    
    Parameters:
      binary_vectors (np.ndarray): Array of shape (m, n) with binary values (0 or 1).
      ternary_matrix (np.ndarray): Array of shape (n, p) with values -1, 0, or 1.

    Returns:
      np.ndarray: Binary output vectors after thresholding.
    """
    # Step 1: Matrix multiplication
    result = np.matmul(binary_vectors, ternary_matrix)

    # Step 2: Add bias computed from the ternary matrix
    bias = -0.5 * np.sum(ternary_matrix, axis=0)
    print("Bias vector:", bias)
    print("Bias shape:", bias.shape)
    result = result + bias

    # Step 3: Add a random noise matrix.
    # Here we generate noise with values 0 or 1, with a unique noise vector per row.
    noise = np.random.choice([0, 1], size=result.shape)
    print("Noise matrix shape:", noise.shape)
    
    # Step 4: Amplify the result and add the noise.
    result = result * 2
    result = result + noise

    # Step 5: Simple thresholding: if result > 0 then 1, else 0.
    output = np.where(result > 0, 1, 0)

    return output

def analyze_binary_randomness(vectors):
    """Perform statistical tests on binary vectors."""
    results = {}
    test_results = {}
    
    # Basic statistics
    results['overall_mean'] = np.mean(vectors)
    results['std_dev'] = np.std(vectors)
    
    # Distribution of ones and zeros (ratio of ones to vector length)
    counts = np.sum(vectors, axis=1)
    results['zero_one_ratio'] = np.mean(counts) / vectors.shape[1]
    
    # Chi-square test for uniformity
    observed_counts = np.array([np.sum(vectors == 0), np.sum(vectors == 1)])
    expected_counts = np.array([vectors.size / 2, vectors.size / 2])
    chi2_stat, chi2_p = stats.chisquare(observed_counts, expected_counts)
    results['chi2_p_value'] = chi2_p
    # Save chi-square details
    results['chi2_details'] = {
        'observed_counts': observed_counts,
        'expected_counts': expected_counts,
        'chi2_statistic': chi2_stat
    }
    
    # Runs test: count the number of runs in each vector.
    runs_test_results = []
    for vector in vectors:
        runs = np.diff(vector).astype(bool).sum() + 1
        runs_test_results.append(runs)
    results['avg_runs'] = np.mean(runs_test_results)
    
    # Autocorrelation: compute the first 4 lags.
    autocorr_results = []
    for vector in vectors:
        autocorr = np.correlate(vector - np.mean(vector), vector - np.mean(vector), mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr_results.append(autocorr[1:5] / autocorr[0])
    results['mean_autocorr'] = np.mean(autocorr_results, axis=0)
    
    # Entropy calculation: for binary sequences, maximum entropy is 1.0.
    entropies = []
    for vector in vectors:
        unique, counts = np.unique(vector, return_counts=True)
        freqs = counts / vector.size
        entropy = -np.sum(freqs * np.log2(freqs))
        entropies.append(entropy)
    results['avg_entropy'] = np.mean(entropies)
    
    # Serial correlation (lag-1)
    serial_corrs = []
    for vector in vectors:
        if vector.size > 1:
            corr = np.corrcoef(vector[:-1], vector[1:])[0, 1]
            serial_corrs.append(corr)
    results['avg_serial_corr'] = np.mean(serial_corrs)
    
    # Additional randomness check: Block Frequency Test.
    # Divide each vector into blocks of size 32 and compute the std deviation of block frequencies.
    block_size = 32
    block_freqs = []
    for vector in vectors:
        num_blocks = vector.size // block_size
        # Reshape vector to have num_blocks rows, each of length block_size.
        blocks = vector[:num_blocks * block_size].reshape(num_blocks, block_size)
        freqs = np.mean(blocks, axis=1)
        block_freqs.append(freqs)
    block_freqs = np.array(block_freqs)
    std_block_freq = np.mean(np.std(block_freqs, axis=1))
    results['avg_block_freq_std'] = std_block_freq
    expected_std = 0.5 / np.sqrt(block_size)  # Theoretical std for block frequency with p=0.5.
    
    test_results['block_frequency_test'] = {
        'pass': abs(std_block_freq - expected_std) < 0.02,
        'criterion': f'block frequency std close to {expected_std:.4f} (±0.02)',
        'value': std_block_freq
    }
    
    # Define test criteria and evaluate pass/fail for the other tests.
    test_results['mean_test'] = {
        'pass': 0.45 <= results['overall_mean'] <= 0.55,
        'criterion': '0.45 <= mean <= 0.55',
        'value': results['overall_mean']
    }
    
    test_results['zero_one_ratio_test'] = {
        'pass': 0.45 <= results['zero_one_ratio'] <= 0.55,
        'criterion': '0.45 <= ratio <= 0.55',
        'value': results['zero_one_ratio']
    }
    
    test_results['chi2_test'] = {
        'pass': results['chi2_p_value'] >= 0.05,
        'criterion': 'p-value >= 0.05',
        'value': results['chi2_p_value'],
        'observed_counts': results['chi2_details']['observed_counts'],
        'expected_counts': results['chi2_details']['expected_counts'],
        'chi2_statistic': results['chi2_details']['chi2_statistic']
    }
    
    expected_runs = vectors.shape[1] / 2 + 1
    test_results['runs_test'] = {
        'pass': 0.9 * expected_runs <= results['avg_runs'] <= 1.1 * expected_runs,
        'criterion': f'within ±10% of expected runs ({expected_runs:.1f})',
        'value': results['avg_runs']
    }
    
    max_autocorr = np.max(np.abs(results['mean_autocorr']))
    test_results['autocorr_test'] = {
        'pass': max_autocorr < 0.1,
        'criterion': 'max autocorrelation < 0.1',
        'value': max_autocorr
    }
    
    test_results['entropy_test'] = {
        'pass': 0.99 <= results['avg_entropy'] <= 1.01,  # expecting near maximal entropy for binary.
        'criterion': 'average entropy close to 1.0',
        'value': results['avg_entropy']
    }
    
    test_results['serial_corr_test'] = {
        'pass': abs(results['avg_serial_corr']) < 0.05,
        'criterion': 'absolute lag-1 serial correlation < 0.05',
        'value': results['avg_serial_corr']
    }
    
    results['test_results'] = test_results
    return results

def plot_analysis(vectors, results, vector_type='binary'):
    """Create visualizations of the analysis results."""
    plt.figure(figsize=(15, 10))
    
    # Distribution histogram
    plt.subplot(2, 2, 1)
    unique, counts = np.unique(vectors, return_counts=True)
    plt.bar(unique, counts / len(vectors), tick_label=[str(x) for x in unique])
    plt.title(f'Distribution of Values ({vector_type})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Heatmap of first 50 vectors
    plt.subplot(2, 2, 2)
    plt.imshow(vectors[:50], aspect='auto', cmap='binary')
    plt.title(f'First 50 Vectors Visualization ({vector_type})')
    plt.xlabel('Position')
    plt.ylabel('Vector Index')
    
    # Average autocorrelation
    plt.subplot(2, 2, 3)
    plt.plot(range(1, 5), results['mean_autocorr'], 'o-')
    plt.title('Average Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    
    # QQ plot
    plt.subplot(2, 2, 4)
    stats.probplot(vectors.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

def print_test_results(results, vector_type):
    """Print formatted test results."""
    print(f"\n{vector_type.upper()} Vector Analysis Results:")
    print("=" * 50)
    
    test_results = results['test_results']
    all_tests_passed = all(test['pass'] for test in test_results.values())
    
    # Print overall status
    status = "✓ ALL TESTS PASSED" if all_tests_passed else "✗ SOME TESTS FAILED"
    print(f"\n{status}\n")
    print("-" * 50)
    
    # Print individual test results
    for test_name, test_data in test_results.items():
        result_symbol = "✓" if test_data['pass'] else "✗"
        print(f"\n{test_name.upper()}: {result_symbol}")
        print(f"Criterion: {test_data['criterion']}")
        print(f"Value: {test_data['value']:.4f}")
        # For chi-square test, print additional details.
        if test_name == 'chi2_test':
            print(f"Observed counts: {test_data['observed_counts']}")
            print(f"Expected counts: {test_data['expected_counts']}")
            print(f"Chi-square statistic: {test_data['chi2_statistic']:.4f}")
    
    print("\nDetailed Statistics:")
    print("-" * 50)
    print(f"Standard deviation: {results['std_dev']:.4f}")
    print("\nMean autocorrelation for first 4 lags:")
    for i, corr in enumerate(results['mean_autocorr'], 1):
        print(f"Lag {i}: {corr:.4f}")

def main():
    # Generate vectors and matrix
    binary_vectors = generate_binary_vectors()
    ternary_matrix = generate_ternary_matrix()
    
    # Apply matrix multiplication and threshold
    result_vectors = apply_matrix_and_threshold(binary_vectors, ternary_matrix)
    
    # Analyze original and result vectors
    input_results = analyze_binary_randomness(binary_vectors)
    result_results = analyze_binary_randomness(result_vectors)
    
    # Print results
    print_test_results(input_results, "Input Binary")
    print_test_results(result_results, "Result Binary")
    
    # Create visualizations
    plot_analysis(binary_vectors, input_results, "Input Binary")
    plot_analysis(result_vectors, result_results, "Result Binary")

if __name__ == "__main__":
    main()

