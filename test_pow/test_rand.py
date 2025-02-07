import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def generate_binary_vectors(n_vectors=1000, vector_size=256):
    """Generate random binary vectors."""
    return np.random.randint(0, 2, size=(n_vectors, vector_size))

def apply_threshold(input_vectors, noise_vectors):
    """
    1. Sum binary vectors (gives 0,1,2)
    2. Subtract 1 to center around 0 (gives -1,0,1)
    3. Apply ReLU with random choice for 0:
       - value > 0  -> 1
       - value < 0  -> 0
       - value = 0  -> random binary
    """
    # Sum and center around 0
    centered = input_vectors + noise_vectors - 1
    
    # Initialize result array
    result = np.zeros_like(input_vectors)
    
    # Apply ReLU rules
    result[centered > 0] = 1
    result[centered < 0] = 0
    
    # For zeros, use random binary values
    zero_mask = (centered == 0)
    result[zero_mask] = np.random.randint(0, 2, size=np.sum(zero_mask))
    
    return result

def analyze_binary_randomness(vectors):
    """Perform statistical tests on binary vectors."""
    results = {}
    test_results = {}
    
    # Basic statistics
    mean_values = np.mean(vectors, axis=1)
    results['overall_mean'] = np.mean(vectors)
    results['std_dev'] = np.std(vectors)
    
    # Distribution of ones and zeros
    counts = np.sum(vectors, axis=1)
    results['zero_one_ratio'] = np.mean(counts) / vectors.shape[1]
    
    # Chi-square test for uniformity
    observed_counts = np.array([np.sum(vectors == 0), np.sum(vectors == 1)])
    expected_counts = np.array([vectors.size/2, vectors.size/2])
    chi2_stat, chi2_p = stats.chisquare(observed_counts, expected_counts)
    results['chi2_p_value'] = chi2_p
    
    # Runs test
    runs_test_results = []
    for vector in vectors:
        runs = np.diff(vector).astype(bool).sum() + 1
        runs_test_results.append(runs)
    results['avg_runs'] = np.mean(runs_test_results)
    
    # Autocorrelation
    autocorr_results = []
    for vector in vectors:
        autocorr = np.correlate(vector - np.mean(vector), 
                              vector - np.mean(vector), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_results.append(autocorr[1:5] / autocorr[0])
    results['mean_autocorr'] = np.mean(autocorr_results, axis=0)
    
    # Define test criteria and evaluate pass/fail
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
        'value': results['chi2_p_value']
    }
    
    expected_runs = vectors.shape[1]/2 + 1
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
    
    results['test_results'] = test_results
    return results

def plot_analysis(vectors, results, vector_type='binary'):
    """Create visualizations of the analysis results."""
    plt.figure(figsize=(15, 10))
    
    # Distribution histogram
    plt.subplot(2, 2, 1)
    unique, counts = np.unique(vectors, return_counts=True)
    plt.bar(unique, counts/len(vectors), 
           tick_label=[str(x) for x in unique])
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
    
    print("\nDetailed Statistics:")
    print("-" * 50)
    print(f"Standard deviation: {results['std_dev']:.4f}")
    print("\nMean autocorrelation for first 4 lags:")
    for i, corr in enumerate(results['mean_autocorr'], 1):
        print(f"Lag {i}: {corr:.4f}")

def main():
    # Generate vectors
    input_vectors = generate_binary_vectors()
    noise_vectors = generate_binary_vectors()
    
    # Apply threshold to get result
    result_vectors = apply_threshold(input_vectors, noise_vectors)
    
    # Analyze original and result vectors
    input_results = analyze_binary_randomness(input_vectors)
    result_results = analyze_binary_randomness(result_vectors)
    
    # Print results
    print_test_results(input_results, "Input Binary")
    print_test_results(result_results, "Result Binary")
    
    # Create visualizations
    plot_analysis(input_vectors, input_results, "Input Binary")
    plot_analysis(result_vectors, result_results, "Result Binary")

if __name__ == "__main__":
    main()
