import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def generate_binary_vectors(n_vectors=1000, vector_size=256):
    """Generate random binary vectors."""
    return np.random.randint(0, 2, size=(n_vectors, vector_size))

def analyze_randomness(vectors):
    """Perform statistical tests on the generated vectors."""
    results = {}
    test_results = {}
    
    # 1. Basic statistics
    mean_values = np.mean(vectors, axis=1)
    results['overall_mean'] = np.mean(vectors)
    results['std_dev'] = np.std(vectors)
    
    # 2. Distribution of ones and zeros
    counts = np.sum(vectors, axis=1)
    results['zero_one_ratio'] = np.mean(counts) / vectors.shape[1]
    
    # 3. Chi-square test for uniformity
    observed_counts = np.array([np.sum(vectors == 0), np.sum(vectors == 1)])
    expected_counts = np.array([vectors.size/2, vectors.size/2])
    chi2_stat, chi2_p = stats.chisquare(observed_counts, expected_counts)
    results['chi2_p_value'] = chi2_p
    
    # 4. Runs test (for independence)
    runs_test_results = []
    for vector in vectors:
        runs = np.diff(vector).astype(bool).sum() + 1
        runs_test_results.append(runs)
    results['avg_runs'] = np.mean(runs_test_results)
    
    # 5. Autocorrelation test
    autocorr_results = []
    for vector in vectors:
        autocorr = np.correlate(vector - np.mean(vector), 
                              vector - np.mean(vector), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_results.append(autocorr[1:5] / autocorr[0])  # First 4 lags
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
    
    # Expected number of runs for a random sequence
    expected_runs = vectors.shape[1]/2 + 1
    test_results['runs_test'] = {
        'pass': 0.9 * expected_runs <= results['avg_runs'] <= 1.1 * expected_runs,
        'criterion': f'within ±10% of expected runs ({expected_runs:.1f})',
        'value': results['avg_runs']
    }
    
    # Autocorrelation should be close to 0 for random sequences
    max_autocorr = np.max(np.abs(results['mean_autocorr']))
    test_results['autocorr_test'] = {
        'pass': max_autocorr < 0.1,
        'criterion': 'max autocorrelation < 0.1',
        'value': max_autocorr
    }
    
    results['test_results'] = test_results
    return results

def plot_analysis(vectors, results):
    """Create visualizations of the analysis results."""
    plt.figure(figsize=(15, 10))
    
    # 1. Distribution of ones in vectors
    plt.subplot(2, 2, 1)
    counts = np.sum(vectors, axis=1)
    plt.hist(counts, bins=30, density=True)
    plt.title('Distribution of Ones per Vector')
    plt.xlabel('Number of Ones')
    plt.ylabel('Density')
    
    # 2. Heatmap of first 50 vectors
    plt.subplot(2, 2, 2)
    plt.imshow(vectors[:50], aspect='auto', cmap='binary')
    plt.title('First 50 Vectors Visualization')
    plt.xlabel('Bit Position')
    plt.ylabel('Vector Index')
    
    # 3. Average autocorrelation
    plt.subplot(2, 2, 3)
    plt.plot(range(1, 5), results['mean_autocorr'], 'o-')
    plt.title('Average Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    
    # 4. Runs distribution
    plt.subplot(2, 2, 4)
    runs = [np.diff(vector).astype(bool).sum() + 1 for vector in vectors]
    plt.hist(runs, bins=30, density=True)
    plt.title('Distribution of Runs')
    plt.xlabel('Number of Runs')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate vectors
    vectors = generate_binary_vectors()
    
    # Analyze randomness
    results = analyze_randomness(vectors)
    
    # Print results with pass/fail indicators
    print("\nRandomness Analysis Results:")
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
    
    # Create visualizations
    plot_analysis(vectors, results)

if __name__ == "__main__":
    main()
