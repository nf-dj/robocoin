import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

class HashAnalyzer:
    def __init__(self, hash_binary="./tens_hash"):
        self.hash_binary = hash_binary
        # Fixed seed for all hash operations
        self.fixed_seed = "f" * 64
    
    def get_hash(self, input_hex):
        """Run the hash program with fixed seed and input"""
        try:
            result = subprocess.run(
                [self.hash_binary, self.fixed_seed, input_hex],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running hash: {e}")
            return None

    def generate_sequential_inputs(self, start_input, count):
        """Generate sequential inputs by incrementing the hex value"""
        start = int(start_input, 16)
        for i in range(count):
            yield format(start + i, '064x')

    def hex_to_bits(self, hex_str):
        """Convert hex string to bit array"""
        return ''.join(bin(int(c, 16))[2:].zfill(4) for c in hex_str)

    def analyze_avalanche(self, base_input, num_flips=100):
        """Analyze avalanche effect by flipping single bits in input"""
        base_hash = self.get_hash(base_input)
        base_bits = self.hex_to_bits(base_hash)
        results = []
        
        input_bits = self.hex_to_bits(base_input)
        input_len = len(input_bits)
        
        for i in tqdm(range(min(num_flips, input_len)), desc="Analyzing avalanche"):
            # Flip one bit in the input
            modified_bits = list(input_bits)
            modified_bits[i] = '1' if modified_bits[i] == '0' else '0'
            modified_input = format(int(''.join(modified_bits), 2), '064x')
            
            # Get new hash
            modified_hash = self.get_hash(modified_input)
            modified_bits = self.hex_to_bits(modified_hash)
            
            # Count differing bits
            diff_count = sum(a != b for a, b in zip(base_bits, modified_bits))
            results.append(diff_count / len(base_bits))
            
        return results

    def run_statistical_tests(self, hashes, significance_level=0.05):
        """Run various statistical tests on hash outputs"""
        # Convert hashes to numbers and scale them to [0,1]
        numbers = [int(h, 16) / (2**256) for h in hashes]
        bits = ''.join(self.hex_to_bits(h) for h in hashes)
        
        results = {}
        
        # Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_p = stats.kstest(numbers, 'uniform', 
                                   args=(0, 1))
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'passed': ks_p > significance_level
        }
        
        # Chi-square test for bit distribution
        observed = [bits.count('0'), bits.count('1')]
        expected = [len(bits)/2, len(bits)/2]
        chi2_stat, chi2_p = stats.chisquare(observed, expected)
        results['chi2_test'] = {
            'statistic': chi2_stat,
            'p_value': chi2_p,
            'passed': chi2_p > significance_level
        }
        
        # Runs test for independence
        runs, n1, n2 = 1, bits.count('1'), bits.count('0')
        for i in range(len(bits)-1):
            if bits[i] != bits[i+1]:
                runs += 1
        runs_z = (runs - ((2*n1*n2)/(n1+n2) + 1)) / \
                 np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                        ((n1+n2)**2*(n1+n2-1)))
        runs_p = 2*(1 - stats.norm.cdf(abs(runs_z)))
        results['runs_test'] = {
            'statistic': runs_z,
            'p_value': runs_p,
            'passed': runs_p > significance_level
        }
        
        return results

    def analyze_byte_distribution(self, hashes):
        """Analyze distribution of bytes in hash outputs"""
        byte_counts = defaultdict(int)
        for h in hashes:
            # Split hash into bytes
            bytes_list = [h[i:i+2] for i in range(0, len(h), 2)]
            for byte in bytes_list:
                byte_counts[byte] = byte_counts[byte] + 1
        
        return dict(byte_counts)

    def plot_analysis(self, input_start="0"*64, num_hashes=1000):
        """Generate and plot various analysis visualizations"""
        # Generate hashes
        print(f"Generating {num_hashes} hashes...")
        inputs = list(self.generate_sequential_inputs(input_start, num_hashes))
        hashes = [self.get_hash(input_val) for input_val in tqdm(inputs)]
        hashes = [h for h in hashes if h]  # Remove any None values
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Byte distribution heatmap
        print("Analyzing byte distribution...")
        plt.subplot(2, 2, 1)
        byte_dist = self.analyze_byte_distribution(hashes)
        byte_matrix = np.zeros((16, 16))
        for byte, count in byte_dist.items():
            row = int(byte[0], 16)
            col = int(byte[1], 16)
            byte_matrix[row][col] = count
        sns.heatmap(byte_matrix, cmap='viridis')
        plt.title('Byte Distribution Heatmap')
        
        # 2. Avalanche effect
        print("Analyzing avalanche effect...")
        plt.subplot(2, 2, 2)
        avalanche_results = self.analyze_avalanche(input_start)
        plt.hist(avalanche_results, bins=20, edgecolor='black')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Ideal')
        plt.title('Avalanche Effect Distribution')
        plt.xlabel('Proportion of Bits Changed')
        plt.ylabel('Frequency')
        
        # 3. Sequential correlation
        print("Analyzing sequential correlation...")
        plt.subplot(2, 2, 3)
        hash_nums = [int(h, 16) / (2**256) for h in hashes]
        plt.scatter(hash_nums[:-1], hash_nums[1:], alpha=0.5, s=1)
        plt.title('Sequential Hash Correlation')
        plt.xlabel('Hash n')
        plt.ylabel('Hash n+1')
        
        # 4. Statistical test results
        print("Running statistical tests...")
        plt.subplot(2, 2, 4)
        stats_results = self.run_statistical_tests(hashes)
        test_names = list(stats_results.keys())
        p_values = [stats_results[test]['p_value'] for test in test_names]
        plt.bar(test_names, p_values)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level')
        plt.title('Statistical Test Results (p-values)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig, stats_results

def main():
    analyzer = HashAnalyzer()
    
    # Starting input (all zeros)
    input_start = "0" * 64
    
    # Run analysis with fixed seed
    print(f"Using fixed seed: {analyzer.fixed_seed}")
    fig, stats_results = analyzer.plot_analysis(input_start, num_hashes=1000)
    
    # Print statistical test results
    print("\nStatistical Test Results:")
    for test, results in stats_results.items():
        print(f"\n{test}:")
        print(f"  Statistic: {results['statistic']:.4f}")
        print(f"  P-value: {results['p_value']:.4f}")
        print(f"  Passed: {results['passed']}")
    
    plt.show()

if __name__ == "__main__":
    main()
