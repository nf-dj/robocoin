import random
import numpy as np
import math
import sys

###############################################################################
# 1. Random Invertible Ternary Matrix Generation using NumPy
###############################################################################
def random_invertible_ternary_matrix(n, fill_prob=0.4, max_tries=500):
    """
    Generates an n x n matrix with entries in {-1, 0, +1} using NumPy.
    Each entry is 0 with probability (1 - fill_prob) and ±1 with probability fill_prob/2.
    Returns an invertible matrix (i.e. full rank) or None after max_tries.
    """
    for _ in range(max_tries):
        # Generate a random matrix of floats in [0,1)
        R = np.random.random((n, n))
        # Create an array of random choices from {1, -1}
        choices = np.random.choice([1, -1], size=(n, n))
        # Where R < fill_prob, choose the random sign; otherwise, 0.
        M = np.where(R < fill_prob, choices, 0)
        if np.linalg.matrix_rank(M) == n:
            return M
    return None

###############################################################################
# 2. Bias Computation
###############################################################################
def compute_bias(M):
    """
    Computes a bias vector b from matrix M such that:
      b[i] = -0.5 * (sum of row i)
    Returns a NumPy array of shape (n,).
    """
    return -0.5 * np.sum(M, axis=1)

###############################################################################
# 3. Matrix-Vector Multiplication with Bias
###############################################################################
def multiply_with_bias(M, v, bias):
    """
    Given a NumPy matrix M (n x n), binary vector v (shape (n,)),
    and a bias vector (shape (n,)), returns r = M·v + bias.
    """
    return M.dot(v) + bias

###############################################################################
# 4. Symmetric Threshold Function
###############################################################################
def threshold_with_symmetry(x):
    """
    Vectorized function: for each x, return 1 if x > 0, 0 if x < 0,
    and if x == 0, randomly choose 0 or 1.
    (In practice, with continuous noise, exactly 0 will be rare.)
    """
    # x is a NumPy array
    out = np.where(x > 0, 1, 0)
    # Find elements exactly equal to 0 (rare) and assign randomly.
    zero_mask = (x == 0)
    if np.any(zero_mask):
        random_bits = np.random.randint(0, 2, size=np.sum(zero_mask))
        out[zero_mask] = random_bits
    return out

###############################################################################
# 5. Noise Addition and Thresholding (Conversion to Binary)
###############################################################################
def add_noise_and_threshold(values, noise_amp=1.0):
    """
    Given a NumPy array 'values', add independent noise drawn uniformly from
    [-noise_amp, noise_amp] and then apply symmetric thresholding:
      output bit = 1 if (value + noise) > 0, 0 if < 0, and random if exactly 0.
    Returns a NumPy array of binary bits.
    """
    noise = np.random.uniform(-noise_amp, noise_amp, size=values.shape)
    noisy_values = values + noise
    return threshold_with_symmetry(noisy_values)

###############################################################################
# 6. Statistical Tests (using Python functions)
###############################################################################
def monobit_test(bits):
    n = len(bits)
    count_ones = np.sum(bits)
    proportion = count_ones / n
    expected = 0.5 * n
    variance = n * 0.25
    z = (count_ones - expected) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": "Monobit Frequency", "proportion_ones": proportion, "z_value": z, "p_value": p_value}

def runs_test(bits):
    n = len(bits)
    if n < 2:
        return {"test": "Runs", "p_value": None, "note": "Not enough bits"}
    pi = np.sum(bits) / n
    if pi == 0 or pi == 1:
        return {"test": "Runs", "p_value": 0.0, "note": "All bits identical"}
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    mean_runs = 1 + 2 * n * pi * (1 - pi)
    var_runs = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1) / (n - 1)
    z = (runs - mean_runs) / math.sqrt(var_runs)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": "Runs", "num_runs": runs, "z_value": z, "p_value": p_value}

def block_frequency_test(bits, block_size=128):
    n = len(bits)
    num_blocks = n // block_size
    chi_sq = 0.0
    idx = 0
    for _ in range(num_blocks):
        block = bits[idx: idx + block_size]
        idx += block_size
        count_ones = np.sum(block)
        diff = count_ones - (block_size / 2)
        chi_sq += (4.0 * diff * diff) / block_size
    p_value = chi_square_survival(chi_sq, df=num_blocks)
    return {"test": "Block Frequency", "block_size": block_size, "num_blocks": num_blocks, "chi_square": chi_sq, "p_value": p_value}

def serial_test(bits):
    n = len(bits)
    count_00 = count_01 = count_10 = count_11 = 0
    for i in range(n - 1):
        pair = (bits[i], bits[i+1])
        if pair == (0,0):
            count_00 += 1
        elif pair == (0,1):
            count_01 += 1
        elif pair == (1,0):
            count_10 += 1
        elif pair == (1,1):
            count_11 += 1
    exp = (n - 1) / 4
    chi_sq = ((count_00 - exp)**2 + (count_01 - exp)**2 +
              (count_10 - exp)**2 + (count_11 - exp)**2) / exp
    p_value = chi_square_survival(chi_sq, df=3)
    return {"test": "Serial 2-bit", "counts": (count_00, count_01, count_10, count_11), "chi_square": chi_sq, "p_value": p_value}

def cusum_test(bits):
    N = len(bits)
    xs = [2 * b - 1 for b in bits]
    cumulative = 0
    S = []
    for x in xs:
        cumulative += x
        S.append(cumulative)
    M = max(abs(s) for s in S)
    start = int(math.ceil(M / math.sqrt(N)))
    sum_term = 0.0
    for k in range(start, 101):
        arg1 = ((4 * k + 1) * M) / math.sqrt(N)
        arg2 = ((4 * k - 1) * M) / math.sqrt(N)
        term = math.erfc(arg1 / math.sqrt(2)) - math.erfc(arg2 / math.sqrt(2))
        sum_term += term
    p_value = 1 - sum_term
    return {"test": "Cumulative Sums", "M": M, "p_value": p_value}

def autocorrelation_test(bits, lag=1):
    N = len(bits)
    xs = [2 * b - 1 for b in bits]
    S = sum(xs[i] * xs[i+lag] for i in range(N - lag))
    z = S / math.sqrt(N - lag)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": f"Autocorrelation (lag={lag})", "sum": S, "z_value": z, "p_value": p_value}

def chi_square_survival(x, df):
    mean = df
    std = math.sqrt(2 * df) if df > 0 else 0
    if std == 0:
        return None
    z = (x - mean) / std
    if z > 10:
        return 0.0
    if z < -10:
        return 1.0
    try:
        return regularized_gamma_upper(df / 2, x / 2)
    except OverflowError:
        return 0.0 if x > df else 1.0

def regularized_gamma_upper(s, z):
    if z < s + 1:
        return 1.0 - regularized_gamma_lower(s, z)
    else:
        val = gamma_inc_cf(s, z)
        return val / math.gamma(s)

def regularized_gamma_lower(s, z, max_iter=100, eps=1e-12):
    gln = math.lgamma(s)
    sum_term = 1.0 / s
    term = 1.0 / s
    for n in range(1, max_iter + 1):
        term *= z / (s + n)
        sum_term += term
        if abs(term) < eps * abs(sum_term):
            break
    return math.exp(-z + s * math.log(z) - gln) * sum_term

def gamma_inc_cf(a, x, max_iter=100, eps=1e-12):
    gln = math.lgamma(a)
    b0 = 0.0
    b1 = 1.0
    a0 = 1.0
    a1 = x
    for n in range(1, max_iter + 1):
        an = float(n)
        ana = an - a
        a0 = a1 + a0 * ana
        b0 = b1 + b0 * ana
        if abs(a0) < 1e-12:
            a0 = 1e-12
        if abs(b0) < 1e-12:
            b0 = 1e-12
        a1 = a0 * x
        b1 = b0 * x
        fac = 1.0 / a1 if a1 != 0 else 1.0
        a0 *= fac
        a1 *= fac
        b0 *= fac
        b1 *= fac
        if abs(1.0 - b1) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * a1

###############################################################################
# 8. Main: Generate Bitstream and Run Tests
###############################################################################
def main():
    n = 256
    print("Generating a random invertible 256x256 ternary matrix using numpy...")
    M = random_invertible_ternary_matrix(n=n, fill_prob=0.4, max_tries=500)
    if M is None:
        print("Failed to find an invertible matrix after 500 tries.")
        return
    print("Matrix found.")
    
    # Compute bias vector: b = -0.5 * sum(M, axis=1)
    bias = compute_bias(M)
    print("Bias vector computed.")
    
    print("Now generating bitstream using numpy for matmult, bias, scaling, noise addition, and symmetric thresholding.")
    scale_factor = 2.0  # Increase the dynamic range.
    num_vectors = 20000  # Each vector produces 256 bits.
    noise_amp = 1.0     # Adjust noise amplitude.
    
    all_bits = []
    progress_interval = max(1, num_vectors // 20)
    
    for i in range(num_vectors):
        # Generate a random 256-bit input vector (binary vector)
        v = np.random.randint(0, 2, size=n)
        # Multiply M by v and add bias using numpy.
        int_result = M.dot(v) + bias
        # Scale the result.
        scaled_result = int_result * scale_factor
        # Add noise and apply symmetric thresholding.
        final_bits = add_noise_and_threshold(scaled_result, noise_amp)
        all_bits.extend(final_bits.tolist())
        
        if (i+1) % progress_interval == 0:
            pct = 100.0 * (i+1) / num_vectors
            print(f"Processed {i+1} of {num_vectors} vectors ({pct:.1f}%)", file=sys.stderr)
    
    print(f"\nDone generating {len(all_bits)} bits total.\n")
    
    # Run statistical tests.
    results = []
    results.append(monobit_test(np.array(all_bits)))
    results.append(runs_test(np.array(all_bits)))
    results.append(block_frequency_test(np.array(all_bits), block_size=128))
    results.append(serial_test(np.array(all_bits)))
    results.append(cusum_test(all_bits))
    results.append(autocorrelation_test(all_bits, lag=1))
    
    alpha = 0.01
    print("=== BASIC STATISTICAL TEST RESULTS ===")
    for r in results:
        test_name = r.get("test", "Unknown")
        p = r.get("p_value", None)
        print(f"\nTest: {test_name}")
        for k, v in r.items():
            if k == "test":
                continue
            print(f"  {k} = {v}")
        if p is None:
            print("  => Cannot determine pass/fail (no p-value).")
        else:
            if p < alpha:
                print(f"  => FAIL (p_value={p:.4g} < {alpha})")
            else:
                print(f"  => PASS (p_value={p:.4g} >= {alpha})")

if __name__ == "__main__":
    main()

