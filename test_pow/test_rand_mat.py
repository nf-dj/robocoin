import random
import numpy as np
import math
import sys

###############################################################################
# 1. Random Invertible Ternary Matrix Generation
###############################################################################
def random_ternary_matrix(n, fill_prob=0.4):
    """
    Generates an n x n matrix with entries in {-1, 0, +1}.
    Each entry is 0 with probability (1 - fill_prob) and ±1 with probability fill_prob/2.
    """
    M = []
    for _ in range(n):
        row = []
        for _ in range(n):
            if random.random() < fill_prob:
                row.append(random.choice([1, -1]))
            else:
                row.append(0)
        M.append(row)
    return M

def is_invertible(matrix):
    arr = np.array(matrix, dtype=float)
    return np.linalg.matrix_rank(arr) == arr.shape[0]

def random_invertible_ternary_matrix(n=256, fill_prob=0.4, max_tries=500):
    """
    Attempts to generate an invertible n x n ternary matrix within max_tries.
    """
    for _ in range(max_tries):
        M = random_ternary_matrix(n, fill_prob)
        if is_invertible(M):
            return M
    return None

###############################################################################
# 2. Bias Computation
###############################################################################
def compute_bias(M):
    """
    Computes a bias vector b for matrix M such that for each row i:
      b[i] = -0.5 * (sum of row i)
    (Because E[v_j]=0.5 for random binary v.)
    """
    bias = []
    for row in M:
        s = sum(row)
        bias.append(-0.5 * s)
    return bias

###############################################################################
# 3. Matrix-Vector Multiplication with Bias
###############################################################################
def multiply_int_with_bias(M, v, bias):
    """
    Multiplies matrix M (n x n) by binary vector v (length n) using integer arithmetic,
    and then adds the corresponding bias.
    Returns a list of real numbers (length n).
    """
    n = len(M)
    result = [0] * n
    for i in range(n):
        s = 0
        for j in range(n):
            s += M[i][j] * v[j]
        result[i] = s + bias[i]
    return result

###############################################################################
# 4. Symmetric Threshold Function
###############################################################################
def threshold_with_symmetry(x):
    """
    Returns 1 if x > 0, 0 if x < 0, and if x == 0, returns 0 or 1 at random.
    """
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return random.choice([0, 1])

###############################################################################
# 5. Noise Addition and Thresholding (Conversion to Binary)
###############################################################################
def add_noise_and_threshold(values, noise_amp=1.0):
    """
    For each value in 'values', adds noise uniformly drawn from [-noise_amp, noise_amp]
    and then applies symmetric thresholding.
    Returns a list of binary bits.
    """
    out_bits = []
    for x in values:
        noise = random.uniform(-noise_amp, noise_amp)
        val = x + noise
        out_bits.append(threshold_with_symmetry(val))
    return out_bits

###############################################################################
# 6. Statistical Tests
###############################################################################
def monobit_test(bits):
    n = len(bits)
    count_ones = sum(bits)
    proportion = count_ones / n
    expected = 0.5 * n
    variance = n * 0.25
    if variance <= 0:
        return {"test": "Monobit Frequency", "p_value": None, "note": "Zero variance"}
    z = (count_ones - expected) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": "Monobit Frequency", "proportion_ones": proportion, "z_value": z, "p_value": p_value}

def runs_test(bits):
    n = len(bits)
    if n < 2:
        return {"test": "Runs", "p_value": None, "note": "Not enough bits"}
    pi = sum(bits) / n
    if pi == 0 or pi == 1:
        return {"test": "Runs", "p_value": 0.0, "note": "All bits identical"}
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1
    mean_runs = 1 + 2 * n * pi * (1 - pi)
    var_runs = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1) / (n - 1)
    if var_runs <= 0:
        return {"test": "Runs", "p_value": None, "note": "Variance <= 0"}
    z = (runs - mean_runs) / math.sqrt(var_runs)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": "Runs", "num_runs": runs, "z_value": z, "p_value": p_value}

def block_frequency_test(bits, block_size=128):
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks < 1:
        return {"test": "Block Frequency", "p_value": None, "note": "Not enough bits for block_size"}
    chi_sq = 0.0
    idx = 0
    for _ in range(num_blocks):
        block = bits[idx: idx + block_size]
        idx += block_size
        count_ones = sum(block)
        diff = count_ones - (block_size / 2)
        chi_sq += (4.0 * diff * diff) / block_size
    p_value = chi_square_survival(chi_sq, df=num_blocks)
    return {"test": "Block Frequency", "block_size": block_size, "num_blocks": num_blocks, "chi_square": chi_sq, "p_value": p_value}

def serial_test(bits):
    n = len(bits)
    if n < 2:
        return {"test": "Serial 2-bit", "p_value": None, "note": "Not enough bits"}
    count_00 = count_01 = count_10 = count_11 = 0
    for i in range(n - 1):
        pair = (bits[i], bits[i + 1])
        if pair == (0, 0):
            count_00 += 1
        elif pair == (0, 1):
            count_01 += 1
        elif pair == (1, 0):
            count_10 += 1
        elif pair == (1, 1):
            count_11 += 1
    exp = (n - 1) / 4
    chi_sq = ((count_00 - exp)**2 + (count_01 - exp)**2 +
              (count_10 - exp)**2 + (count_11 - exp)**2) / exp
    p_value = chi_square_survival(chi_sq, df=3)
    return {"test": "Serial 2-bit", "counts": (count_00, count_01, count_10, count_11), "chi_square": chi_sq, "p_value": p_value}

###############################################################################
# 6a. Cumulative Sums Test (Cusum)
###############################################################################
def cusum_test(bits):
    """
    Converts bits (0,1) to ±1 and computes the cumulative sum.
    Let M be the maximum absolute cumulative sum.
    The p-value is approximated by summing terms as in the NIST Cusum test.
    (This is a simplified version.)
    """
    N = len(bits)
    xs = [2 * b - 1 for b in bits]
    cumulative = 0
    S = []
    for x in xs:
        cumulative += x
        S.append(cumulative)
    M = max(abs(s) for s in S)
    # Sum over k from ceil(M/sqrt(N)) to a fixed upper bound (e.g., 100)
    start = int(math.ceil(M / math.sqrt(N)))
    sum_term = 0.0
    for k in range(start, 101):
        arg1 = ((4 * k + 1) * M) / math.sqrt(N)
        arg2 = ((4 * k - 1) * M) / math.sqrt(N)
        term = math.erfc(arg1 / math.sqrt(2)) - math.erfc(arg2 / math.sqrt(2))
        sum_term += term
    p_value = 1 - sum_term
    return {"test": "Cumulative Sums", "M": M, "p_value": p_value}

###############################################################################
# 6b. Autocorrelation Test (lag = 1)
###############################################################################
def autocorrelation_test(bits, lag=1):
    """
    Converts bits (0,1) to ±1 and computes the autocorrelation at the given lag.
    Returns a z-value and p-value.
    """
    N = len(bits)
    xs = [2 * b - 1 for b in bits]
    S = sum(xs[i] * xs[i + lag] for i in range(N - lag))
    z = S / math.sqrt(N - lag)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"test": f"Autocorrelation (lag={lag})", "sum": S, "z_value": z, "p_value": p_value}

###############################################################################
# 7. Chi-Square Survival Function (with overflow safeguards)
###############################################################################
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
    print("Generating a random invertible 256x256 ternary matrix...")
    M = random_invertible_ternary_matrix(n=n, fill_prob=0.4, max_tries=500)
    if M is None:
        print("Failed to find an invertible matrix after 500 tries.")
        return
    print("Matrix found.")
    
    # Compute bias vector: b[i] = -0.5 * (sum of row i)
    bias = compute_bias(M)
    print("Bias vector computed.")
    
    print("Now generating bitstream using integer matmul with bias, scaling, noise addition, and symmetric thresholding.")
    scale_factor = 2.0  # Increase the dynamic range.
    num_vectors = 20000  # Each vector produces 256 bits.
    noise_amp = 1.0  # Adjust noise amplitude as needed.
    all_bits = []
    progress_interval = max(1, num_vectors // 20)
    
    for i in range(num_vectors):
        # Generate a random 256-bit input vector (binary vector)
        v = [random.randint(0, 1) for _ in range(n)]
        # Multiply M by v and add bias.
        int_result = multiply_int_with_bias(M, v, bias)
        # Scale the result.
        scaled_result = [x * scale_factor for x in int_result]
        # Add noise and apply symmetric thresholding.
        final_bits = add_noise_and_threshold(scaled_result, noise_amp)
        all_bits.extend(final_bits)
        
        if (i + 1) % progress_interval == 0:
            pct = 100.0 * (i + 1) / num_vectors
            print(f"Processed {i+1} of {num_vectors} vectors ({pct:.1f}%)", file=sys.stderr)
    
    print(f"\nDone generating {len(all_bits)} bits total.\n")
    
    # Run basic statistical tests.
    results = []
    results.append(monobit_test(all_bits))
    results.append(runs_test(all_bits))
    results.append(block_frequency_test(all_bits, block_size=128))
    results.append(serial_test(all_bits))
    # New tests:
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

