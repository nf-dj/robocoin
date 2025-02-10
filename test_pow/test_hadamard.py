import random
import numpy as np
import math
import sys

###############################################################################
# 1. Hadamard Matrix Generation (Sylvester’s construction)
###############################################################################
def hadamard_matrix(n):
    """
    Recursively constructs a Hadamard matrix of order n using Sylvester's construction.
    n must be a power of 2.
    """
    if n == 1:
        return np.array([[1]])
    else:
        H_small = hadamard_matrix(n // 2)
        # Construct H using block matrix: [H H; H -H]
        return np.block([[H_small, H_small],
                         [H_small, -H_small]])

def randomize_hadamard(H):
    """
    Multiplies each row of Hadamard matrix H by a random sign (+1 or -1)
    to randomize the bias (e.g. the first row will no longer be all +1's).
    Returns the randomized Hadamard matrix.
    """
    n = H.shape[0]
    row_signs = np.random.choice([1, -1], size=(n, 1))
    return H * row_signs

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
# 4. Simple Threshold Function (No Hysteresis)
###############################################################################
def simple_threshold(values):
    """
    Applies a simple thresholding to convert continuous values to binary bits.
    For each element in the NumPy array 'values':
      - If value > 0, output 1.
      - If value < 0, output 0.
      - If value == 0, randomly choose 0 or 1.
    Returns a NumPy array of binary bits.
    """
    out = np.where(values > 0, 1, 0)
    zero_mask = (values == 0)
    if np.any(zero_mask):
        out[zero_mask] = np.random.randint(0, 2, size=np.sum(zero_mask))
    return out

###############################################################################
# 5. Noise Addition and Thresholding (Using Gaussian Noise, No Hysteresis)
###############################################################################
def add_noise_and_threshold(values, noise_std=0.5):
    """
    Given a NumPy array 'values', adds Gaussian noise with mean 0 and standard deviation noise_std,
    then applies the simple thresholding function.
    Returns a NumPy array of binary bits.
    """
    noise = np.random.normal(0, noise_std, size=values.shape)
    #noisy_values = values + noise
    noisy_values = values
    return simple_threshold(noisy_values)

###############################################################################
# 6. Statistical Tests
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

def block_frequency_test(bits, block_size=2048):
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
    M_val = max(abs(s) for s in S)
    start = int(math.ceil(M_val / math.sqrt(N)))
    sum_term = 0.0
    for k in range(start, 101):
        arg1 = ((4 * k + 1) * M_val) / math.sqrt(N)
        arg2 = ((4 * k - 1) * M_val) / math.sqrt(N)
        term = math.erfc(arg1 / math.sqrt(2)) - math.erfc(arg2 / math.sqrt(2))
        sum_term += term
    p_value = 1 - sum_term
    return {"test": "Cumulative Sums", "M": M_val, "p_value": p_value}

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
# 8. Main: Multiple Rounds Processing and Run Tests Using Hadamard Matrix
###############################################################################
def main():
    n = 256
    num_rounds = 16  # Number of rounds to cascade.
    print("Generating a Hadamard matrix of size 256 using Sylvester's construction...")
    H = hadamard_matrix(n)
    # Randomize rows by multiplying each row with a random sign (+1 or -1).
    H = H * np.random.choice([1, -1], size=(n, 1))
    # (H is always invertible; no need to check rank here.)
    M = H  # Use the Hadamard matrix as our mixing matrix.
    print("Hadamard matrix generated and randomized.")
    
    bias = compute_bias(M)
    print("Bias vector computed.")
    
    print("Now processing input vectors for {} rounds...".format(num_rounds))
    scale_factor = 1.0  # Adjust as needed.
    noise_std = 1.0     # Gaussian noise standard deviation.
    num_vectors = 20000  # Each initial vector produces one final 256-bit output.
    
    all_bits = []
    progress_interval = max(1, num_vectors // 20)
    
    for i in range(num_vectors):
        # Generate a random 256-bit binary input vector.
        v = np.random.randint(0, 2, size=n)
        # Perform multiple rounds.
        for r in range(num_rounds):
            # Compute r = M*v + bias
            int_result = M.dot(v) + bias
            # Scale the result.
            scaled_result = int_result * scale_factor
            # Add Gaussian noise and threshold.
            v = add_noise_and_threshold(scaled_result, noise_std)
        # After all rounds, v is the final 256-bit output.
        all_bits.extend(v.tolist())
        
        if (i+1) % progress_interval == 0:
            pct = 100.0 * (i+1) / num_vectors
            print(f"Processed {i+1} of {num_vectors} vectors ({pct:.1f}%)", file=sys.stderr)
    
    print(f"\nDone generating {len(all_bits)} bits total.\n")
    
    # Run statistical tests.
    results = []
    results.append(monobit_test(np.array(all_bits)))
    results.append(runs_test(np.array(all_bits)))
    # Use a larger block size (2048 bits) for block frequency.
    results.append(block_frequency_test(np.array(all_bits), block_size=8192))
    results.append(serial_test(np.array(all_bits)))
    results.append(cusum_test(all_bits))
    results.append(autocorrelation_test(np.array(all_bits), lag=1))
    
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

