#!/usr/bin/env python3
import numpy as np
import argparse
import random
import math

np.set_printoptions(threshold=1000000)

###############################################################################
# Hadamard Matrix Functions
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
        return np.block([[H_small, H_small],
                         [H_small, -H_small]])

def generate_randomized_hadamard_matrix(n):
    """
    Generates a Hadamard matrix of size n using Sylvester's construction,
    multiplies each row by a random sign (+1 or -1), and then randomly permutes
    both rows and columns.
    Returns the randomized Hadamard matrix.
    """
    H = hadamard_matrix(n)
    # Multiply each row by a random sign.
    row_signs = np.random.choice([1, -1], size=(n, 1))
    H = H * row_signs
    # Randomly permute rows and columns.
    row_perm = np.random.permutation(n)
    col_perm = np.random.permutation(n)
    H = H[row_perm, :][:, col_perm]
    return H

###############################################################################
# Greedy Matrix Generation Functions
###############################################################################
def generate_candidate_row(n, q):
    """
    Generate a candidate row of length n with entries in {-1, 0, 1} where:
       +1 with probability q,
       -1 with probability q,
       0  with probability 1-2q.
    """
    r = np.random.rand(n)
    row = np.zeros(n, dtype=int)
    row[r < q] = 1
    row[(r >= q) & (r < 2*q)] = -1
    return row

def generate_approx_orthogonal_ternary_matrix(n, q, tol, max_attempts):
    """
    Generate an n x n matrix A with entries in {-1,0,1} row-by-row.
    A candidate row is accepted if its dot product with every previously accepted
    row is within ±tol.
    
    Returns A as a NumPy array if successful, or None otherwise.
    """
    rows = []
    for i in range(n):
        accepted = False
        for attempt in range(max_attempts):
            candidate = generate_candidate_row(n, q)
            if all(abs(np.dot(candidate, r)) <= tol for r in rows):
                rows.append(candidate)
                print(f"Row {i+1}/{n} accepted after {attempt+1} attempts. Nonzeros: {np.count_nonzero(candidate)}")
                accepted = True
                break
        if not accepted:
            print(f"Failed to generate row {i+1} after {max_attempts} attempts.")
            return None
    return np.vstack(rows)

###############################################################################
# Noise and Thresholding Functions
###############################################################################
def simple_threshold(values):
    """
    Converts each element in the NumPy array 'values' to a binary bit:
      - 1 if value > 0,
      - 0 if value < 0,
      - if value == 0, randomly choose 0 or 1.
    Returns a NumPy array of binary bits.
    """
    out = np.where(values > 0, 1, 0)
    zero_mask = (values == 0)
    if np.any(zero_mask):
        out[zero_mask] = np.random.randint(0, 2, size=np.sum(zero_mask))
    return out

def add_noise_and_threshold(values, noise_std=1.0):
    """
    Adds Gaussian noise (mean 0, std noise_std) to the NumPy array 'values'
    and then applies simple_threshold.
    Returns a NumPy array of binary bits.
    """
    noise = np.random.normal(0, noise_std, size=values.shape)
    noisy_values = values + noise
    return simple_threshold(noisy_values)

###############################################################################
# Matrix-Vector Multiplication and Bias Functions
###############################################################################
def compute_bias(M):
    """
    Computes a bias vector b from matrix M such that:
      b[i] = -0.5 * (sum of row i).
    Returns a NumPy array.
    """
    return -0.5 * np.sum(M, axis=1)

def multiply_with_bias(M, v, bias):
    """
    Computes r = M · v + bias.
    M is an (n x n) NumPy array, v is a binary vector of length n,
    and bias is a vector of length n.
    """
    return M.dot(v) + bias

###############################################################################
# Statistical Test Functions
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

###############################################################################
# Main: Generate Matrix, Transform Inputs with Multiple Rounds, and Run Tests
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Generate an n x n ternary weighting matrix with approximate orthogonality using a greedy method (seeded), then transform random input vectors using multiple rounds of matrix multiplication, bias, scaling, Gaussian noise, and thresholding. Optionally, use randomly permuted Hadamard matrices instead. Finally, check the resulting bitstream using standard randomness tests."
    )
    parser.add_argument("--n", type=int, default=256,
                        help="Matrix size (n x n). Default is 256.")
    parser.add_argument("--q", type=float, default=0.05,
                        help="Probability for each entry to be +1 (and -1).")
    parser.add_argument("--tol", type=int, default=3,
                        help="Tolerance for dot product between distinct rows. Default is 3.")
    parser.add_argument("--max_attempts", type=int, default=100000,
                        help="Maximum candidate attempts per row. Default is 100000.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (optional).")
    parser.add_argument("--num_vectors", type=int, default=20000,
                        help="Number of random input vectors to process. Default is 20000.")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="Scaling factor for matrix-vector product. Default is 1.0.")
    parser.add_argument("--noise_std", type=float, default=1.0,
                        help="Standard deviation for Gaussian noise. Default is 1.0.")
    parser.add_argument("--num_rounds", type=int, default=16,
                        help="Number of rounds of processing per input vector. Default is 16.")
    parser.add_argument("--block_size", type=int, default=8192,
                        help="Block size for block frequency test. Default is 8192.")
    parser.add_argument("--use_hadamard", action="store_true",
                        help="If set, use randomly permuted Hadamard matrices for each round instead of greedy matrices.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    n = args.n
    q = args.q
    tol = args.tol
    max_attempts = args.max_attempts

    # If not using Hadamard matrices, generate a sample greedy matrix and show diagnostics.
    if not args.use_hadamard:
        print(f"Generating a {n}x{n} ternary matrix with q={q} and tol={tol} (max_attempts={max_attempts}) using the greedy method.")
        A = generate_approx_orthogonal_ternary_matrix(n, q, tol, max_attempts)
        if A is None:
            print("Matrix generation failed.")
            return
        print("\nGenerated Matrix A:")
        print(A)
        # Verify approximate orthogonality.
        orthogonal = True
        for i in range(n):
            for j in range(i+1, n):
                d = abs(np.dot(A[i], A[j]))
                if d > tol:
                    print(f"Rows {i} and {j} have dot product {d} (tolerance {tol}).")
                    orthogonal = False
                    break
            if not orthogonal:
                break
        print("\nApproximate orthogonality check:", "Passed" if orthogonal else "Failed")
        nonzero_counts = [np.count_nonzero(row) for row in A]
        avg_nonzeros = np.mean(nonzero_counts)
        print(f"Average nonzeros per row: {avg_nonzeros:.2f}")
    else:
        print("Using randomly permuted Hadamard matrices for each round (greedy matrix diagnostics skipped).")
    
    # Generate matrices for rounds.
    print(f"\nGenerating matrices for {args.num_rounds} rounds...")
    round_matrices = []
    round_biases = []
    if args.use_hadamard:
        for r in range(args.num_rounds):
            H = hadamard_matrix(n)
            # Randomize rows by multiplying by a random sign.
            row_signs = np.random.choice([1, -1], size=(n, 1))
            H = H * row_signs
            # Randomly permute rows and columns.
            row_perm = np.random.permutation(n)
            col_perm = np.random.permutation(n)
            H = H[row_perm, :][:, col_perm]
            round_matrices.append(H)
            round_biases.append(-0.5 * np.sum(H, axis=1))
            print(f"Round {r+1} Hadamard matrix generated.")
    else:
        for r in range(args.num_rounds):
            M_r = generate_approx_orthogonal_ternary_matrix(n, q, tol, max_attempts)
            if M_r is None:
                print(f"Matrix generation failed in round {r+1}.")
                return
            round_matrices.append(M_r)
            round_biases.append(-0.5 * np.sum(M_r, axis=1))
            print(f"Round {r+1} matrix generated.")
    
    num_vectors = args.num_vectors
    scale_factor = args.scale_factor
    noise_std = args.noise_std
    num_rounds = args.num_rounds
    
    print(f"\nTransforming random input vectors through {num_rounds} rounds and generating bitstream...")
    all_bits = []
    progress_interval = max(1, num_vectors // 20)
    
    for i in range(num_vectors):
        # Generate a random 256-bit binary input vector.
        v = np.random.randint(0, 2, size=n)
        # Cascade through the rounds.
        for r in range(num_rounds):
            M_r = round_matrices[r]
            bias_r = round_biases[r]
            int_result = M_r.dot(v) + bias_r
            scaled_result = int_result * scale_factor
            v = simple_threshold(scaled_result + np.random.normal(0, noise_std, size=scaled_result.shape))
        # After num_rounds, v is the final 256-bit output.
        all_bits.extend(v.tolist())
        if (i+1) % progress_interval == 0:
            pct = 100.0 * (i+1) / num_vectors
            print(f"Processed {i+1} of {num_vectors} vectors ({pct:.1f}%)")
    
    print(f"\nDone generating {len(all_bits)} bits total.\n")
    
    # Run statistical tests.
    results = []
    results.append(monobit_test(np.array(all_bits)))
    results.append(runs_test(np.array(all_bits)))
    results.append(block_frequency_test(np.array(all_bits), block_size=args.block_size))
    results.append(serial_test(np.array(all_bits)))
    results.append(cusum_test(all_bits))
    results.append(autocorrelation_test(np.array(all_bits), lag=1))
    
    alpha = 0.01
    print("\n=== BASIC STATISTICAL TEST RESULTS ===")
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

if __name__ == '__main__':
    main()

