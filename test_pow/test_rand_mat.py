import random
import numpy as np
import math
import sys

###############################################################################
# 1. AES S-Box (Standard AES S-box: 256-entry lookup table for 8-bit values)
###############################################################################
AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]

###############################################################################
# 2. Random Invertible Ternary Matrix Generation
###############################################################################
def random_ternary_matrix(n, fill_prob=0.4):
    """
    Generates an n x n matrix with entries in {-1, 0, +1}.
    Each entry is 0 with probability (1 - fill_prob) and ±1 with probability fill_prob/2 each.
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
# 3. Bias Computation
###############################################################################
def compute_bias(M):
    """
    Computes a bias vector b for matrix M such that for each row i:
      b[i] = -0.5 * (sum of row i)
    This subtracts the expected value when multiplying by a random binary vector (E[v_j]=0.5).
    """
    bias = []
    for row in M:
        s = sum(row)
        bias.append(-0.5 * s)
    return bias

###############################################################################
# 4. Matrix-Vector Multiplication with Bias
###############################################################################
def multiply_int_with_bias(M, v, bias):
    """
    Multiplies matrix M (n x n) by vector v (length n) using integer arithmetic,
    and then adds the corresponding bias.
    Returns a list of integers (length n).
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
# 5. Apply AES S-box on Integer Vector and Extract the MSB
###############################################################################
def apply_aes_sbox_on_integers(int_vector):
    """
    For each integer x in int_vector:
      1. Compute byte_val = int(x % 256) (mapping x into 0..255, as an integer).
      2. Look up AES_SBOX[byte_val].
      3. Extract the most significant bit (MSB) of the S-box output.
    Returns a list of bits (0 or 1) of the same length as int_vector.
    """
    out_bits = []
    for x in int_vector:
        byte_val = int(x % 256)  # Convert the result to an integer.
        sbox_val = AES_SBOX[byte_val]
        msb = (sbox_val >> 7) & 1
        out_bits.append(msb)
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
    p_value = math.erfc(abs(z)/math.sqrt(2))
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
        if bits[i] != bits[i-1]:
            runs += 1
    mean_runs = 1 + 2 * n * pi * (1 - pi)
    var_runs = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1) / (n - 1)
    if var_runs <= 0:
        return {"test": "Runs", "p_value": None, "note": "Variance <= 0"}
    z = (runs - mean_runs) / math.sqrt(var_runs)
    p_value = math.erfc(abs(z)/math.sqrt(2))
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
    for i in range(n-1):
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
        return regularized_gamma_upper(df/2, x/2)
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
        if abs(a0) < eps:
            a0 = eps
        if abs(b0) < eps:
            b0 = eps
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
    
    print("Now generating bitstream using integer matmul with bias and S-box substitution.")
    # Number of input vectors (each produces 256 output bits)
    num_vectors = 20000
    all_bits = []
    progress_interval = max(1, num_vectors // 20)
    
    for i in range(num_vectors):
        # Generate a random 256-bit input vector (binary vector of 0's and 1's)
        v = [random.randint(0, 1) for _ in range(n)]
        # Multiply M by v using integer arithmetic and add bias.
        int_result = multiply_int_with_bias(M, v, bias)
        # Apply AES S-box on each integer and extract the MSB.
        final_bits = apply_aes_sbox_on_integers(int_result)
        # Append the 256-bit output to our overall bitstream.
        all_bits.extend(final_bits)
        
        if (i + 1) % progress_interval == 0:
            pct = 100.0 * (i + 1) / num_vectors
            print(f"Processed {i+1} of {num_vectors} vectors ({pct:.1f}%)", file=sys.stderr)
    
    print(f"\nDone generating {len(all_bits)} bits total.\n")
    
    # Run basic statistical tests on the overall bitstream.
    results = []
    results.append(monobit_test(all_bits))
    results.append(runs_test(all_bits))
    results.append(block_frequency_test(all_bits, block_size=128))
    results.append(serial_test(all_bits))
    
    # Significance threshold for tests.
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

