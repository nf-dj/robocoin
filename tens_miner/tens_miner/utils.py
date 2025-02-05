"""Utility functions for TensHash mining."""

def count_leading_zero_bits(hash_bytes):
    """Count the number of leading zero bits in the hash (big-endian display)."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            for bit in range(7, -1, -1):
                if (byte >> bit) & 1:
                    return count
                count += 1
    return count

def print_hex_le(hash_bytes):
    """Return a hex string for the bytes in reverse order (big-endian display)."""
    return "".join("{:02x}".format(b) for b in hash_bytes[::-1])
