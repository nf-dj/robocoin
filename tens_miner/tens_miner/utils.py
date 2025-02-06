"""Utility functions."""

def count_leading_zero_bits(hash_bytes):
    """Count leading zero bits in a hash."""
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
    """Print hash bytes in little-endian hex format."""
    return "".join("{:02x}".format(b) for b in hash_bytes)

def print_hex_msb(hash_bytes):
    """Print hash bytes in MSB-first hex format."""
    # Take bytes in reverse order and format each
    return "".join("{:02x}".format(b) for b in reversed(hash_bytes))