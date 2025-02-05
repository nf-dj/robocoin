"""Constants used throughout the TensHash miner."""

# Mining constants
IN_SIZE = 32         # 32 bytes input
BITS = IN_SIZE * 8   # 256 bits
HIDDEN = 256         # state size (256 bits)
ROUNDS = 64
BATCH_SIZE = 1024    # adjustable
OPS_PER_HASH = 256 * 256 * 2 * 64  # 8,388,608 operations per hash
