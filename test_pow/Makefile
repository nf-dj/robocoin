# Paths
OPENSSL_PATH = /opt/homebrew/opt/openssl@3
SODIUM_PATH = /opt/homebrew/opt/libsodium

# Compiler and flags
CC = clang

# Common flags
CFLAGS = -Wall -Wextra -O3 -std=c11 \
         -I$(OPENSSL_PATH)/include \
         -I$(SODIUM_PATH)/include

# Common linker flags
LDFLAGS = -L$(OPENSSL_PATH)/lib -L$(SODIUM_PATH)/lib \
          -lssl -lcrypto -lsodium

# Targets
all: tens_hash tens_pow

tens_hash: tens_hash.c
	$(CC) $(CFLAGS) -DHASH_MAIN -o $@ $< $(LDFLAGS)

tens_pow: tens_pow.c tens_hash.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f tens_hash tens_pow

.PHONY: all clean

