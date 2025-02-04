# Paths
OPENSSL_PATH = /opt/homebrew/opt/openssl@3
SODIUM_PATH = /opt/homebrew/opt/libsodium
METAL_PATH = $(shell xcrun -f metal)
METALLIB_PATH = $(shell xcrun -f metallib)

# Compiler and flags
CC = clang
CXX = clang++

# Common flags
CFLAGS = -Wall -Wextra -O3 -std=c11 \
         -I$(OPENSSL_PATH)/include \
         -I$(SODIUM_PATH)/include

CXXFLAGS = -Wall -Wextra -O3 -std=c++17 \
           -framework Metal -framework Foundation \
           -fobjc-arc

# Common linker flags
LDFLAGS = -L$(OPENSSL_PATH)/lib -L$(SODIUM_PATH)/lib \
          -lssl -lcrypto -lsodium

# Targets
all: tens_hash tens_pow tens_pow_metal default.metallib

tens_hash: tens_hash.c
	$(CC) $(CFLAGS) -DHASH_MAIN -o $@ $< $(LDFLAGS)

tens_pow: tens_pow.c tens_hash.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tens_pow_metal: tens_pow_metal.mm
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

default.metallib: tens_pow_metal.metal
	$(METAL_PATH) -c tens_pow_metal.metal -o tens_pow_metal.air
	$(METALLIB_PATH) tens_pow_metal.air -o default.metallib

clean:
	rm -f tens_hash tens_pow tens_pow_metal *.air *.metallib

.PHONY: all clean