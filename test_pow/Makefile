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
           -fobjc-arc \
           -I$(OPENSSL_PATH)/include \
           -I$(SODIUM_PATH)/include

# Common linker flags
LDFLAGS = -L$(OPENSSL_PATH)/lib -L$(SODIUM_PATH)/lib \
          -lssl -lcrypto -lsodium

# Targets
all: tens_hash tens_pow tens_pow_metal_int8 default.metallib check_metal tens_hash_coreml test_coreml hadamard_pow

check_metal: check_metal.mm
	$(CXX) -framework Metal -framework Foundation -o $@ $^

test_trits: test_trits.c
	$(CC) $(CFLAGS) -DHASH_MAIN -o $@ $< $(LDFLAGS)

tens_hash: tens_hash.c
	$(CC) $(CFLAGS) -DHASH_MAIN -o $@ $< $(LDFLAGS)

tens_pow: tens_pow.c tens_hash.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tens_pow_metal_int8: tens_pow_metal_int8.mm
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

tens_hash_coreml: tens_hash_coreml.m
	$(CC) -fobjc-arc -framework Foundation -framework CoreML -o $@ $^

libnoise.so: noise_gen.c
	gcc -O3 -shared -fPIC -o libnoise.so noise_gen.c -lsodium $(CFLAGS) $(LDFLAGS)

default.metallib: tens_pow_metal_int8.metal
	$(METAL_PATH) -c tens_pow_metal_int8.metal -o tens_pow_metal_int8.air
	$(METALLIB_PATH) tens_pow_metal_int8.air -o default.metallib
	rm -f tens_pow_metal_int8.air

test_coreml: test_coreml.m
	$(CC) $(CFLAGS) -fobjc-arc -framework Foundation -framework CoreML -o $@ $^ $(LDFLAGS)

hadamard_pow: hadamard_pow.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f tens_hash tens_pow tens_pow_metal_int8 tens_hash_coreml test_coreml hadamard_pow *.air *.metallib

.PHONY: all clean
