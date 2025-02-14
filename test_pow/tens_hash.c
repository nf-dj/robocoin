#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sodium.h>

// --- Macro Definitions ---
#define INPUT_SIZE 256      // 256 bits (from 32 bytes) for input vector.
#define HIDDEN_SIZE 1024    // Hidden layer size.
#define NUM_HIDDEN_LAYERS 64
// Total layers: expansion (1) + hidden layers + compression (1)
#define NUM_LAYERS (1 + NUM_HIDDEN_LAYERS + 1)
#define NUM_NONZERO 512

// --- Layer Structure ---
typedef struct {
    int input_dim;
    int output_dim;
    // Matrix stored in row-major order, dimensions: (output_dim x input_dim)
    int8_t *matrix;
} Layer;

// Global variables for reporting.
int32_t global_max_accum = 0;
int global_max_nonzero = 0;

/* --- Utility Functions --- */

// Convert a hex string into bytes. Expects exactly len*2 hex characters.
int hex_to_bytes(const char *hex, uint8_t *bytes, size_t len) {
    if (strlen(hex) != len * 2) {
        return -1;
    }
    for (size_t i = 0; i < len; i++) {
        if (sscanf(hex + 2 * i, "%2hhx", &bytes[i]) != 1) {
            return -1;
        }
    }
    return 0;
}

// Given a binary vector (256 bits stored as ints 0 or 1),
// pack every 8 bits into one byte (big-endian) and store into out_bytes (32 bytes).
void pack_bits(const int *bits, uint8_t *out_bytes) {
    memset(out_bytes, 0, 32);
    for (int i = 0; i < INPUT_SIZE; i++) {
        if (bits[i])
            out_bytes[i / 8] |= (1 << (7 - (i % 8)));
    }
}

/* --- Matrix Generation Functions --- */

// Generates a dense matrix of shape (rows x cols) with entries in {-1, 0, 1}
// using a ChaCha20-based RNG. The key must be a 32-byte array.
// The nonce is built from nonce_counter in big-endian order.
// The matrix array must be pre-allocated by the caller with size rows * cols.
int generate_dense_matrix(int rows, int cols, const uint8_t *seed, uint64_t nonce_counter, int8_t *matrix) {
    int total = rows * cols;
    // Allocate a temporary buffer for the keystream.
    uint8_t *random_bytes = malloc(total);
    if (!random_bytes) {
        return -1;
    }

    // Build an 8-byte nonce (crypto_stream_chacha20_NONCEBYTES is 8 for ChaCha20)
    uint8_t nonce[crypto_stream_chacha20_NONCEBYTES];
    for (int i = 0; i < crypto_stream_chacha20_NONCEBYTES; i++) {
        nonce[i] = (uint8_t)(nonce_counter >> (8 * (crypto_stream_chacha20_NONCEBYTES - 1 - i)));
    }

    // Generate a pseudorandom keystream by encrypting a zero buffer.
    memset(random_bytes, 0, total);
    crypto_stream_chacha20_xor_ic(random_bytes, random_bytes, total, nonce, 0, seed);

    // Map each byte to a value in {-1, 0, 1} using modulo 4.
    // Mapping: 0 -> 0, 1 -> 1, 2 -> 0, 3 -> -1.
    for (int i = 0; i < total; i++) {
        uint8_t mod = random_bytes[i] % 4;
        if (mod == 0 || mod == 1) {
            matrix[i] = 0;
        } else if (mod == 2) {
            matrix[i] = 1;
        } else { // mod == 3
            matrix[i] = -1;
        }
    }

    free(random_bytes);
    return 0;
}

int generate_sparse_matrix(int rows, int cols, const uint8_t *seed, uint64_t nonce_counter, int8_t *matrix) {
    // Need 2 bytes per nonzero element per row, NUM_NONZERO positions per row.
    const int bytes_per_row = NUM_NONZERO * 2;
    size_t total_bytes = rows * bytes_per_row;
    
    // Allocate memory for random bytes.
    uint8_t *random_bytes = malloc(total_bytes);
    if (!random_bytes) return -1;
    
    // Create nonce for ChaCha20.
    uint8_t nonce[crypto_stream_chacha20_NONCEBYTES];
    for (int i = 0; i < crypto_stream_chacha20_NONCEBYTES; i++) {
        nonce[i] = (uint8_t)(nonce_counter >> (8 * (crypto_stream_chacha20_NONCEBYTES - 1 - i)));
    }
    
    // Generate random bytes.
    memset(random_bytes, 0, total_bytes);
    crypto_stream_chacha20_xor_ic(random_bytes, random_bytes, total_bytes, nonce, 0, seed);
    
    // Initialize matrix to zeros.
    memset(matrix, 0, rows * cols * sizeof(int8_t));
    
    // For each row, fill in exactly NUM_NONZERO nonzeros.
    for (int row = 0; row < rows; row++) {
        const uint8_t *row_bytes = random_bytes + row * bytes_per_row;
        for (int i = 0; i < NUM_NONZERO; i++) {
            uint8_t byte1 = row_bytes[i * 2];
            uint8_t byte2 = row_bytes[i * 2 + 1];
            
            // MSB of byte1 determines value.
            int8_t val = (byte1 & 0x80) ? 1 : -1;
            
            // Remaining 15 bits determine position mod cols.
            int pos = (((byte1 & 0x7F) << 8) | byte2) % cols;
            
            matrix[row * cols + pos] = val;
        }
    }
    
    free(random_bytes);
    return 0;
}

/* --- Forward Propagation --- */

// Compute output = A * (2*x - 1), where A is of shape (output_dim x input_dim).
// Here, input and A use int8_t (with values 0/1 and -1/0/1 respectively),
// but the multiplication is accumulated in int32_t.
void layer_forward(const Layer *layer, const int8_t *input, int8_t *output) {
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;

    // Precompute mapped input: x_mapped[i] = 2*x[i] - 1.
    // (This converts 0 -> -1 and 1 -> 1.)
    int8_t *x_mapped = malloc(in_dim * sizeof(int8_t));
    if (!x_mapped) {
        fprintf(stderr, "Memory allocation error in layer_forward\n");
        exit(1);
    }
    for (int i = 0; i < in_dim; i++) {
        x_mapped[i] = (int8_t)(2 * input[i] - 1);
    }

    // For each output neuron (each row of the matrix).
    for (int j = 0; j < out_dim; j++) {
        int32_t sum = 0;
        const int8_t *row = &layer->matrix[j * in_dim];
        int num_nonzero = 0;
        for (int i = 0; i < in_dim; i++) {
            sum += row[i] * x_mapped[i];
            if (row[i] != 0) {
                num_nonzero++;
            }
        }
        if (abs(sum) > global_max_accum) {
            global_max_accum = abs(sum);
        }
        if (num_nonzero > global_max_nonzero) {
            global_max_nonzero = num_nonzero;
        }
        output[j] = sum > 0 ? 1 : 0;
    }

    free(x_mapped);
}

/* --- Main Program --- */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <seed_hex> <input_hex>\n", argv[0]);
        fprintf(stderr, "Both seed and input should be 32-byte hex strings (64 hex characters).\n");
        return 1;
    }
    
    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }
    
    // Parse seed.
    uint8_t seed[32];
    if (hex_to_bytes(argv[1], seed, 32) != 0) {
        fprintf(stderr, "Invalid seed hex. Expected 64 hex characters.\n");
        return 1;
    }
    
    // Parse input.
    uint8_t input_bytes[32];
    if (hex_to_bytes(argv[2], input_bytes, 32) != 0) {
        fprintf(stderr, "Invalid input hex. Expected 64 hex characters.\n");
        return 1;
    }
    
    // Convert the 32-byte input into a 256-element int8_t vector.
    // Each bit (0 or 1) becomes one int8_t.
    int8_t *x = malloc(INPUT_SIZE * sizeof(int8_t));
    if (!x) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    for (int i = 0; i < INPUT_SIZE; i++) {
        int byte_index = i / 8;
        int bit_index = 7 - (i % 8);
        int bit = (input_bytes[byte_index] >> bit_index) & 1;
        x[i] = (int8_t)bit;
    }
    
    // Set up the layers.
    // Layers:
    //   - Expansion: from INPUT_SIZE (256) to HIDDEN_SIZE (1024)
    //   - Hidden layers: NUM_HIDDEN_LAYERS layers of (HIDDEN_SIZE x HIDDEN_SIZE)
    //   - Compression: from HIDDEN_SIZE (1024) to INPUT_SIZE (256)
    int num_layers = NUM_LAYERS;
    Layer *layers = malloc(num_layers * sizeof(Layer));
    if (!layers) {
        fprintf(stderr, "Memory allocation error\n");
        free(x);
        return 1;
    }
    uint64_t nonce_counter = 0;
    
    // --- Expansion layer ---
    layers[0].input_dim = INPUT_SIZE;
    layers[0].output_dim = HIDDEN_SIZE;
    layers[0].matrix = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(int8_t));
    if (!layers[0].matrix) {
        fprintf(stderr, "Memory allocation error\n");
        free(x);
        free(layers);
        return 1;
    }
    if (generate_dense_matrix(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter, layers[0].matrix) != 0) {
        fprintf(stderr, "Error generating expansion matrix\n");
        free(x);
        free(layers);
        return 1;
    }
    nonce_counter++;
    
    // --- Hidden layers ---
    for (int l = 1; l <= NUM_HIDDEN_LAYERS; l++) {
        layers[l].input_dim = HIDDEN_SIZE;
        layers[l].output_dim = HIDDEN_SIZE;
        layers[l].matrix = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(int8_t));
        if (!layers[l].matrix) {
            fprintf(stderr, "Memory allocation error\n");
            return 1;
        }
        if (generate_dense_matrix(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter, layers[l].matrix) != 0) {
            fprintf(stderr, "Error generating hidden matrix at layer %d\n", l);
            return 1;
        }
        nonce_counter++;
    }
    
    // --- Compression layer ---
    int comp_index = NUM_HIDDEN_LAYERS + 1;
    layers[comp_index].input_dim = HIDDEN_SIZE;
    layers[comp_index].output_dim = INPUT_SIZE;
    layers[comp_index].matrix = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(int8_t));
    if (!layers[comp_index].matrix) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    if (generate_dense_matrix(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter, layers[comp_index].matrix) != 0) {
        fprintf(stderr, "Error generating compression matrix\n");
        return 1;
    }
    nonce_counter++;
    
    /* --- Forward Propagation --- */
    // Pre-allocate two buffers. The maximum dimension among all layers is HIDDEN_SIZE.
    int max_dim = HIDDEN_SIZE;
    int8_t *buf1 = malloc(max_dim * sizeof(int8_t));
    int8_t *buf2 = malloc(max_dim * sizeof(int8_t));
    if (!buf1 || !buf2) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    // Copy the initial input (of size INPUT_SIZE) into buf1.
    memcpy(buf1, x, INPUT_SIZE * sizeof(int8_t));
    free(x);
    
    // Set up pointers for swapping.
    int8_t *current = buf1;
    int8_t *next = buf2;
    for (int l = 0; l < num_layers; l++) {
        int out_dim = layers[l].output_dim;
        layer_forward(&layers[l], current, next);
        // Swap pointers for the next layer.
        int8_t *temp = current;
        current = next;
        next = temp;
    }
    // After the loop, "current" holds the final output vector (of dimension INPUT_SIZE).
    
    // Threshold at 0.5 (or simply >0) to obtain a binary vector.
    int bits[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        bits[i] = (current[i] > 0) ? 1 : 0;
    }
    uint8_t output_bytes[32];
    pack_bits(bits, output_bytes);
    
    // Print the final output as hex.
    for (int i = 0; i < 32; i++) {
        printf("%02x", output_bytes[i]);
    }
    printf("\n");
    
    // Cleanup.
    free(buf1);
    free(buf2);
    for (int l = 0; l < num_layers; l++) {
        free(layers[l].matrix);
    }
    free(layers);
    
    printf("Maximum absolute accumulator value: %d\n", global_max_accum);
    printf("Maximum number of nonzeros in a row: %d\n", global_max_nonzero);
    
    return 0;
}

