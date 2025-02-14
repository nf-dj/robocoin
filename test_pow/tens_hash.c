#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sodium.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 1024
#define NUM_HIDDEN_LAYERS 64
// Total layers: expansion (1) + hidden layers + compression (1)
#define NUM_LAYERS (1 + NUM_HIDDEN_LAYERS + 1)

typedef struct {
    int input_dim;
    int output_dim;
    // Matrix stored in row-major order, dimensions: (input_dim x output_dim)
    float *matrix;
    // Bias removed.
} Layer;

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

// Using ChaCha20, generate a matrix of dimensions (rows x cols)
// with entries chosen from {-1, 0, 1} (mapping: 0→-1, 1→0, 2→1).
// The key is the 32-byte seed. The nonce is computed from nonce_counter
// as an 8-byte big-endian value.
int generate_matrix(int rows, int cols, const uint8_t *seed, uint64_t nonce_counter, float *matrix) {
    size_t num_bytes = rows * cols;
    uint8_t *rand_bytes = malloc(num_bytes);
    if (!rand_bytes) {
        return -1;
    }
    // Create an 8-byte nonce.
    uint8_t nonce[crypto_stream_chacha20_NONCEBYTES];
    for (int i = 0; i < crypto_stream_chacha20_NONCEBYTES; i++) {
        nonce[i] = (uint8_t)(nonce_counter >> (8 * (crypto_stream_chacha20_NONCEBYTES - 1 - i)));
    }
    // Generate pseudorandom bytes (encrypting a zero buffer).
    memset(rand_bytes, 0, num_bytes);
    crypto_stream_chacha20_xor_ic(rand_bytes, rand_bytes, num_bytes, nonce, 0, seed);
    
    // Map each byte modulo 3 to an element in {-1, 0, 1}.
    for (size_t i = 0; i < num_bytes; i++) {
        uint8_t mod = rand_bytes[i] % 3;
        if (mod == 0)
            matrix[i] = -1.0f;
        else if (mod == 1)
            matrix[i] = 0.0f;
        else
            matrix[i] = 1.0f;
    }
    
    free(rand_bytes);
    return 0;
}

// Transpose a matrix. 'in' has dimensions (rows x cols); 'out' will have dimensions (cols x rows).
void transpose_matrix(const float *in, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

/* --- Forward Propagation --- */

// Perform forward propagation for one layer.
// Instead of adding a bias, we initialize the output with zeros.
void layer_forward(const Layer *layer, const float *input, float *output) {
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;

    // Initialize output with 0.0 (bias removed).
    for (int j = 0; j < out_dim; j++) {
        output[j] = 0.0f;
    }
    
    // For each input element, compute its mapped value (2*x - 1)
    // and accumulate its contribution to the output.
    for (int i = 0; i < in_dim; i++) {
        float x_mapped = 2.0f * input[i] - 1.0f;
        const float *mat_row = &layer->matrix[i * out_dim];
        for (int j = 0; j < out_dim; j++) {
            output[j] += x_mapped * mat_row[j];
        }
    }
    
    // Clip the results to the [0, 1] range.
    for (int j = 0; j < out_dim; j++) {
        if (output[j] < 0.0f)
            output[j] = 0.0f;
        else if (output[j] > 1.0f)
            output[j] = 1.0f;
    }
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
    
    // Parse seed (32 bytes).
    uint8_t seed[32];
    if (hex_to_bytes(argv[1], seed, 32) != 0) {
        fprintf(stderr, "Invalid seed hex. Expected 64 hex characters.\n");
        return 1;
    }
    
    // Parse input (32 bytes).
    uint8_t input_bytes[32];
    if (hex_to_bytes(argv[2], input_bytes, 32) != 0) {
        fprintf(stderr, "Invalid input hex. Expected 64 hex characters.\n");
        return 1;
    }
    
    // Convert the 32-byte input into a 256-element float vector.
    // Each bit becomes one element (0.0 or 1.0).
    float *x = malloc(INPUT_SIZE * sizeof(float));
    if (!x) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    for (int i = 0; i < INPUT_SIZE; i++) {
        int byte_index = i / 8;
        int bit_index = 7 - (i % 8);
        int bit = (input_bytes[byte_index] >> bit_index) & 1;
        x[i] = (float) bit;
    }
    
    // Set up the layers.
    // Layer 0: Expansion layer (INPUT_SIZE -> HIDDEN_SIZE)
    // Layers 1 to NUM_HIDDEN_LAYERS: Hidden layers (HIDDEN_SIZE -> HIDDEN_SIZE)
    // Layer NUM_HIDDEN_LAYERS+1: Compression layer (HIDDEN_SIZE -> INPUT_SIZE)
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
    // Generate a matrix of shape (HIDDEN_SIZE x INPUT_SIZE) then transpose it.
    float *temp_matrix = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    if (!temp_matrix) {
        fprintf(stderr, "Memory allocation error\n");
        free(x);
        free(layers);
        return 1;
    }
    if (generate_matrix(HIDDEN_SIZE, INPUT_SIZE, seed, nonce_counter, temp_matrix) != 0) {
        fprintf(stderr, "Error generating expansion matrix\n");
        free(x);
        free(layers);
        free(temp_matrix);
        return 1;
    }
    nonce_counter++;
    layers[0].matrix = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    if (!layers[0].matrix) {
        fprintf(stderr, "Memory allocation error\n");
        free(x);
        free(layers);
        free(temp_matrix);
        return 1;
    }
    transpose_matrix(temp_matrix, layers[0].matrix, HIDDEN_SIZE, INPUT_SIZE);
    free(temp_matrix);
    
    // --- Hidden layers ---
    for (int l = 1; l <= NUM_HIDDEN_LAYERS; l++) {
        layers[l].input_dim = HIDDEN_SIZE;
        layers[l].output_dim = HIDDEN_SIZE;
        layers[l].matrix = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
        if (!layers[l].matrix) {
            fprintf(stderr, "Memory allocation error\n");
            return 1;
        }
        if (generate_matrix(HIDDEN_SIZE, HIDDEN_SIZE, seed, nonce_counter, layers[l].matrix) != 0) {
            fprintf(stderr, "Error generating hidden matrix at layer %d\n", l);
            return 1;
        }
        nonce_counter++;
    }
    
    // --- Compression layer ---
    int comp_index = NUM_HIDDEN_LAYERS + 1;
    layers[comp_index].input_dim = HIDDEN_SIZE;
    layers[comp_index].output_dim = INPUT_SIZE;
    temp_matrix = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    if (!temp_matrix) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    if (generate_matrix(INPUT_SIZE, HIDDEN_SIZE, seed, nonce_counter, temp_matrix) != 0) {
        fprintf(stderr, "Error generating compression matrix\n");
        return 1;
    }
    nonce_counter++;
    layers[comp_index].matrix = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    if (!layers[comp_index].matrix) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    transpose_matrix(temp_matrix, layers[comp_index].matrix, INPUT_SIZE, HIDDEN_SIZE);
    free(temp_matrix);
    
    /* --- Forward Propagation Without Loop Allocations --- */
    // Pre-allocate two buffers. The maximum dimension among all layers is HIDDEN_SIZE.
    int max_dim = HIDDEN_SIZE;
    float *buf1 = malloc(max_dim * sizeof(float));
    float *buf2 = malloc(max_dim * sizeof(float));
    if (!buf1 || !buf2) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }
    // Copy the initial input (of size INPUT_SIZE) into buf1.
    memcpy(buf1, x, INPUT_SIZE * sizeof(float));
    free(x);
    
    // Set up pointers for swapping.
    float *current = buf1;
    float *next = buf2;
    
    for (int l = 0; l < num_layers; l++) {
        int out_dim = layers[l].output_dim;
        layer_forward(&layers[l], current, next);
        // Swap pointers so that "next" becomes "current" for the next iteration.
        float *temp = current;
        current = next;
        next = temp;
    }
    // After the loop, "current" holds the final output vector.
    
    // Threshold at 0.5 to obtain a binary vector.
    int bits[INPUT_SIZE];  // Final output dimension is INPUT_SIZE (256).
    for (int i = 0; i < INPUT_SIZE; i++) {
        bits[i] = (current[i] > 0.5f) ? 1 : 0;
    }
    uint8_t output_bytes[32];
    pack_bits(bits, output_bytes);
    
    // Print the output as hex.
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
    
    return 0;
}
