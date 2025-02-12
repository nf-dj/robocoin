#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>
#import <sodium.h>

// Debug flag
static BOOL debugMode = NO;

#define INPUT_SIZE 32
#define VECTOR_SIZE 256
#define HIDDEN_SIZE 1024
#define NOISE_SIZE 256
#define BATCH_SIZE 2048
#define ROUNDS 64

// Noise generation functions from noise_gen.c
void compute_binary_and_noise_vectors(const uint8_t *input, float *binary_out, float *noise_out) {
    unsigned char hash[crypto_hash_sha256_BYTES];
    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        binary_out[i] = (float)((input[i / 8] >> (7 - (i % 8))) & 1);
    }
    
    crypto_hash_sha256(hash, input, crypto_hash_sha256_BYTES);
    
    for (int i = 0; i < NOISE_SIZE; i++) {
        noise_out[i] = (int8_t)((hash[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void compute_binary_and_noise_batch(const uint8_t *inputs, float *binary_out, float *noise_out, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        compute_binary_and_noise_vectors(
            inputs + (b * INPUT_SIZE),
            binary_out + (b * VECTOR_SIZE),
            noise_out + (b * NOISE_SIZE)
        );
    }
}

void generate_sequential_nonces(uint8_t *outputs, uint64_t start_nonce, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        // Zero out the full input
        memset(outputs + (i * INPUT_SIZE), 0, INPUT_SIZE);
        // Write sequential nonce to last 8 bytes (LSB)
        uint64_t nonce = start_nonce + i;
        for (int j = 0; j < 8; j++) {
            outputs[(i * INPUT_SIZE) + (24 + j)] = (nonce >> (8 * (7 - j))) & 0xFF;
        }
    }
}

// Helper function to print vectors
void print_vector(const char *label, float *vector, int size) {
    printf("%s: [", label);
    for (int i = 0; i < size; i++) {
        printf("%.0f", vector[i]);
        if (i < size - 1) printf(", ");
        if (i > 20) {
            printf("...");
            break;
        }
    }
    printf("]\n");
}

int count_leading_zeros(MLMultiArray *output, NSInteger row) {
    int zeros = 0;
    // Check each bit from the start until we find a 1
    for (NSInteger i = 0; i < 256; i++) {
        if ([output[(row * 256) + i] intValue] == 0) {
            zeros++;
        } else {
            break;
        }
    }
    return zeros;
}

@protocol MLFeatureProvider;

@interface InputFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong) MLMultiArray *input;
@property (nonatomic, strong) MLMultiArray *noise;
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@end

@implementation InputFeatureProvider
- (instancetype)initWithBatchSize:(NSInteger)batchSize startNonce:(uint64_t)startNonce {
    self = [super init];
    if (self) {
        NSError *error = nil;
        
        // Create input arrays
        self.input = [[MLMultiArray alloc] initWithShape:@[@(batchSize), @256]
                                              dataType:MLMultiArrayDataTypeFloat32
                                                error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error);
            return nil;
        }
        
        self.noise = [[MLMultiArray alloc] initWithShape:@[@(batchSize), @256]
                                              dataType:MLMultiArrayDataTypeFloat32
                                                error:&error];
        if (error) {
            NSLog(@"Error creating noise array: %@", error);
            return nil;
        }
        
        // Allocate memory for input batch
        uint8_t *input_batch = (uint8_t *)malloc(batchSize * INPUT_SIZE);
        generate_sequential_nonces(input_batch, startNonce, (int)batchSize);
        
        // Get pointers to MLMultiArray data
        float *binary_data = (float *)self.input.dataPointer;
        float *noise_data = (float *)self.noise.dataPointer;
        
        // Generate binary and noise vectors
        compute_binary_and_noise_batch(input_batch, binary_data, noise_data, (int)batchSize);
        
        // Debug output for first vectors in batch
        if (debugMode) {
            // Print nonce as 32-byte hex
            printf("Nonce: %016llx", startNonce);
            // Last 24 bytes as zeros
            for (int i = 0; i < 24; i++) {
                printf("00");
            }
            printf("\n");
            print_vector("First input vector", binary_data, VECTOR_SIZE);
            print_vector("First noise vector", noise_data, NOISE_SIZE);
        }
        
        free(input_batch);
        
        //self.featureNames = [NSSet setWithArray:@[@"input", @"noise"]];
        self.featureNames = [NSSet setWithArray:@[@"input"]];
    }
    return self;
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.input];
    } else if ([featureName isEqualToString:@"noise"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.noise];
    }
    return nil;
}
@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Parse arguments
        if (argc < 2) {
            NSLog(@"Usage: %s [-d] <difficulty>", argv[0]);
            return 1;
        }
        
        if (sodium_init() < 0) {
            NSLog(@"Error initializing libsodium");
            return 1;
        }
        
        // Parse arguments
        int argIndex = 1;
        if (strcmp(argv[argIndex], "-d") == 0) {
            debugMode = YES;
            argIndex++;
            if (argIndex >= argc) {
                NSLog(@"Missing difficulty parameter");
                return 1;
            }
        }
        
        // Parse difficulty
        int target_difficulty = atoi(argv[argIndex]);
        
        // Configure compute units
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        // Load and compile model
        NSString *modelPath = @"test_coreml.mlpackage";
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSError *error = nil;
        
        NSURL *compiledUrl = [MLModel compileModelAtURL:modelURL error:&error];
        if (error) {
            NSLog(@"Error compiling model: %@", error);
            return 1;
        }
        
        MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        
        NSLog(@"Mining with difficulty: %d%@", target_difficulty, debugMode ? @" (Debug mode enabled)" : @"");
        
        // Mining parameters
        NSInteger batchSize = BATCH_SIZE;
        __block uint64_t nonce = 0;
        __block uint64_t totalHashes = 0;
        __block NSDate *startTime = [NSDate date];
        __block int best_difficulty = 0;
        
        // Status display timer - 1 second intervals
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        
        dispatch_source_set_event_handler(timer, ^{
            NSTimeInterval elapsed = -[startTime timeIntervalSinceNow];
            double hashrate = totalHashes / elapsed;
            // TOPS = (hashes_per_second * OPS_PER_HASH) / 1e12
            double tops = (hashrate * (ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * VECTOR_SIZE * 4)) / 1e12;
            NSLog(@"Nonce: %llu | Hashrate: %.2f H/s | TOPS: %.2f | Best difficulty: %d", 
                  nonce, hashrate, tops, best_difficulty);
        });
        
        dispatch_resume(timer);
        
        // Mining loop
        while (true) {
            @autoreleasepool {
                // Create new input provider for current batch
                InputFeatureProvider *inputProvider = [[InputFeatureProvider alloc] 
                    initWithBatchSize:batchSize 
                    startNonce:nonce];
                
                if (!inputProvider) {
                    NSLog(@"Error creating input provider");
                    continue;
                }
                
                // Run inference
                id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
                if (error) {
                    NSLog(@"Error during inference: %@", error);
                    continue;
                }
                
                // Get output feature
                MLFeatureValue *outputFeature = [output featureValueForName:@"clip_65"];
                //MLFeatureValue *outputFeature = [output featureValueForName:@"clip_63"];
                if (!outputFeature) {
                    NSLog(@"Could not find output feature");
                    continue;
                }
                
                MLMultiArray *outputArray = [outputFeature multiArrayValue];
                
                // Debug output for first output vector
                if (debugMode) {
                    float *output_data = (float *)outputArray.dataPointer;
                    print_vector("First output vector", output_data, VECTOR_SIZE);
                    
                    // Convert first output to hex
                    uint8_t first_output_bytes[32] = {0};
                    for (NSInteger j = 0; j < 256; j++) {
                        NSInteger bitIndex = 7 - j % 8;
                        NSInteger byteIndex = j / 8;
                        first_output_bytes[byteIndex] |= ([outputArray[j] intValue] & 1) << bitIndex;
                    }
                    NSMutableString *outputHex = [NSMutableString string];
                    for (int j = 0; j < 32; j++) {
                        [outputHex appendFormat:@"%02x", first_output_bytes[j]];
                    }
                    NSLog(@"First output (hex): %@", outputHex);
                }
                
                // Check each output in batch
                for (NSInteger i = 0; i < batchSize; i++) {
                    int zeros = count_leading_zeros(outputArray, i);
                    if (zeros > best_difficulty) {
                        best_difficulty = zeros;
                    }
                    
                    if (zeros >= target_difficulty) {
                        // Found a solution!
                        uint64_t solution_nonce = nonce + i;
                        NSLog(@"\nSolution found!");
                        NSLog(@"Nonce: %llu", solution_nonce);
                        NSLog(@"Trailing zeros: %d", zeros);
                        
                        // Print full 32-byte input in hex format
                        NSMutableString *inputHex = [NSMutableString string];
                        // First 24 bytes are zeros
                        for (int j = 0; j < 24; j++) {
                            [inputHex appendString:@"00"];
                        }
                        // Last 8 bytes are the nonce
                        for (int j = 0; j < 8; j++) {
                            [inputHex appendFormat:@"%02x", (uint8_t)(solution_nonce >> (8 * (7 - j))) & 0xFF];
                        }
                        NSLog(@"Solution input (hex): %@", inputHex);
                        
                        // Convert model output to hex and display (in MSB order)
                        NSMutableString *outputHex = [NSMutableString string];
                        uint8_t output_bytes[32] = {0};
                        for (NSInteger j = 0; j < 256; j++) {
                            NSInteger bitIndex = 7 - j % 8;
                            NSInteger byteIndex = j / 8;
                            output_bytes[byteIndex] |= ([outputArray[(i * 256) + j] intValue] & 1) << bitIndex;
                        }
                        for (int j = 0; j < 32; j++) {
                            [outputHex appendFormat:@"%02x", output_bytes[j]];
                        }
                        NSLog(@"Model output (hex): %@", outputHex);
                        
                        dispatch_source_cancel(timer);
                        return 0;
                    }
                }
                
                nonce += batchSize;
                totalHashes += batchSize;
            }
        }
    }
    return 0;
}
