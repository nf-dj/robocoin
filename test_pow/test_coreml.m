#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>
#import <sodium.h>

#define INPUT_SIZE 32
#define VECTOR_SIZE 256
#define NOISE_SIZE 256

// Noise generation functions from noise_gen.c
void compute_binary_and_noise_vectors(const uint8_t *input, float *binary_out, float *noise_out) {
    unsigned char first_hash[crypto_hash_sha256_BYTES];
    unsigned char second_hash[crypto_hash_sha256_BYTES];
    
    // First SHA256 for binary vector
    crypto_hash_sha256(first_hash, input, INPUT_SIZE);
    
    // Convert first hash to binary vector
    for (int i = 0; i < VECTOR_SIZE; i++) {
        binary_out[i] = (float)((first_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
    
    // Second SHA256 for noise
    crypto_hash_sha256(second_hash, first_hash, crypto_hash_sha256_BYTES);
    
    // Convert second hash to noise vector
    for (int i = 0; i < NOISE_SIZE; i++) {
        noise_out[i] = (int8_t)((second_hash[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void compute_binary_and_noise_batch(const uint8_t *inputs, float *binary_out, float *noise_out, int batch_size) {
    #pragma omp parallel for
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
        
        // Write sequential nonce to last 8 bytes
        uint64_t nonce = start_nonce + i;
        for (int j = 0; j < 8; j++) {
            outputs[(i * INPUT_SIZE) + (INPUT_SIZE - 8) + j] = (nonce >> (8 * (7 - j))) & 0xFF;
        }
    }
}

int count_trailing_zeros(MLMultiArray *output, NSInteger row) {
    int zeros = 0;
    
    // Check each bit from the end until we find a 1
    for (NSInteger i = 255; i >= 0; i--) {
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
        
        free(input_batch);
        
        self.featureNames = [NSSet setWithArray:@[@"input", @"noise"]];
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
        if (argc != 2) {
            NSLog(@"Usage: %s <difficulty>", argv[0]);
            return 1;
        }
        
        if (sodium_init() < 0) {
            NSLog(@"Error initializing libsodium");
            return 1;
        }
        
        // Parse difficulty
        int target_difficulty = atoi(argv[1]);
        
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
        NSLog(@"Model compiled successfully");
        
        MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        NSLog(@"Model loaded successfully");
        NSLog(@"Mining with target difficulty: %d leading zeros", target_difficulty);
        
        // Mining parameters
        //NSInteger batchSize = 8192;
        NSInteger batchSize = 1;
        uint64_t nonce = 0;
        uint64_t totalHashes = 0;
        NSDate *startTime = [NSDate date];
        int best_difficulty = 0;
        
        // Status display timer - 1 second intervals
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        
        dispatch_source_set_event_handler(timer, ^{
            NSTimeInterval elapsed = -[startTime timeIntervalSinceNow];
            double hashrate = totalHashes / elapsed;
            NSLog(@"Nonce: %llu | Hashrate: %.2f H/s | Best difficulty: %d leading zeros", 
                  nonce, hashrate, best_difficulty);
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
                //MLFeatureValue *outputFeature = [output featureValueForName:@"clip_63"];
                MLFeatureValue *outputFeature = [output featureValueForName:@"clip_0"];
                if (!outputFeature) {
                    NSLog(@"Could not find output feature");
                    continue;
                }
                
                MLMultiArray *outputArray = [outputFeature multiArrayValue];
                
                // Print binary vectors for debugging
                /*NSMutableString *inputStr = [NSMutableString stringWithString:@"input: ["];
                NSMutableString *noiseStr = [NSMutableString stringWithString:@"noise: ["];
                NSMutableString *outputStr = [NSMutableString stringWithString:@"output: ["];
                
                // Build input string
                for (NSInteger j = 0; j < 256; j++) {
                    [inputStr appendFormat:@"%d", [inputProvider.input[j] intValue]];
                    if (j < 255) [inputStr appendString:@" "];
                }
                [inputStr appendString:@"]\n"];
                
                // Build noise string
                for (NSInteger j = 0; j < 256; j++) {
                    [noiseStr appendFormat:@"%d", [inputProvider.noise[j] intValue]];
                    if (j < 255) [noiseStr appendString:@" "];
                }
                [noiseStr appendString:@"]\n"];
                
                // Build output string
                for (NSInteger j = 0; j < 256; j++) {
                    [outputStr appendFormat:@"%d", [outputArray[j] intValue]];
                    if (j < 255) [outputStr appendString:@" "];
                }
                [outputStr appendString:@"]\n"];
                
                NSLog(@"%@%@%@", inputStr, noiseStr, outputStr);*/
                
                // Check each output in batch
                for (NSInteger i = 0; i < batchSize; i++) {
                    int zeros = count_trailing_zeros(outputArray, i);
                    if (zeros > best_difficulty) {
                        best_difficulty = zeros;
                    }
                    
                    if (zeros >= target_difficulty) {
                        // Found a solution!
                        uint64_t solution_nonce = nonce + i;
                        NSLog(@"\nSolution found!");
                        NSLog(@"Nonce: %llu", solution_nonce);
                        NSLog(@"Leading zeros: %d", zeros);
                        
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
                            NSInteger byteIndex = 31 - (j / 8); // Store in reverse byte order
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
