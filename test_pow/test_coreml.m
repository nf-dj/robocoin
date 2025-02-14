#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>
#import <sodium.h>

// Debug flag
static BOOL debugMode = NO;

#define INPUT_SIZE 32         // 32 bytes per nonce input
#define VECTOR_SIZE 256       // 256 bits per sample (32 bytes * 8)
#define HIDDEN_SIZE 1024
#define BATCH_SIZE 2048
#define ROUNDS 64

// Compute binary vector from a 32-byte input to a 256-element float vector.
void compute_binary_vector(const uint8_t *input, float *binary_out) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        binary_out[i] = (float)((input[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void generate_sequential_nonces(uint8_t *outputs, uint64_t start_nonce, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        // Zero out the full input (32 bytes per sample)
        memset(outputs + (i * INPUT_SIZE), 0, INPUT_SIZE);
        // Write sequential nonce to the last 8 bytes (LSB)
        uint64_t nonce = start_nonce + i;
        for (int j = 0; j < 8; j++) {
            outputs[(i * INPUT_SIZE) + (24 + j)] = (nonce >> (8 * (7 - j))) & 0xFF;
        }
    }
}

// Helper function to print vectors (assumes contiguous array of given size)
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

// For an MLMultiArray with shape [256, BATCH_SIZE] stored in C-order,
// the element for sample 'col' and bit 'row' is at offset: row * BATCH_SIZE + col.
int count_leading_zeros(MLMultiArray *output, NSInteger sampleIndex) {
    int zeros = 0;
    for (NSInteger j = 0; j < VECTOR_SIZE; j++) {
        NSInteger offset = j * BATCH_SIZE + sampleIndex;
        if ([output[offset] intValue] == 0) {
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
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@end

@implementation InputFeatureProvider
- (instancetype)initWithBatchSize:(NSInteger)batchSize startNonce:(uint64_t)startNonce {
    self = [super init];
    if (self) {
        NSError *error = nil;
        
        // Create input array with shape [256, batchSize] (each column is one sample)
        self.input = [[MLMultiArray alloc] initWithShape:@[@(VECTOR_SIZE), @(batchSize)]
                                                 dataType:MLMultiArrayDataTypeFloat32
                                                    error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error);
            return nil;
        }
        
        // Allocate temporary memory for input batch (32 bytes per sample)
        uint8_t *input_batch = (uint8_t *)malloc(batchSize * INPUT_SIZE);
        if (!input_batch) {
            NSLog(@"Memory allocation error");
            return nil;
        }
        generate_sequential_nonces(input_batch, startNonce, (int)batchSize);
        
        // Get pointer to MLMultiArray data
        float *binary_data = (float *)self.input.dataPointer;
        
        // For each sample, compute the binary vector and store it into the appropriate column.
        for (int b = 0; b < batchSize; b++) {
            uint8_t *sample_input = input_batch + (b * INPUT_SIZE);
            float sample_binary[VECTOR_SIZE] = {0};
            compute_binary_vector(sample_input, sample_binary);
            // Write the computed 256-bit vector into column b.
            for (int i = 0; i < VECTOR_SIZE; i++) {
                binary_data[i * batchSize + b] = sample_binary[i];
            }
        }
        
        // Debug output for the first sample (column 0)
        if (debugMode) {
            printf("Nonce: ");
            for (int i = 0; i < INPUT_SIZE; i++) {
                printf("%02x", input_batch[i]);
            }
            printf("\n");
            
            float first_input[VECTOR_SIZE];
            for (int i = 0; i < VECTOR_SIZE; i++) {
                first_input[i] = binary_data[i * batchSize + 0];
            }
            print_vector("First input vector", first_input, VECTOR_SIZE);
        }
        
        free(input_batch);
        
        self.featureNames = [NSSet setWithArray:@[@"input"]];
    }
    return self;
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.input];
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
        
        NSInteger batchSize = BATCH_SIZE;
        __block uint64_t nonce = 0;
        __block uint64_t totalHashes = 0;
        __block NSDate *startTime = [NSDate date];
        __block int best_difficulty = 0;
        
        // Status display timer - 1 second intervals
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                                         dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        dispatch_source_set_event_handler(timer, ^{
            NSTimeInterval elapsed = -[startTime timeIntervalSinceNow];
            double hashrate = totalHashes / elapsed;
            double tops = (hashrate * (ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * VECTOR_SIZE * 4)) / 1e12;
            NSLog(@"Nonce: %llu | Hashrate: %.2f H/s | TOPS: %.2f | Best difficulty: %d",
                  nonce, hashrate, tops, best_difficulty);
        });
        dispatch_resume(timer);
        
        // Mining loop
        while (true) {
            @autoreleasepool {
                InputFeatureProvider *inputProvider = [[InputFeatureProvider alloc] initWithBatchSize:batchSize startNonce:nonce];
                if (!inputProvider) {
                    NSLog(@"Error creating input provider");
                    continue;
                }
                
                id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
                if (error) {
                    NSLog(@"Error during inference: %@", error);
                    continue;
                }
                
                // Get output feature (ensure the feature name matches your model)
                MLFeatureValue *outputFeature = [output featureValueForName:@"clip_65"];
                if (!outputFeature) {
                    NSLog(@"Could not find output feature");
                    continue;
                }
                
                MLMultiArray *outputArray = [outputFeature multiArrayValue];
                
                if (debugMode) {
                    float first_output[VECTOR_SIZE];
                    for (int j = 0; j < VECTOR_SIZE; j++) {
                        first_output[j] = [outputArray[j * batchSize + 0] floatValue];
                    }
                    print_vector("First output vector", first_output, VECTOR_SIZE);
                    
                    uint8_t first_output_bytes[32] = {0};
                    for (NSInteger j = 0; j < VECTOR_SIZE; j++) {
                        NSInteger bitIndex = 7 - (j % 8);
                        NSInteger byteIndex = j / 8;
                        first_output_bytes[byteIndex] |= (([outputArray[j * batchSize + 0] intValue] & 1) << bitIndex);
                    }
                    NSMutableString *outputHex = [NSMutableString string];
                    for (int j = 0; j < 32; j++) {
                        [outputHex appendFormat:@"%02x", first_output_bytes[j]];
                    }
                    NSLog(@"First output (hex): %@", outputHex);
                }
                
                // Check each sample (column) in the batch
                for (NSInteger i = 0; i < batchSize; i++) {
                    int zeros = count_leading_zeros(outputArray, i);
                    if (zeros > best_difficulty) {
                        best_difficulty = zeros;
                    }
                    
                    if (zeros >= target_difficulty) {
                        uint64_t solution_nonce = nonce + i;
                        NSLog(@"\nSolution found!");
                        NSLog(@"Nonce: %llu", solution_nonce);
                        NSLog(@"Leading zeros: %d", zeros);
                        
                        NSMutableString *inputHex = [NSMutableString string];
                        for (int j = 0; j < 24; j++) {
                            [inputHex appendString:@"00"];
                        }
                        for (int j = 0; j < 8; j++) {
                            [inputHex appendFormat:@"%02x", (uint8_t)((solution_nonce >> (8 * (7 - j))) & 0xFF)];
                        }
                        NSLog(@"Solution input (hex): %@", inputHex);
                        
                        NSMutableString *outputHex = [NSMutableString string];
                        uint8_t output_bytes[32] = {0};
                        for (NSInteger j = 0; j < VECTOR_SIZE; j++) {
                            NSInteger bitIndex = 7 - (j % 8);
                            NSInteger byteIndex = j / 8;
                            output_bytes[byteIndex] |= (([outputArray[j * batchSize + i] intValue] & 1) << bitIndex);
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

