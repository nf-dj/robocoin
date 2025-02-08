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
        noise_out[i] = (float)((second_hash[i % 32] >> (i % 8)) & 1);
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

@protocol MLFeatureProvider;

@interface InputFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong) MLMultiArray *input;
@property (nonatomic, strong) MLMultiArray *bias;
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@end

@implementation InputFeatureProvider
- (instancetype)initWithBatchSize:(NSInteger)batchSize {
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
        
        self.bias = [[MLMultiArray alloc] initWithShape:@[@(batchSize), @256]
                                              dataType:MLMultiArrayDataTypeFloat32
                                                error:&error];
        if (error) {
            NSLog(@"Error creating bias array: %@", error);
            return nil;
        }
        
        // Generate random input batch
        uint8_t *random_inputs = (uint8_t *)malloc(batchSize * INPUT_SIZE);
        for (NSInteger i = 0; i < batchSize * INPUT_SIZE; i++) {
            random_inputs[i] = arc4random_uniform(256);
        }
        
        // Get pointers to MLMultiArray data
        float *binary_data = (float *)self.input.dataPointer;
        float *noise_data = (float *)self.bias.dataPointer;
        
        // Generate binary and noise vectors
        compute_binary_and_noise_batch(random_inputs, binary_data, noise_data, (int)batchSize);
        
        free(random_inputs);
        
        self.featureNames = [NSSet setWithArray:@[@"input", @"bias"]];
    }
    return self;
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.input];
    } else if ([featureName isEqualToString:@"bias"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.bias];
    }
    return nil;
}
@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (sodium_init() < 0) {
            NSLog(@"Error initializing libsodium");
            return 1;
        }
        
        // Parse command line arguments
        NSInteger batchSize = 8192;
        NSInteger numInferences = 1000;
        NSString *modelPath = @"test_coreml.mlpackage";
        
        // Configure compute units
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        // Load and compile model
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
        
        // Performance metrics
        NSInteger numOperations = (64 * 256 * 256 * 2 + 3 * 256) * batchSize;
        NSMutableArray<NSNumber *> *inferenceTimes = [NSMutableArray array];
        __block MLMultiArray *lastOutput = nil;
        __block NSDate *lastOutputTime = [NSDate date];
        __block BOOL firstOutput = YES;
        
        // Status display dispatch source - 1 second for performance metrics
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        
        dispatch_source_set_event_handler(timer, ^{
            if (inferenceTimes.count > 0) {
                NSTimeInterval totalTime = 0;
                for (NSNumber *time in inferenceTimes) {
                    totalTime += time.doubleValue;
                }
                NSTimeInterval avgTime = totalTime / inferenceTimes.count;
                double tops = (numOperations / avgTime) / 1e12;
                
                // Print performance metrics every second
                NSLog(@"Average inference time: %.6f seconds | Estimated TOPS: %.6f", avgTime, tops);
                
                // Print last output values every 10 seconds
                NSTimeInterval timeSinceLastOutput = -[lastOutputTime timeIntervalSinceNow];
                if (lastOutput && timeSinceLastOutput >= 10.0) {
                    NSLog(@"\n=== Last Batch Output (last row) ===");
                    NSMutableString *outputStr = [NSMutableString string];
                    NSInteger startIdx = (batchSize - 1) * 256;
                    for (NSInteger i = 0; i < 256; i++) {
                        if (i % 16 == 0) {
                            [outputStr appendString:@"\n"];
                        }
                        [outputStr appendFormat:@"%d ", [lastOutput[startIdx + i] intValue]];
                    }
                    NSLog(@"%@\n", outputStr);
                    lastOutputTime = [NSDate date];
                }
            }
        });
        
        dispatch_resume(timer);
        
        // Main inference loop
        for (NSInteger i = 0; i < numInferences; i++) {
            // Create new input provider for each batch
            InputFeatureProvider *inputProvider = [[InputFeatureProvider alloc] initWithBatchSize:batchSize];
            if (!inputProvider) {
                NSLog(@"Error creating input provider");
                continue;
            }
            
            NSDate *startTime = [NSDate date];
            
            id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
            if (error) {
                NSLog(@"Error during inference: %@", error);
                continue;
            }
            
            // Debug: Print available feature names on first output
            if (firstOutput) {
                NSSet *featureNames = [output featureNames];
                NSLog(@"Available output feature names: %@", featureNames);
                firstOutput = NO;
            }
            
            // Try to get the output feature
            MLFeatureValue *outputFeature = [output featureValueForName:@"clip_63"];
            if (outputFeature) {
                lastOutput = [outputFeature multiArrayValue];
                if (i == 0) {  // Print shape info for first inference
                    NSLog(@"Output shape: %@", lastOutput.shape);
                }
            } else {
                NSLog(@"Could not find feature 'clip_63' in iteration %ld", (long)i);
            }
            
            NSTimeInterval inferenceTime = -[startTime timeIntervalSinceNow];
            [inferenceTimes addObject:@(inferenceTime)];
        }
        
        dispatch_source_cancel(timer);
        
        // Final performance metrics
        if (inferenceTimes.count > 0) {
            NSTimeInterval totalTime = 0;
            for (NSNumber *time in inferenceTimes) {
                totalTime += time.doubleValue;
            }
            NSTimeInterval avgTime = totalTime / inferenceTimes.count;
            double tops = (numOperations / avgTime) / 1e12;
            NSLog(@"\n=== Final Performance Metrics ===");
            NSLog(@"Final Average inference time: %.6f seconds", avgTime);
            NSLog(@"Final Estimated TOPS: %.6f", tops);
            
            if (lastOutput) {
                NSLog(@"\n=== Final Batch Output (last row) ===");
                NSMutableString *outputStr = [NSMutableString string];
                NSInteger startIdx = (batchSize - 1) * 256;
                for (NSInteger i = 0; i < 256; i++) {
                    if (i % 16 == 0) {
                        [outputStr appendString:@"\n"];
                    }
                    [outputStr appendFormat:@"%d ", [lastOutput[startIdx + i] intValue]];
                }
                NSLog(@"%@\n", outputStr);
            }
        }
    }
    return 0;
}
