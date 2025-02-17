#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>
#import <sodium.h>

// Debug flag
static BOOL debugMode = NO;

#define INPUT_SIZE 32
#define VECTOR_SIZE 256
#define HIDDEN_SIZE 1024
#define BATCH_SIZE 2048
#define ROUNDS 64

// Helper functions unchanged
void compute_binary_vector(const uint8_t *input, float *binary_out) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        binary_out[i] = (float)((input[i / 8] >> (7 - (i % 8))) & 1);
    }
}

void generate_sequential_nonces(uint8_t *outputs, uint64_t start_nonce, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        memset(outputs + (i * INPUT_SIZE), 0, INPUT_SIZE);
        uint64_t nonce = start_nonce + i;
        for (int j = 0; j < 8; j++) {
            outputs[(i * INPUT_SIZE) + (24 + j)] = (nonce >> (8 * (7 - j))) & 0xFF;
        }
    }
}

// Modified to output JSON-RPC format
void print_jsonrpc_notification(NSString *method, NSDictionary *params) {
    NSDictionary *jsonrpc = @{
        @"jsonrpc": @"2.0",
        @"method": method,
        @"params": params
    };
    
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:jsonrpc options:0 error:&error];
    if (!error) {
        NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        printf("%s\n", [jsonString UTF8String]);
        fflush(stdout);
    }
}

void print_jsonrpc_error(NSInteger code, NSString *message, id data) {
    NSDictionary *error = @{
        @"code": @(code),
        @"message": message,
        @"data": data ? data : [NSNull null]
    };
    
    NSDictionary *jsonrpc = @{
        @"jsonrpc": @"2.0",
        @"error": error,
        @"id": [NSNull null]
    };
    
    NSError *jsonError = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:jsonrpc options:0 error:&jsonError];
    if (!jsonError) {
        NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        printf("%s\n", [jsonString UTF8String]);
        fflush(stdout);
    }
}

// Keep existing InputFeatureProvider implementation unchanged
@interface InputFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong) MLMultiArray *input;
@property (nonatomic, strong) NSSet<NSString *> *featureNames;
@end

@implementation InputFeatureProvider
- (instancetype)initWithBatchSize:(NSInteger)batchSize startNonce:(uint64_t)startNonce {
    self = [super init];
    if (self) {
        NSError *error = nil;
        
        self.input = [[MLMultiArray alloc] initWithShape:@[@(VECTOR_SIZE), @(batchSize)]
                                                 dataType:MLMultiArrayDataTypeFloat32
                                                    error:&error];
        if (error) {
            print_jsonrpc_error(-32603, @"Error creating input array", error.localizedDescription);
            return nil;
        }
        
        uint8_t *input_batch = (uint8_t *)malloc(batchSize * INPUT_SIZE);
        if (!input_batch) {
            print_jsonrpc_error(-32603, @"Memory allocation error", nil);
            return nil;
        }
        generate_sequential_nonces(input_batch, startNonce, (int)batchSize);
        
        float *binary_data = (float *)self.input.dataPointer;
        
        for (int b = 0; b < batchSize; b++) {
            uint8_t *sample_input = input_batch + (b * INPUT_SIZE);
            float sample_binary[VECTOR_SIZE] = {0};
            compute_binary_vector(sample_input, sample_binary);
            for (int i = 0; i < VECTOR_SIZE; i++) {
                binary_data[i * batchSize + b] = sample_binary[i];
            }
        }
        
        if (debugMode) {
            NSMutableString *nonceHex = [NSMutableString string];
            for (int i = 0; i < INPUT_SIZE; i++) {
                [nonceHex appendFormat:@"%02x", input_batch[i]];
            }
            print_jsonrpc_notification(@"debug", @{@"nonce": nonceHex});
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

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            print_jsonrpc_error(-32602, @"Invalid params", [NSString stringWithFormat:@"Usage: %s [-d] <difficulty>", argv[0]]);
            return 1;
        }
        
        if (sodium_init() < 0) {
            print_jsonrpc_error(-32603, @"Internal error", @"Error initializing libsodium");
            return 1;
        }

        int argIndex = 1;
        if (strcmp(argv[argIndex], "-d") == 0) {
            debugMode = YES;
            argIndex++;
        }
        
        int target_difficulty = atoi(argv[argIndex]);
        
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        NSString *modelPath = @"tens_hash.mlpackage";
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSError *error = nil;
        
        NSURL *compiledUrl = [MLModel compileModelAtURL:modelURL error:&error];
        if (error) {
            print_jsonrpc_error(-32603, @"Model compilation error", error.localizedDescription);
            return 1;
        }
        
        MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error) {
            print_jsonrpc_error(-32603, @"Model loading error", error.localizedDescription);
            return 1;
        }

        NSInteger batchSize = BATCH_SIZE;
        __block uint64_t nonce = 0;
        __block uint64_t totalHashes = 0;
        __block NSDate *startTime = [NSDate date];
        __block int best_difficulty = 0;
        
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                                       dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        dispatch_source_set_event_handler(timer, ^{
            NSTimeInterval elapsed = -[startTime timeIntervalSinceNow];
            double hashrate = totalHashes / elapsed;
            double tops = (hashrate * (ROUNDS * HIDDEN_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * VECTOR_SIZE * 4)) / 1e12;
            
            print_jsonrpc_notification(@"status", @{
                @"nonce": @(nonce),
                @"hashrate": @(hashrate),
                @"tops": @(tops),
                @"best_difficulty": @(best_difficulty)
            });
        });
        dispatch_resume(timer);
        
        while (true) {
            @autoreleasepool {
                InputFeatureProvider *inputProvider = [[InputFeatureProvider alloc] initWithBatchSize:batchSize startNonce:nonce];
                if (!inputProvider) {
                    continue;
                }
                
                id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
                if (error) {
                    continue;
                }
                
                MLFeatureValue *outputFeature = [output featureValueForName:@"clip_65"];
                if (!outputFeature) {
                    continue;
                }
                
                MLMultiArray *outputArray = [outputFeature multiArrayValue];
                
                for (NSInteger i = 0; i < batchSize; i++) {
                    int zeros = count_leading_zeros(outputArray, i);
                    if (zeros > best_difficulty) {
                        best_difficulty = zeros;
                    }
                    
                    if (zeros >= target_difficulty) {
                        uint64_t solution_nonce = nonce + i;
                        
                        NSMutableString *inputHex = [NSMutableString string];
                        for (int j = 0; j < 24; j++) {
                            [inputHex appendString:@"00"];
                        }
                        for (int j = 0; j < 8; j++) {
                            [inputHex appendFormat:@"%02x", (uint8_t)((solution_nonce >> (8 * (7 - j))) & 0xFF)];
                        }
                        
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
                        
                        print_jsonrpc_notification(@"solution", @{
                            @"nonce": @(solution_nonce),
                            @"leading_zeros": @(zeros),
                            @"input_hex": inputHex,
                            @"output_hex": outputHex
                        });
                        
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