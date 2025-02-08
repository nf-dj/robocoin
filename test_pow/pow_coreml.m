#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <pthread.h>

// Constants
const NSUInteger BATCH_SIZE = 8192;
const uint64_t OPS_PER_HASH = 256 * 256 * 2 * 64;

// Progress tracking
uint64_t totalAttempts = 0;
uint32_t bestBits = 0;
NSDate *startTime;
volatile bool shouldStop = false;

@interface NSData (HexString)
+ (NSData *)dataWithHexString:(NSString *)hexString;
@end

@implementation NSData (HexString)
+ (NSData *)dataWithHexString:(NSString *)hexString {
    const char *chars = [hexString UTF8String];
    NSUInteger length = strlen(chars);
    unsigned char *bytes = malloc(length / 2);
    if (!bytes) return nil;
    
    for (NSUInteger i = 0; i < length; i += 2) {
        char high = chars[i], low = chars[i + 1];
        high = (high >= 'a') ? (high - 'a' + 10) : ((high >= 'A') ? (high - 'A' + 10) : (high - '0'));
        low = (low >= 'a') ? (low - 'a' + 10) : ((low >= 'A') ? (low - 'A' + 10) : (low - '0'));
        bytes[i/2] = (high << 4) | low;
    }
    
    NSData *data = [NSData dataWithBytes:bytes length:length/2];
    free(bytes);
    return data;
}
@end

uint32_t countLeadingZeroBits(const uint8_t *bytes, size_t length) {
    uint32_t count = 0;
    for (size_t i = 0; i < length && bytes[i] == 0; i++) count += 8;
    if (count < length * 8) {
        uint8_t b = bytes[count / 8];
        while ((b & 0x80) == 0) {
            count++;
            b <<= 1;
        }
    }
    return count;
}

void *progressMonitorThread(void *arg) {
    while (!shouldStop) {
        NSTimeInterval totalTime = [[NSDate date] timeIntervalSinceDate:startTime];
        double hashRate = totalAttempts / totalTime;
        double tOps = (hashRate * OPS_PER_HASH) / 1e12;
        printf("T:%4.0fs H:%7.0f/s TOPS:%6.3f Tot:%12llu Best:%3u\r",
               totalTime, hashRate, tOps, totalAttempts, bestBits);
        fflush(stdout);
        [NSThread sleepForTimeInterval:1.0];
    }
    return NULL;
}

void printMLMultiArray(MLMultiArray *array, const char *name, int limit) {
    printf("%s: shape=%s count=%lu\n", name, array.shape.description.UTF8String, (unsigned long)array.count);
    float *data = (float *)array.dataPointer;
    printf("First %d values: ", limit);
    for (int i = 0; i < MIN(limit, (int)array.count); i++) {
        printf("%.1f ", data[i]);
    }
    printf("\n");
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            printf("Usage: %s <target-hex>\n", argv[0]);
            return 1;
        }
        
        NSString *targetHex = [NSString stringWithUTF8String:argv[1]];
        if (targetHex.length != 64) {
            printf("Error: Target must be 32 bytes (64 hex chars)\n");
            return 1;
        }
        
        NSData *targetData = [NSData dataWithHexString:targetHex];
        const uint8_t *targetBytes = targetData.bytes;
        
        printf("Loading model...\n");
        NSError *error = nil;
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@"PowModel.mlmodelc"]
                                          configuration:config error:&error];
        if (error) {
            printf("Error loading model: %s\n", error.description.UTF8String);
            return 1;
        }
        
        // Print model description in detail
        MLModelDescription *modelDesc = model.modelDescription;
        printf("\nModel Input Descriptions:\n");
        for (NSString *key in modelDesc.inputDescriptionsByName) {
            MLFeatureDescription *desc = modelDesc.inputDescriptionsByName[key];
            printf("%s: %s\n", key.UTF8String, desc.description.UTF8String);
        }
        
        printf("\nModel Output Descriptions:\n");
        for (NSString *key in modelDesc.outputDescriptionsByName) {
            MLFeatureDescription *desc = modelDesc.outputDescriptionsByName[key];
            printf("%s: %s\n", key.UTF8String, desc.description.UTF8String);
        }
        
        NSArray *inputShape = @[@(BATCH_SIZE), @256];
        printf("\nCreating input arrays with shape: %s\n", inputShape.description.UTF8String);
        
        MLMultiArray *binaryInput = [[MLMultiArray alloc] initWithShape:inputShape
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            printf("Error creating binary input: %s\n", error.description.UTF8String);
            return 1;
        }
        
        MLMultiArray *noiseInput = [[MLMultiArray alloc] initWithShape:inputShape
                                                             dataType:MLMultiArrayDataTypeFloat32
                                                                error:&error];
        if (error) {
            printf("Error creating noise input: %s\n", error.description.UTF8String);
            return 1;
        }
        
        float *binaryPtr = (float *)binaryInput.dataPointer;
        float *noisePtr = (float *)noiseInput.dataPointer;
        
        startTime = [NSDate date];
        pthread_t progressThread;
        pthread_create(&progressThread, NULL, progressMonitorThread, NULL);
        
        printf("\nStarting inference loop...\n");
        int debugCounter = 0;
        NSString *outputKey = nil;
        
        // Try to find output key name from model description
        for (NSString *key in modelDesc.outputDescriptionsByName) {
            outputKey = key;  // Just take the first (and should be only) output key
            break;
        }
        
        if (!outputKey) {
            printf("Error: Could not find output key in model description\n");
            return 1;
        }
        printf("Using output key: %s\n", outputKey.UTF8String);
        
        while (!shouldStop) {
            @autoreleasepool {
                // Generate random inputs
                for (NSUInteger i = 0; i < BATCH_SIZE * 256; i++) {
                    binaryPtr[i] = (float)(arc4random_uniform(2));
                    noisePtr[i] = ((float)arc4random_uniform(1000) / 1000.0f) * 0.1f;
                }
                
                if (debugCounter < 3) {
                    printf("\nDebug iteration %d:\n", debugCounter);
                    printMLMultiArray(binaryInput, "Binary Input", 10);
                    printMLMultiArray(noiseInput, "Noise Input", 10);
                }
                
                // Run inference
                NSDictionary *inputDict = @{
                    @"binary_input": binaryInput,
                    @"noise_input": noiseInput
                };
                
                MLDictionaryFeatureProvider *inputFeatures = [[MLDictionaryFeatureProvider alloc] 
                    initWithDictionary:inputDict error:&error];
                    
                if (error) {
                    printf("\nError creating feature provider: %s\n", error.description.UTF8String);
                    continue;
                }
                
                id<MLFeatureProvider> outputFeatures = [model predictionFromFeatures:inputFeatures error:&error];
                if (error) {
                    printf("\nInference error: %s\n", error.description.UTF8String);
                    continue;
                }
                
                MLFeatureValue *outputValue = [outputFeatures featureValueForName:outputKey];
                if (!outputValue) {
                    printf("\nNo output value for key %s\n", outputKey.UTF8String);
                    continue;
                }
                
                MLMultiArray *outputArray = outputValue.multiArrayValue;
                if (!outputArray) {
                    printf("\nOutput value exists but no multiarray\n");
                    continue;
                }
                
                if (debugCounter < 3) {
                    printMLMultiArray(outputArray, "Output", 10);
                    debugCounter++;
                }
                
                float *outputPtr = (float *)outputArray.dataPointer;
                
                for (NSUInteger i = 0; i < BATCH_SIZE; i++) {
                    uint8_t outputBytes[32] = {0};
                    for (NSUInteger j = 0; j < 256; j++) {
                        if (outputPtr[i * 256 + j] > 0.5) {
                            outputBytes[j / 8] |= (1 << (7 - (j % 8)));
                        }
                    }
                    
                    uint32_t zeros = countLeadingZeroBits(outputBytes, 32);
                    if (zeros > bestBits) {
                        bestBits = zeros;
                    }
                    
                                            if (memcmp(outputBytes, targetBytes, 32) < 0) {
                        shouldStop = true;
                        printf("\nSolution found!\n");
                        
                        // Print input in hex
                        printf("Input hex:  ");
                        uint8_t inputBytes[32] = {0};
                        for (NSUInteger j = 0; j < 256; j++) {
                            if (binaryPtr[i * 256 + j] > 0.5) {
                                inputBytes[j / 8] |= (1 << (7 - (j % 8)));
                            }
                        }
                        for (int j = 0; j < 32; j++) {
                            printf("%02x", inputBytes[j]);
                        }
                        
                        // Print output in hex
                        printf("\nOutput hex: ");
                        for (int j = 0; j < 32; j++) {
                            printf("%02x", outputBytes[j]);
                        }
                        printf("\n");
                        pthread_join(progressThread, NULL);
                        return 0;
                    }
                }
                totalAttempts += BATCH_SIZE;
            }
        }
        pthread_join(progressThread, NULL);
    }
    return 0;
}
