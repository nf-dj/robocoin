#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <QuartzCore/QuartzCore.h>

#define NUM_TESTS 1000000
#define VECTOR_SIZE 256

void print_performance(uint64_t ops, double duration) {
    double tops = (ops * 2.0) / (duration * 1e12);  // *2 for multiply-add
    printf("Time: %.3f seconds\n", duration);
    printf("Ops: %llu\n", ops);
    printf("Performance: %.2f TOPS\n", tops);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSError *error = nil;
        
        // Load model
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        NSString *modelPath = @"tens_hash.mlmodelc";
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL 
                                         configuration:config
                                               error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        printf("Model loaded successfully\n");
        
        // Create input tensor once
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @VECTOR_SIZE]
                                                           dataType:MLMultiArrayDataTypeFloat32
                                                              error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error);
            return 1;
        }
        
        // Fill with random binary values
        float *inputPtr = (float *)inputArray.dataPointer;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            inputPtr[i] = (arc4random() % 2) ? 1.0f : 0.0f;
        }
        
        // Create input features once
        MLDictionaryFeatureProvider *inputFeatures = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{@"input": inputArray} error:&error];
        
        printf("Starting %d matrix multiplications...\n", NUM_TESTS);
        uint64_t total_ops = (uint64_t)NUM_TESTS * VECTOR_SIZE * VECTOR_SIZE;
        
        // Time the matrix multiplications
        CFTimeInterval startTime = CACurrentMediaTime();
        
        for(int i = 0; i < NUM_TESTS; i++) {
            @autoreleasepool {
                id<MLFeatureProvider> outputFeatures = [model predictionFromFeatures:inputFeatures 
                                                                            error:&error];
                if (error) {
                    NSLog(@"Error running inference: %@", error);
                    return 1;
                }
                
                if (i % 100000 == 0) {
                    printf("Completed %d tests\n", i);
                }
            }
        }
        
        CFTimeInterval duration = CACurrentMediaTime() - startTime;
        print_performance(total_ops, duration);
    }
    return 0;
}