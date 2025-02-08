#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

void printMLMultiArrayInfo(MLMultiArray *array, const char *name) {
    NSLog(@"%s info:", name);
    NSLog(@"  Shape: %@", array.shape);
    NSLog(@"  Strides: %@", array.strides);
    NSLog(@"  Data type: %ld", (long)array.dataType);
    
    float *ptr = (float *)array.dataPointer;
    NSLog(@"  All values:");
    for (int i = 0; i < 4; i++) {
        NSLog(@"    %d: %f", i, ptr[i]);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSError *error = nil;
        
        NSLog(@"Starting program...");
        
        // Configure model loading
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        NSString *modelPath = @"tens_hash.mlmodelc";
        NSLog(@"Checking for model at path: %@", modelPath);
        if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
            NSLog(@"Error: Model file not found at %@", modelPath);
            return 1;
        }
        
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSLog(@"Loading model...");
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL 
                                         configuration:config
                                               error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        NSLog(@"Model loaded successfully");
        
        // Create input tensor
        NSLog(@"Creating MLMultiArray...");
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @4]
                                                           dataType:MLMultiArrayDataTypeFloat32
                                                              error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error);
            return 1;
        }
        
        // Fill with binary test values [1,0,1,0]
        float *inputPtr = (float *)inputArray.dataPointer;
        inputPtr[0] = 1.0f;
        inputPtr[1] = 0.0f;
        inputPtr[2] = 1.0f;
        inputPtr[3] = 0.0f;
        
        NSLog(@"Input array created");
        printMLMultiArrayInfo(inputArray, "Input array");
        
        // Create input features
        NSLog(@"Creating input features...");
        MLDictionaryFeatureProvider *inputFeatures = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{@"input": inputArray} error:&error];
        
        if (error) {
            NSLog(@"Error creating input features: %@", error);
            return 1;
        }
        
        // Run inference
        NSLog(@"Starting inference...");
        id<MLFeatureProvider> outputFeatures = [model predictionFromFeatures:inputFeatures 
                                                                    error:&error];
        if (error) {
            NSLog(@"Error running inference: %@", error);
            return 1;
        }
        NSLog(@"Inference completed");
        
        // Get output
        MLMultiArray *outputArray = (MLMultiArray *)[outputFeatures featureValueForName:@"output"].multiArrayValue;
        NSLog(@"Output received");
        printMLMultiArrayInfo(outputArray, "Output array");
                
        NSLog(@"Program completed successfully");
    }
    return 0;
}