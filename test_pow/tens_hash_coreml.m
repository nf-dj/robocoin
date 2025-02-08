#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

void printMLMultiArrayInfo(MLMultiArray *array, const char *name) {
    NSLog(@"%s info:", name);
    NSLog(@"  Shape: %@", array.shape);
    NSLog(@"  Strides: %@", array.strides);
    NSLog(@"  Data type: %ld", (long)array.dataType);
    
    float *ptr = (float *)array.dataPointer;
    NSLog(@"  First 5 values:");
    for (int i = 0; i < 5; i++) {
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
        
        // Create input tensors
        NSLog(@"Creating MLMultiArrays...");
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @256]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                               error:&error];
        if (error) {
            NSLog(@"Error creating input array: %@", error);
            return 1;
        }
        
        MLMultiArray *noiseArray = [[MLMultiArray alloc] initWithShape:@[@1, @256]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                               error:&error];
        if (error) {
            NSLog(@"Error creating noise array: %@", error);
            return 1;
        }
        
        // Fill input with binary (0,1) and noise with integers
        float *inputPtr = (float *)inputArray.dataPointer;
        float *noisePtr = (float *)noiseArray.dataPointer;
        for (int i = 0; i < 256; i++) {
            inputPtr[i] = (arc4random() % 2) ? 1.0f : 0.0f;
            // Generate noise between -64 and 64
            noisePtr[i] = (float)(arc4random() % 129) - 64;
        }
        
        NSLog(@"Input arrays created");
        printMLMultiArrayInfo(inputArray, "Input array");
        printMLMultiArrayInfo(noiseArray, "Noise array");
        
        // Create input features
        NSLog(@"Creating input features...");
        MLDictionaryFeatureProvider *inputFeatures = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{
                @"input": inputArray,
                @"noise": noiseArray
            } error:&error];
        
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
        
        float *outputPtr = (float *)outputArray.dataPointer;
        NSLog(@"First 10 output values:");
        for (int i = 0; i < 10; i++) {
            NSLog(@"%d: %f", i, outputPtr[i]);
        }
        
        NSLog(@"Program completed successfully");
    }
    return 0;
}
