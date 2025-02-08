#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

// Helper function to print an MLMultiArray's shape, count, and values.
void printMLMultiArray(MLMultiArray *array, const char *name) {
    NSLog(@"%s:", name);
    NSLog(@"  Shape: %@", array.shape);
    NSLog(@"  Count: %lu", (unsigned long)array.count);
    
    float *data = (float *)array.dataPointer;
    NSMutableString *values = [NSMutableString stringWithString:@"["];
    for (NSUInteger i = 0; i < array.count; i++) {
        [values appendFormat:@"%f", data[i]];
        if (i < array.count - 1) {
            [values appendString:@", "];
        }
    }
    [values appendString:@"]"];
    NSLog(@"  Values: %@", values);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Set the path to the compiled Core ML model folder.
        NSString *modelPath = @"SimpleModel.mlmodelc";
        if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
            NSLog(@"Error: Model not found at path %@", modelPath);
            return 1;
        }
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSError *error = nil;
        
        // Create a configuration that uses all available compute units.
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        // Load the model.
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        NSLog(@"Model loaded successfully.");
        
        // Print model input descriptions.
        NSDictionary *inputDescriptions = model.modelDescription.inputDescriptionsByName;
        NSLog(@"Model Input Descriptions: %@", inputDescriptions);
        
        // The model expects a 2-D input feature named "input" with shape [1, 4].
        NSString *inputKey = @"input";
        
        // Create an MLMultiArray with shape [1, 4] (rank 2 array).
        NSArray *inputShape = @[@1, @4];
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:inputShape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                  error:&error];
        if (error) {
            NSLog(@"Error creating input MLMultiArray: %@", error);
            return 1;
        }
        
        // Fill the input array with values [1, 0, 1, 0].
        float *inputPtr = (float *)inputArray.dataPointer;
        inputPtr[0] = 1.0f;
        inputPtr[1] = 0.0f;
        inputPtr[2] = 1.0f;
        inputPtr[3] = 0.0f;
        NSLog(@"Input MLMultiArray created:");
        printMLMultiArray(inputArray, "Input Array");
        
        // Create a feature provider using the correct key.
        MLDictionaryFeatureProvider *inputFeatures = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputKey: inputArray} error:&error];
        if (error) {
            NSLog(@"Error creating input feature provider: %@", error);
            return 1;
        }
        
        // Run inference.
        id<MLFeatureProvider> outputFeatures = [model predictionFromFeatures:inputFeatures error:&error];
        if (error) {
            NSLog(@"Error during prediction: %@", error);
            return 1;
        }
        NSLog(@"Inference completed.");
        
        // Print model output descriptions.
        NSDictionary *outputDescriptions = model.modelDescription.outputDescriptionsByName;
        NSLog(@"Model Output Descriptions: %@", outputDescriptions);
        
        // The output key is "var_5" as indicated in the description.
        NSString *outputKey = @"var_5";
        MLFeatureValue *outputValue = [outputFeatures featureValueForName:outputKey];
        MLMultiArray *outputArray = outputValue.multiArrayValue;
        if (!outputArray) {
            NSLog(@"Error: Could not retrieve the output array for key '%@'.", outputKey);
            return 1;
        }
        
        NSLog(@"Output MLMultiArray:");
        printMLMultiArray(outputArray, "Output Array");
    }
    return 0;
}

