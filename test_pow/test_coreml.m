#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>

@protocol MLFeatureProvider;  // Forward declaration

// MLFeatureProvider implementation for our inputs
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
        
        // Fill arrays with random binary values (0 or 1)
        for (NSInteger i = 0; i < batchSize * 256; i++) {
            self.input[i] = @(arc4random_uniform(2));  // 0 or 1
            self.bias[i] = @(arc4random_uniform(2));   // 0 or 1
        }
        
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
        // Parse command line arguments
        NSInteger batchSize = 8192;
        NSInteger numInferences = 1000;
        NSString *modelPath = @"test_coreml.mlpackage";
        
        // Load and compile model
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSError *error = nil;
        
        // First compile the model
        NSURL *compiledUrl = [MLModel compileModelAtURL:modelURL error:&error];
        if (error) {
            NSLog(@"Error compiling model: %@", error);
            return 1;
        }
        NSLog(@"Model compiled successfully");
        
        // Then load the compiled model
        MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl error:&error];
        if (error) {
            NSLog(@"Error loading model: %@", error);
            return 1;
        }
        NSLog(@"Model loaded successfully");
        
        // Create input provider
        InputFeatureProvider *inputProvider = [[InputFeatureProvider alloc] initWithBatchSize:batchSize];
        if (!inputProvider) {
            NSLog(@"Error creating input provider");
            return 1;
        }
        
        // Performance metrics
        NSInteger numOperations = (64 * 256 * 256 * 2 + 3 * 256) * batchSize;
        NSMutableArray<NSNumber *> *inferenceTimes = [NSMutableArray array];
        MLMultiArray *lastOutput = nil;
        
        // Status display dispatch source
        dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC, 0);
        
        // Status display handler
        dispatch_source_set_event_handler(timer, ^{
            if (inferenceTimes.count > 0) {
                // Calculate average inference time
                NSTimeInterval totalTime = 0;
                for (NSNumber *time in inferenceTimes) {
                    totalTime += time.doubleValue;
                }
                NSTimeInterval avgTime = totalTime / inferenceTimes.count;
                double tops = (numOperations / avgTime) / 1e12;
                
                // Print performance metrics
                NSLog(@"Average inference time: %.6f seconds | Estimated TOPS: %.6f", avgTime, tops);
                
                // Print last output values (first row)
                if (lastOutput) {
                    NSMutableString *outputStr = [NSMutableString string];
                    for (NSInteger i = 0; i < 256; i++) {
                        [outputStr appendFormat:@"%.4f ", [lastOutput[i] floatValue]];
                    }
                    NSLog(@"Last output (size=256): %@", outputStr);
                }
            }
        });
        
        // Start the timer
        dispatch_resume(timer);
        
        // Warm-up run
        [model predictionFromFeatures:inputProvider error:&error];
        if (error) {
            NSLog(@"Error during warm-up: %@", error);
            return 1;
        }
        
        // Main inference loop
        for (NSInteger i = 0; i < numInferences; i++) {
            NSDate *startTime = [NSDate date];
            
            id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
            if (error) {
                NSLog(@"Error during inference: %@", error);
                continue;
            }
            
            NSTimeInterval inferenceTime = -[startTime timeIntervalSinceNow];
            [inferenceTimes addObject:@(inferenceTime)];
            
            // Store last output
            lastOutput = [(MLFeatureValue *)[output featureValueForName:@"output"] multiArrayValue];
        }
        
        // Cancel the timer
        dispatch_source_cancel(timer);
        
        // Final performance metrics
        if (inferenceTimes.count > 0) {
            NSTimeInterval totalTime = 0;
            for (NSNumber *time in inferenceTimes) {
                totalTime += time.doubleValue;
            }
            NSTimeInterval avgTime = totalTime / inferenceTimes.count;
            double tops = (numOperations / avgTime) / 1e12;
            NSLog(@"Final Average inference time: %.6f seconds | Final Estimated TOPS: %.6f", avgTime, tops);
        }
    }
    return 0;
}
