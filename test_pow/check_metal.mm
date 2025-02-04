#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char* argv[]) {
    @autoreleasepool {
        NSLog(@"Starting Metal check...");
        
        // Get all devices
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        NSLog(@"Available Metal devices: %@", devices);
        
        // Try to get default device first
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSLog(@"Default Metal device: %@", device);
        
        // If default device failed, try to use the first available device
        if (!device && devices.count > 0) {
            device = devices.firstObject;
            NSLog(@"Using first available device instead: %@", device);
        }
        
        if (!device) {
            NSLog(@"No Metal device available");
            return 1;
        }
        
        NSLog(@"Metal device name: %@", device.name);
        NSLog(@"Is device headless: %d", device.isHeadless);
        
        // Try to create a command queue to verify device works
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (commandQueue) {
            NSLog(@"Successfully created command queue");
        } else {
            NSLog(@"Failed to create command queue");
        }
        
        return 0;
    }
}