#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sodium.h>
#include <inttypes.h>

#define IN_SIZE 32
#define HIDDEN 256
#define ROUNDS 64
#define BATCH_SIZE 256  // Process multiple nonces in parallel
#define OPS_PER_HASH (256*256*64+32*256*2)

@interface TensPowMetal : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> hashPipelineState;
@property (nonatomic, strong) id<MTLBuffer> expandMatBuffer;
@property (nonatomic, strong) id<MTLBuffer> middleMatBuffers;
@property (nonatomic, strong) id<MTLBuffer> compressMatBuffer;

- (instancetype)initWithSeed:(uint8_t*)seed;
- (void)processNonceBatch:(const uint8_t*)nonces 
              noiseVectors:(const int8_t*)noiseVectors 
                    count:(int)count 
                  outputs:(uint8_t*)outputs;
@end

@implementation TensPowMetal

- (instancetype)initWithSeed:(uint8_t*)seed {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            NSLog(@"Metal is not supported on this device");
            return nil;
        }
        
        _commandQueue = [_device newCommandQueue];
        
        // Load Metal library and create pipeline state
        NSError *error = nil;
        id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
        if (!defaultLibrary) {
            NSLog(@"Failed to load Metal library");
            return nil;
        }
        
        id<MTLFunction> hashFunction = [defaultLibrary newFunctionWithName:@"tensor_hash_metal"];
        _hashPipelineState = [_device newComputePipelineStateWithFunction:hashFunction error:&error];
        
        if (!_hashPipelineState) {
            NSLog(@"Failed to create pipeline state: %@", error);
            return nil;
        }
        
        // Generate matrices using libsodium
        size_t total_size = (HIDDEN * IN_SIZE) + (ROUNDS * HIDDEN * HIDDEN) + (IN_SIZE * HIDDEN);
        uint8_t *data = (uint8_t*)malloc(total_size);
        unsigned char nonce[crypto_stream_chacha20_NONCEBYTES] = {0};
        crypto_stream_chacha20(data, total_size, nonce, seed);
        
        // Upload matrices to GPU memory (using private storage mode for efficiency)
        uint8_t *pos = data;
        
        // Expansion matrix
        _expandMatBuffer = [_device newBufferWithBytes:pos
                                              length:HIDDEN * IN_SIZE
                                             options:MTLResourceStorageModePrivate];
        pos += HIDDEN * IN_SIZE;
        
        // Middle matrices (all in one buffer)
        _middleMatBuffers = [_device newBufferWithBytes:pos
                                              length:ROUNDS * HIDDEN * HIDDEN
                                             options:MTLResourceStorageModePrivate];
        pos += ROUNDS * HIDDEN * HIDDEN;
        
        // Compression matrix
        _compressMatBuffer = [_device newBufferWithBytes:pos
                                                length:IN_SIZE * HIDDEN
                                               options:MTLResourceStorageModePrivate];
        
        free(data);
    }
    return self;
}

- (void)processNonceBatch:(const uint8_t*)nonces 
              noiseVectors:(const int8_t*)noiseVectors 
                    count:(int)count 
                  outputs:(uint8_t*)outputs {
    // Create buffers for input/output
    id<MTLBuffer> nonceBuffer = [_device newBufferWithBytes:nonces
                                                   length:count * IN_SIZE
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> noiseBuffer = [_device newBufferWithBytes:noiseVectors
                                                    length:count * IN_SIZE
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [_device newBufferWithLength:count * IN_SIZE
                                                    options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:_hashPipelineState];
    
    // Set buffers
    [computeEncoder setBuffer:nonceBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:noiseBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_expandMatBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:_middleMatBuffers offset:0 atIndex:3];
    [computeEncoder setBuffer:_compressMatBuffer offset:0 atIndex:4];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:5];
    
    // Calculate dispatch size
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(count, 256), 1, 1);
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    
    // Execute and wait
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    memcpy(outputs, [outputBuffer contents], count * IN_SIZE);
}

@end

void print_hex(uint8_t *bytes, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", bytes[len - 1 - i]);
    }
    printf("\n");
}

int count_leading_zero_bits(uint8_t *hash) {
    int count = 0;
    for (int i = IN_SIZE - 1; i >= 0; i--) {
        uint8_t byte = hash[i];
        if (byte == 0) {
            count += 8;
        } else {
            for (int bit = 7; bit >= 0; bit--) {
                if ((byte >> bit) & 1)
                    return count;
                count++;
            }
            break;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <seed_hex> <difficulty>\n", argv[0]);
        return 1;
    }

    if (sodium_init() < 0) {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return 1;
    }

    uint8_t seed[32];
    for (size_t i = 0; i < 32; i++) {
        sscanf(argv[1] + 2 * (31 - i), "%2hhx", &seed[i]);
    }
    
    int difficulty = atoi(argv[2]);
    if (difficulty < 1 || difficulty > 256) {
        fprintf(stderr, "Difficulty must be between 1 and 256\n");
        return 1;
    }

    @autoreleasepool {
        TensPowMetal *miner = [[TensPowMetal alloc] initWithSeed:seed];
        if (!miner) {
            fprintf(stderr, "Failed to initialize Metal miner\n");
            return 1;
        }
        
        uint64_t attempts = 0;
        time_t start_time = time(NULL);
        time_t last_report = start_time;
        uint64_t last_attempts = 0;
        int best_zero_bits = 0;
        
        printf("Mining with Metal acceleration (batch size: %d):\n", BATCH_SIZE);
        printf("  Seed: ");
        print_hex(seed, 32);
        printf("  Difficulty: %d leading 0 bits\n", difficulty);
        printf("\nProgress:\n");
        printf("  Time    Hash Rate      TOPS         Total Hashes    Best Bits\n");
        printf("  ----    ---------    --------      ------------    ----------\n");

        // Allocate batch buffers
        uint8_t *batch_nonces = (uint8_t*)malloc(BATCH_SIZE * IN_SIZE);
        int8_t *batch_noise = (int8_t*)malloc(BATCH_SIZE * IN_SIZE);
        uint8_t *batch_outputs = (uint8_t*)malloc(BATCH_SIZE * IN_SIZE);
        
        while (1) {
            // Generate batch of nonces and their noise vectors
            for (int i = 0; i < BATCH_SIZE; i++) {
                uint64_t nonce_num = attempts + i;
                
                // Set nonce (little-endian in first 8 bytes)
                memset(batch_nonces + i * IN_SIZE, 0, IN_SIZE);
                for (int j = 0; j < 8; j++) {
                    batch_nonces[i * IN_SIZE + j] = (nonce_num >> (j * 8)) & 0xFF;
                }
                
                // Calculate SHA-256 of nonce for noise vector
                unsigned char digest[crypto_hash_sha256_BYTES];
                crypto_hash_sha256(digest, batch_nonces + i * IN_SIZE, IN_SIZE);
                memcpy(batch_noise + i * IN_SIZE, digest, IN_SIZE);
            }
            
            // Process batch on GPU
            [miner processNonceBatch:batch_nonces 
                       noiseVectors:batch_noise
                             count:BATCH_SIZE 
                           outputs:batch_outputs];
            
            // Check results
            for (int i = 0; i < BATCH_SIZE; i++) {
                uint8_t *hash = &batch_outputs[i * IN_SIZE];
                int zeros = count_leading_zero_bits(hash);
                if (zeros > best_zero_bits) {
                    best_zero_bits = zeros;
                }
                
                if (zeros >= difficulty) {
                    time_t end_time = time(NULL);
                    double duration = difftime(end_time, start_time);
                    uint64_t total_attempts = attempts + i;
                    double avg_tops = (total_attempts * OPS_PER_HASH) / (duration * 1e12);
                    
                    printf("\nSolution found!\n");
                    printf("Nonce: ");
                    print_hex(batch_nonces + i * IN_SIZE, IN_SIZE);
                    printf("Hash:  ");
                    print_hex(hash, IN_SIZE);
                    printf("Stats:\n");
                    printf("  Time: %.1f seconds\n", duration);
                    printf("  Total hashes: %" PRIu64 "\n", total_attempts);
                    printf("  Avg hash rate: %.1f h/s\n", total_attempts / duration);
                    printf("  Avg TOPS: %.6f\n", avg_tops);
                    
                    free(batch_nonces);
                    free(batch_noise);
                    free(batch_outputs);
                    return 0;
                }
            }
            
            attempts += BATCH_SIZE;
            
            // Update progress
            time_t current_time = time(NULL);
            if (current_time > last_report) {
                double interval = difftime(current_time, last_report);
                uint64_t interval_hashes = attempts - last_attempts;
                double hash_rate = interval_hashes / interval;
                double tops = (hash_rate * OPS_PER_HASH) / 1e12;
                double total_time = difftime(current_time, start_time);
                
                printf("  %4.0fs    %7.0f h/s    %.6f    %12" PRIu64 "    %10d\r", 
                       total_time, hash_rate, tops, attempts, best_zero_bits);
                fflush(stdout);
                
                last_report = current_time;
                last_attempts = attempts;
            }
        }
        
        free(batch_nonces);
        free(batch_noise);
        free(batch_outputs);
    }

    return 0;
}
