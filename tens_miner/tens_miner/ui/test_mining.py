"""Test mining thread implementation."""

import time
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from ..mining import TensHashCore, nonces_from_counter
from ..utils import count_leading_zero_bits

# Test mining constants
TEST_SEED = "c98c980927eaab552002ead60ce34caf166a6daf9945fd2059b5b438af113766"
TEST_DIFFICULTY = 24
EXPECTED_NONCE = "00000000000000000000000000000000000000000000000000000000004eddeb"
EXPECTED_HASH = "000000afff99dbb90dfbf8f8fe5efe0b8e824865285223bb0854b2617675db31"

class TestMiningThread(QThread):
    progress = pyqtSignal(dict)
    solution = pyqtSignal(dict)
    status = pyqtSignal(str)

    def __init__(self, device, batch_size):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            # Initialize miner
            miner = TensHashCore(TEST_SEED, self.device, self.batch_size)
            miner.eval()

            start_time = time.time()
            attempts = 0
            best_zero_bits = 0

            while not self.should_stop:
                nonce_batch = nonces_from_counter(attempts, self.batch_size)
                attempts += self.batch_size

                with torch.no_grad():
                    out_batch = miner.forward_batch(nonce_batch)

                out_batch_cpu = out_batch.cpu().numpy()
                nonce_batch_cpu = nonce_batch.cpu().numpy()

                for i in range(self.batch_size):
                    hash_result = bytes(out_batch_cpu[i].tolist())[::-1]
                    nonce_bytes = bytes(nonce_batch_cpu[i].tolist())[::-1]
                    zeros = count_leading_zero_bits(hash_result)

                    if zeros > best_zero_bits:
                        best_zero_bits = zeros
                        self.progress.emit({
                            'attempts': attempts,
                            'best_zero_bits': zeros,
                            'elapsed_time': time.time() - start_time
                        })

                    if zeros >= TEST_DIFFICULTY:
                        nonce_hex = nonce_bytes.hex()
                        hash_hex = hash_result.hex()
                        
                        if nonce_hex == EXPECTED_NONCE and hash_hex == EXPECTED_HASH:
                            self.status.emit("\nTEST PASSED! ✓")
                            self.status.emit("Found matching nonce and hash")
                        else:
                            self.status.emit("\nTEST FAILED! ✗")
                            self.status.emit(f"Found nonce: {nonce_hex}")
                            self.status.emit(f"Expected:    {EXPECTED_NONCE}")
                            self.status.emit(f"Found hash:  {hash_hex}")
                            self.status.emit(f"Expected:    {EXPECTED_HASH}")
                            
                        self.solution.emit({
                            'nonce': nonce_bytes,
                            'hash': hash_result,
                            'attempts': attempts,
                            'time': time.time() - start_time
                        })
                        return

                # Regular progress updates
                if attempts % (self.batch_size * 10) == 0:
                    self.progress.emit({
                        'attempts': attempts,
                        'best_zero_bits': best_zero_bits,
                        'elapsed_time': time.time() - start_time
                    })

        except Exception as e:
            import traceback
            self.status.emit(f"Error in test mining: {str(e)}\n{traceback.format_exc()}")