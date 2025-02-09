import coremltools as ct
import numpy as np
import time
import argparse
from threading import Thread, Event

# Global variables
last_output = None
inference_times = []

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark Core ML model performance.")
    parser.add_argument(
        '-c', '--compute',
        type=str,
        choices=['ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE'],
        default='ALL',
        help="Compute unit to use: ALL, CPU_ONLY, CPU_AND_GPU, or CPU_AND_NE."
    )
    parser.add_argument(
        '-n', '--num',
        type=int,
        default=1000,
        help="Total number of inferences to perform."
    )
    parser.add_argument(
        '-b', '--batch',
        type=int,
        default=1,
        help="Batch size for each inference."
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=1,
        help="Interval in seconds to display TOPS."
    )
    return parser.parse_args()

def get_compute_unit(compute_unit_str):
    compute_unit_mapping = {
        'ALL': ct.ComputeUnit.ALL,
        'CPU_ONLY': ct.ComputeUnit.CPU_ONLY,
        'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
        'CPU_AND_NE': ct.ComputeUnit.CPU_AND_NE
    }
    return compute_unit_mapping.get(compute_unit_str, ct.ComputeUnit.ALL)

def display_tops(stop_event, interval, num_operations):
    global last_output, inference_times
    while not stop_event.is_set():
        time.sleep(interval)
        if inference_times:
            avg_inference_time = sum(inference_times) / len(inference_times)
            tops = (num_operations / avg_inference_time) / 1e12
            print(f"Average inference time: {avg_inference_time:.6f} seconds | Estimated TOPS: {tops:.6f}")
            
            # Print last output (first row of batch) with size
            if last_output is not None:
                first_row = last_output[0][:256]
                output_str = ' '.join(f"{x:.4f}" for x in first_row)
                print(f"Last output (size={len(first_row)}): {output_str}")

def main():
    global last_output, inference_times
    args = parse_arguments()

    # Load the Core ML model with the specified compute unit
    compute_unit = get_compute_unit(args.compute)
    model = ct.models.MLModel('test_coreml.mlpackage', compute_units=compute_unit)

    # Display model input and output names
    print("Model input names:", model.input_description)
    print("Model output names:", model.output_description)

    # Generate random binary input data (0s and 1s) with the appropriate shape
    input_shape = (args.batch, 256)
    input_data = np.random.randint(0, 2, input_shape).astype(np.float32)
    bias_data = np.random.randint(0, 2, input_shape).astype(np.float32)

    # Prepare the input dictionary
    input_dict = {'input': input_data, 'bias': bias_data}

    # Warm-up run (optional but recommended)
    model.predict(input_dict)

    # Calculate the number of operations per inference
    num_operations = (64 * 256 * 256 * 2 + 3 * 256) * args.batch

    # Reset global variables
    inference_times = []
    last_output = None

    # Event to signal the display thread to stop
    stop_event = Event()

    # Start the thread to display TOPS at regular intervals
    display_thread = Thread(target=display_tops, args=(stop_event, args.interval, num_operations))
    display_thread.start()

    try:
        # Measure inference time over the specified number of inferences
        for _ in range(args.num):
            start_time = time.time()
            output = model.predict(input_dict)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            last_output = output['clip_63']  # Update last output
    finally:
        # Signal the display thread to stop and wait for it to finish
        stop_event.set()
        display_thread.join()

    # Calculate overall average inference time
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        tops = (num_operations / avg_inference_time) / 1e12
        print(f"Final Average inference time: {avg_inference_time:.6f} seconds | Final Estimated TOPS: {tops:.6f}")

if __name__ == "__main__":
    main()
