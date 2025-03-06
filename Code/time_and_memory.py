import time
import tracemalloc

def measure_time_and_memory(func, *args, **kwargs):
    """
    Runs the given function once, measuring:
      - Wall-clock time (seconds),
      - Current and peak memory usage (bytes).
    Returns: (result_from_func, elapsed_time, current_mem, peak_mem)
    """
    # Clear any previously recorded traces so we start fresh
    tracemalloc.clear_traces()
    
    # Start the tracemalloc
    tracemalloc.start()
    
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    # Grab memory usage data
    current, peak = tracemalloc.get_traced_memory()
    
    # Stop tracing
    tracemalloc.stop()
    
    elapsed = end_time - start_time
    return result, elapsed, current, peak
