import logging
from NewChiliesSplit import split_ms_list  # Replace with the actual module name
import numpy as np
# Setup logger to show debug output in console
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

def test_split_function():
    test_data = [f"ms_{i}" for i in range(7)]  # Sample data: ['ms_0', 'ms_1', ..., 'ms_9']
    num_processes = 3
    LOG.info(f"Testing split_ms_list with {len(test_data)} items and {num_processes} processes.")
    result = split_ms_list(test_data, num_processes)
    result_ = [j for sub in result for j in sub]

    print("\nResult:")
    for i, chunk in enumerate(result):
        print(f"Process {i+1}: {chunk}")

    print(np.array_split(result_, num_processes))

if __name__ == "__main__":
    test_split_function()
