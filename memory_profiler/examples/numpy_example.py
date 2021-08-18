import numpy as np
import scipy.signal
import sys

from std_memory_profiler import operation_memory_profile, operation_time_profile,operation_cuda_memory_profile, profile

def create_data(ra):
    ret = []
    for n in range(ra):
        ret.append(np.random.randn(1, 70, 71, 72))
    return ret

dict = {'a': 1, 'b': 2, 'b': '3'}

@operation_cuda_memory_profile(stream = sys.stdout,operation_meta=dict)
def process_data(data):
    data = np.concatenate(data)
    detrended = scipy.signal.detrend(data, axis=0)
    return detrended


if __name__ == "__main__":
    for n in range(10):
        data1 = create_data(np.random.randint(10,100))
        data2 = process_data(data1)
