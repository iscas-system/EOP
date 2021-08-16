import numpy as np
import scipy.signal

from std_memory_profiler import operation_memory_profile, operation_time_profile

def create_data(ra):
    ret = []
    for n in range(ra):
        ret.append(np.random.randn(1, 70, 71, 72))
    return ret


@operation_time_profile
def process_data(data):
    data = np.concatenate(data)
    detrended = scipy.signal.detrend(data, axis=0)
    return detrended


if __name__ == "__main__":
    for n in range(10):
        data1 = create_data(np.random.randint(10,100))
        data2 = process_data(data1)
