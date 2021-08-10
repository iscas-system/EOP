import numpy as np
import scipy.signal

from memory_profiler import operation_profile

def create_data(ra):
    ret = []
    for n in range(ra):
        ret.append(np.random.randn(1, 70, 71, 72))
    return ret


@operation_profile
def process_data(data):
    data = np.concatenate(data)
    detrended = scipy.signal.detrend(data, axis=0)
    return detrended


if __name__ == "__main__":
    for n in range(10):
        data1 = create_data(np.random.randint(0,100))
        data2 = process_data(data1)
