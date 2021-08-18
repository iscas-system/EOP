# TVMProfiler

## now
For high level RelayIR, TVMProfiler can consturct computation graph and calculate each op's execution time and memory consumption.

# how to use

1. set proper path:

```
    export TVM_HOME=/root/github/tvm
    export PYTHONPATH=/root/github/TVMProfiler/memory_profiler:/root/github/TVMProfiler/relayIR:$TVM_HOME/python:${PYTHONPATH}
```

2. demo (profile CPU/GPU time, memory and CUDA memory)

```
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
```
