# TVMProfiler

# now
For high level RelayIR, TVMProfiler can consturct computation graph and calculate each op's execution time and memory consumption.

# prerequest

```
    llvm 10.0
    cuda 11.0 (or 10.2)
    nvidia drive 450.51.05
    pytorch 1.7.1
    torchvision 0.8.2+cu110
    torchaudio 0.7.2
    huggingFace (Transformer) needRecheck
    darkNet needRecheck
    OnnX needRecheck
    MxNet needRecheck
```

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

# roadmap

1. support static ananlysis and get necessary input/out put informaction (provide both python and C++ debug modes).
2. support catogerized compile optimization mechanisms.
3. support operator optimizations on heterogeneous resources with reduced cost.

# current support DL model

## from TVM test, Pytorch, MxNet, Keras and Tensorflow (CNN classification)

resnet-*, resnet3d-*, mobilenet, squeezenet_v1.1, inception_v3, InceptionV1, vgg-,densenet-

## from darknet or yolort.models, pytorch classification (RCNN detection)

yolov2, yolov3 or yolov3-tiny

## from Onnx (RNN language)

LSTM,GRU,transformers(e.g., Bert)

## from TVM test (GAN)

DCGAN

## from Pytorch (GCN)

DGL-PyTorch,

# debug tools

1. codelldb (for C++/C): https://github.com/vadimcn/vscode-lldb (may manually download the vsix file)
2. ff-navigator (for python): https://github.com/tqchen/ffi-navigator (may encounter some issues sovlved in https://github.com/tqchen/ffi-navigator/pull/45)