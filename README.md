# TVMProfiler

## now
For high level RelayIR, TVMProfiler can consturct computation graph and calculate each op's execution time and memory consumption.

# how to use

1. set proper path:

```
    export TVM_HOME=/root/github/tvm
    export PYTHONPATH=/root/github/TVMProfiler/memory_profiler:/root/github/TVMProfiler/relayIR:$TVM_HOME/python:${PYTHONPATH}
```
## to do list
For tir, TVMProfiler aims to build low-level graph.
