import tvm
from tvm.relay import transform
from tvm import te
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
from relay_graph import construct_op_graph, profile_resource_usage
from std_memory_profiler import operation_memory_profile, operation_time_profile,operation_cuda_memory_profile, profile
import tvm.topi.nn as nn
import tvm.topi
import onnx_profiler
import relay_graph
import op_statistics
import torch

'''
lstm CPU:
('{"model_name": "lstm", "split": "43.920728%", "strided_slice": "11.807081%", "squeeze": "0.027233%", "tuple": "0.034142%", "concatenate": "44.210817%"}', '{"split": 125972.3802549416, "total_op_time": 286817.6070344518, "strided_slice": 33864.786158099894, "squeeze": 78.10893673426544, "tuple": 97.92396837117622, "concatenate": 126804.40771630484}')

lstm GPU:
('{"model_name": "lstm", "split": "66.557981%", "strided_slice": "6.530812%", "squeeze": "0.042268%", "tuple": "0.052681%", "concatenate": "26.816257%"}', '{"split": 125581.50617446535, "total_op_time": 188679.86062871604, "strided_slice": 12322.327837297229, "squeeze": 79.751342814039, "tuple": 99.39875020725627, "concatenate": 50596.87652393224}')
'''

target = "cuda"
device = tvm.cuda(0)

input_name = ['input', 'h0', 'c0']

input = np.random.uniform(-10, 10, (5, 128, 10)).astype("float32")
h0 = np.random.uniform(-10, 10, (2, 128, 20)).astype("float32")
c0 = np.random.uniform(-10, 10, (2, 128, 20)).astype("float32")
data = [input,h0,c0]

onnx_model = onnx_profiler.create_onnx_model_from_local_path("/root/github/TVMProfiler/model_src/onnx/lstm.onnx")
mod, params, intrp = onnx_profiler.compile_onnx_model(onnx_model, data, target=target, input_names=input_name, device=device)

print(mod)
print(params)
# relay_graph.construct_op_graph(mod)
# tmp = {input_name[i]:data[i] for i in range(len(data))}
# relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target)
# print(op_statistics.calculate_op_distribution("lstm"))