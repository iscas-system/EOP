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
('{"model_name": "lstm", "split": "35.828689%", "strided_slice": "1.352614%", "squeeze": "0.013148%", "tuple": "0.018628%", "concatenate": "12.639268%", "nn.dense": "5.846429%", "add": "11.755701%", "sigmoid": "12.687557%", "tanh": "8.558062%", "multiply": "10.444008%", "expand_dims": "0.003292%", "stack": "0.852602%"}', '{"split": 413525.8828967108, "total_op_time": 1154175.3014585138, "strided_slice": 15611.539587660709, "squeeze": 151.75101076072076, "tuple": 214.99991531015718, "concatenate": 145879.31358017796, "nn.dense": 67478.03936335829, "add": 135681.40083795998, "sigmoid": 146436.6535601757, "tanh": 98775.03985823799, "multiply": 120542.16150005446, "expand_dims": 37.99701639973391, "stack": 9840.522331707594}')
'''

target = "cuda"
device = tvm.cuda(0)

input_name = ['input', 'h0', 'c0']

input = np.random.uniform(-10, 10, (5, 3, 10)).astype("float32")
h0 = np.random.uniform(-10, 10, (2, 3, 20)).astype("float32")
c0 = np.random.uniform(-10, 10, (2, 3, 20)).astype("float32")
data = [input,h0,c0]

onnx_model = onnx_profiler.create_onnx_model_from_local_path("../model_src/onnx/lstm.onnx")
mod, params, intrp = onnx_profiler.compile_onnx_model(onnx_model, data, target=target, input_names=input_name, device=device)

# print(mod)
# print(params)
relay_graph.construct_op_graph(mod)
tmp = {input_name[i]:data[i] for i in range(len(data))}
relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target)
print(op_statistics.calculate_op_distribution("lstm"))