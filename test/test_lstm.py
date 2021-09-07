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
lstm GPU (Tesla T4):

('{"model_name": "lstm", "split": "35.680714%", "strided_slice": "1.731395%", "squeeze": "0.021821%", "tuple": "0.025390%", "concatenate": "12.498807%", "nn.dense": "4.169467%", "add": "12.481705%", "sigmoid": "12.222225%", "tanh": "8.068169%", "multiply": "12.304088%", "expand_dims": "0.005361%", "stack": "0.790859%"}', '{"split": 277048.75467553036, "total_op_time": 776466.4021155308, "strided_slice": 13443.697815679732, "squeeze": 169.43572855967608, "tuple": 197.14390394223176, "concatenate": 97049.03538370853, "nn.dense": 32374.508181266123, "add": 96916.24596258547, "sigmoid": 94901.47405370796, "tanh": 62646.61777533076, "multiply": 95537.10819533461, "expand_dims": 41.62261496135837, "stack": 6140.75782492433}')

lstm CPU (4ocore):
('{"model_name": "lstm", "split": "15.035746%", "strided_slice": "2.033175%", "squeeze": "0.012159%", "tuple": "0.014341%", "concatenate": "17.462748%", "nn.dense": "6.982122%", "add": "15.640527%", "sigmoid": "15.773741%", "tanh": "10.336989%", "multiply": "15.560595%", "expand_dims": "0.003059%", "stack": "1.144797%"}', '{"split": 205835.59090531088, "total_op_time": 1368974.9462482256, "strided_slice": 27833.659337463763, "squeeze": 166.45997918459346, "tuple": 196.31945196345038, "concatenate": 239060.64665575308, "nn.dense": 95583.50366763504, "add": 214114.9005568708, "sigmoid": 215938.5602816046, "tanh": 141510.7941084666, "multiply": 213020.64778143063, "expand_dims": 41.87820148826831, "stack": 15671.98532105346}')

lstm GPU (K40):
('{"model_name": "lstm", "split": "35.828689%", "strided_slice": "1.352614%", "squeeze": "0.013148%", "tuple": "0.018628%", "concatenate": "12.639268%", "nn.dense": "5.846429%", "add": "11.755701%", "sigmoid": "12.687557%", "tanh": "8.558062%", "multiply": "10.444008%", "expand_dims": "0.003292%", "stack": "0.852602%"}', '{"split": 413525.8828967108, "total_op_time": 1154175.3014585138, "strided_slice": 15611.539587660709, "squeeze": 151.75101076072076, "tuple": 214.99991531015718, "concatenate": 145879.31358017796, "nn.dense": 67478.03936335829, "add": 135681.40083795998, "sigmoid": 146436.6535601757, "tanh": 98775.03985823799, "multiply": 120542.16150005446, "expand_dims": 37.99701639973391, "stack": 9840.522331707594}')

CPU/K40:
Total: 1.18
add:1.6
dense: 1.4
split: 0.5
concatenate: 1.6
sigmoid: 1.47
tanh: 1.43
multiply: 1.77

K40/T4:

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