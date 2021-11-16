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

GRU GPU (K40)
('{"model_name": "gru.onnx", "split": "39.915108%", "strided_slice": "0.494428%", "squeeze": "0.008111%", "nn.dense": "15.117942%", "add": "25.909708%", "sigmoid": "3.149892%", "multiply": "7.765988%", "subtract": "2.534276%", "tanh": "3.182416%", "expand_dims": "0.005639%", "tuple": "0.002735%", "concatenate": "1.913757%"}', '{"split": 622886.2438941656, "total_op_time": 1560527.5140866162, "strided_slice": 7715.6891364530065, "squeeze": 126.57162338704418, "nn.dense": 235919.6460860073, "add": 404328.12627306813, "sigmoid": 49154.931404165705, "multiply": 121190.3762392474, "subtract": 39548.08043585027, "tanh": 49662.47463692635, "expand_dims": 87.9968195979471, "tuple": 42.67580767812042, "concatenate": 29864.70173007009}')

GRU CPU
('{"model_name": "gru.onnx", "strided_slice": "0.010630%", "split": "25.233309%", "squeeze": "0.034022%", "tuple": "0.020506%", "concatenate": "20.863130%", "nn.dense": "48.424890%", "add": "0.429449%", "sigmoid": "0.506436%", "multiply": "0.135104%", "tanh": "0.508112%", "subtract": "0.069060%", "expand_dims": "0.006175%", "stack": "3.759177%"}', '{"strided_slice": 43.93042702420329, "total_op_time": 413251.0657968184, "split": 104276.91952773358, "squeeze": 140.59463814063565, "tuple": 84.74080867574597, "concatenate": 86217.10595609115, "nn.dense": 200116.3744677221, "add": 1774.7017970004658, "sigmoid": 2092.85252035714, "multiply": 558.3166903358982, "tanh": 2099.777864463968, "subtract": 285.3923737307651, "expand_dims": 25.519647040797373, "stack": 15534.839078501722}')

GRU: GPU(T4) batchsize = 4
('{"model_name": "gru_4.onnx", "strided_slice": "0.783228%", "split": "38.285761%", "squeeze": "0.019094%", "tuple": "0.009594%", "concatenate": "3.587220%", "nn.dense": "8.204470%", "add": "24.383845%", "sigmoid": "7.800060%", "multiply": "8.068694%", "tanh": "3.912103%", "subtract": "4.130863%", "expand_dims": "0.003513%", "stack": "0.811556%"}', '{"strided_slice": 6329.327141323877, "total_op_time": 808107.4975594197, "split": 309390.10427925864, "squeeze": 154.29805759278753, "tuple": 77.52719023889071, "concatenate": 28988.59358010277, "nn.dense": 66300.93903406149, "add": 197047.67876552575, "sigmoid": 63032.86831145979, "multiply": 65203.71985855969, "tanh": 31613.99661925987, "subtract": 33381.815126027715, "expand_dims": 28.386123919572317, "stack": 6558.243472088876}')

GRU: CPU batchsize = 4
op_name,op_time,op_proportion
strided_slice,13651.0120465002,0.986112%
split,264930.7560810048,19.137878%
squeeze,141.0684308457701,0.010190%
tuple,85.20475191968765,0.006155%
concatenate,70239.51219826529,5.073912%
nn.dense,175563.78254249558,12.682251%
add,425418.88050310226,30.731104%
sigmoid,140780.23180427926,10.169582%
multiply,139542.94755358592,10.080203%
tanh,69972.19993282517,5.054602%
subtract,70283.78381594038,5.077110%
expand_dims,25.527839748927178,0.001844%
stack,13691.786974710452,0.989057%

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

# input_name = ['input', 'h0', 'c0']
input_name = ['input', 'h0']

inputsize = 50

input = np.random.uniform(-10, 10, (10, 4, inputsize)).astype("float32")
h0 = np.random.uniform(-10, 10, (2, 4, 20)).astype("float32")
c0 = np.random.uniform(-10, 10, (2, 3, 20)).astype("float32")
# data = [input,h0,c0]
data = [input,h0]

onnx_model = onnx_profiler.create_onnx_model_from_local_path("../model_src/onnx/gru_inputsize_50.onnx")
mod, params, intrp = onnx_profiler.compile_onnx_model(onnx_model, data, target=target, input_names=input_name, device=device)

# print(mod)
# print(params)
relay_graph.construct_op_graph(mod)
tmp = {input_name[i]:data[i] for i in range(len(data))}
relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target)
print(op_statistics.calculate_op_distribution("lstm"))
s,g = op_statistics.calculate_op_distribution("bert")
file_name = "./data/" + "lstm-cpu-inputsize" + str(inputsize) + ".csv"
csv_file = open(file_name, "w")
g_list = json.loads(g)
key_data = g_list.keys()
value_data = [g_list[key] for key in g_list.keys()]

# csv文件写入对象
csv_writer = csv.writer(csv_file)
# 先写入表头字段数据
csv_writer.writerow(key_data)
# 再写入表的值数据
csv_writer.writerow(value_data)