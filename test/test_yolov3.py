import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__
import onnx_profiler
import relay_graph
import os
import op_statistics

cfg_path = './darknet/cfg/yolov3.cfg'
weights_path = './darknet/yolov3.weights'
lib_path = './darknet/libdarknet.so'

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
print("Converting darknet to relay functions...")
mod, mod_params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
print(mod)
target = 'llvm'
target_host = 'llvm'

print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=mod_params)

relay_graph.construct_op_graph(mod)
parent = os.path.dirname(os.path.realpath(__file__))
data = [data]
input_name = ["data"]
tmp = {input_name[i]:data[i] for i in range(len(data))}
relay_graph.profile_resource_usage(mod_params, tmp,["data"], device = tvm.cuda(0), target = "cuda", output_file = os.path.join(parent,'yolov3.csv'))
op_statistics.calculate_op_distribution("yolov3")