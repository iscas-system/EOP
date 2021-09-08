import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__
import onnx_profiler
import relay_graph
import os
import op_statistics

'''
yolov3: CPU
('{"model_name": "yolov3", "nn.conv2d": "96.600505%", "nn.batch_norm": "1.845165%", "nn.leaky_relu": "0.962330%", "add": "0.245037%", "nn.upsampling": "0.047668%", "tuple": "0.000018%", "nn.bias_add": "0.022487%", "concatenate": "0.028873%", "reshape": "0.000015%", "split": "0.134874%", "sigmoid": "0.113028%"}', '{"nn.conv2d": 240907883.32274368, "total_op_time": 249385738.16623637, "nn.batch_norm": 4601579.534899346, "nn.leaky_relu": 2399913.0289591225, "add": 611086.3021419203, "nn.upsampling": 118876.79676302955, "tuple": 45.94247025457003, "nn.bias_add": 56078.870224520506, "concatenate": 72006.16877805878, "reshape": 37.625818861510545, "split": 336356.019237177, "sigmoid": 281874.5542003876}')
'''

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