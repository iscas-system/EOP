#This code aims at profiling the performance of operators in several deep learning workloads
#resnet, vgg, lstm, and gru are from onnx
#yolov3 are from darknet
#densenet, dcgan are from tvm.relay.testing
#

import onnx_profiler
import relay_graph
from tvm.relay.testing import densenet,dcgan
import tvm
import numpy as np
import os
import op_statistics
import tvm.relay
import torch
import torch.nn as nn
import torchvision
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
from optparse import OptionParser
from cnn_workload_generator import get_network
import tvm.relay as relay
import sys
import csv
import json

"""
Example
----------
python run_test.py -o -m resnet18.onnx -g
python run_test.py -o -m gru_4.onnx -b 4 -g
"""

parser = OptionParser(usage="define the data and input_name in source code before running this code")
parser.add_option("-o", "--onnx", action="store_true",
                  dest="onnx",
                  default=False,
                  help="load onnx model")
parser.add_option("-t", "--tvm", action="store_true",
                  dest="tvm",
                  default=False,
                  help="load model from tvm.relay.testing")
parser.add_option("-p", "--pytorch", action="store_true",
                  dest="pytorch",
                  default=False,
                  help="load model from pytorch")
parser.add_option("-d", "--darknet", action="store_true",
                  dest="darknet",
                  default=False,
                  help="load model from darknet")
parser.add_option("-b", "--batchsize", action="store",
                  dest="batchsize",
                  default=1,
                  type="int",
                  help="set model batchsize")
parser.add_option("-l", "--layer_num", action="store",
                  dest="layer_num",
                  default=2,
                  type="int",
                  help="set number of layers")
parser.add_option("-i", "--input_size", action="store",
                  dest="input_size",
                  default=20,
                  type="int",
                  help="set input size")
parser.add_option("-m", "--model",
                  dest="model",
                  default="resnet18.onnx",
                  type="string",
                  help="model_name = onnx_path | name of ( pytorch | tvm.relay.testing | darknet ) ")
parser.add_option("-g", "--gpu_enabled", action="store_true",
                  dest="gpu",
                  default=False,
                  help="whether to use cuda")

(options, args) = parser.parse_args()

if options.gpu == True:
    target = "cuda"
    device = tvm.cuda(0)
else:
    target = "llvm"
    device = tvm.cpu()


"""
define the input_data and input_data_name

attributes
----------
:attr data: input of the model
:attr input_name: keys of the input

description
----------
resnet18: {"input.1":(1,3,224,224)} 
vgg11: {"input.1":(1,3,224,224)}
lstm: {"input":(5,3,10),"h0":(2,3,20),"c0":(2,3,20)}
gru: {"input":(5,1,10),"h0":(2,1,20)}
densenet: {"data":(1,1,224,224)}
dcgan: {"data":(1,100)}
yolov3: {"data":}
"""

# data = np.random.uniform(-10, 10, (options.batchsize, 3, 224, 224)).astype("float32")
data = np.random.uniform(-10, 10, (options.batchsize, 1, 224, 224)).astype("float32")
# data = np.random.uniform(-10, 10, (1, 100)).astype("float32")
# input = np.random.uniform(-10, 10, (5,options.batchsize,options.input_size)).astype("float32")
# h0 = np.random.uniform(-10, 10, (options.layer_num,options.batchsize,20)).astype("float32")
# c0 = np.random.uniform(-10, 10, (options.layer_num,options.batchsize,20)).astype("float32")
# data = [input,h0,c0]
# data = [input,h0]
data = [data]
# input_name = ["input.1"]
# input_name = ["input","h0","c0"]
# input_name = ["input","h0"]
input_name = ["data"]

if options.onnx == True:
    onnx_model = onnx_profiler.create_onnx_model_from_local_path("./onnx/"+options.model)
    mod, params, intrp = onnx_profiler.compile_onnx_model(onnx_model, data, target=target, device=device, input_names=input_name)

if options.tvm == True:
    mod = None
    params = None
    if options.model == "densenet":
        mod, params = densenet.get_workload(classes=2, batch_size=options.batchsize, image_shape=(1, 224, 224))
    elif options.model == "dcgan":
        mod, params = dcgan.get_workload(1, oshape=(3, 64, 64), ngf=128, random_len=100, layout='NCHW', dtype='float32')
    else :
        mod, params, input_shape, output_shape = get_network(options.model, options.batchsize, )
        data = [np.random.uniform(-10, 10, input_shape).astype("float32")]
        input_name = ["data"]
    with tvm.transform.PassContext(opt_level=1):
        intrp = tvm.relay.build_module.create_executor("graph", mod, device, target)
    onnx_profiler.run_relay_mod(data, intrp, params)

    print(mod)

if options.pytorch == True:
    pass

if options.darknet == True:
    cfg_path = '/root/huyi/talos/tvm-analyzer/relay_profiler/darknet/cfg/yolov3.cfg'
    weights_path = '/root/huyi/talos/tvm-analyzer/relay_profiler/darknet/yolov3.weights'
    lib_path = '/root/huyi/talos/tvm-analyzer/relay_profiler/darknet/libdarknet.so'

    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    dtype = 'float32'
    batch_size = 1
    data = np.empty([batch_size, net.c, net.h, net.w], dtype)
    print("Converting darknet to relay functions...")
    mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
    print(mod)
    if options.gpu == False:
        target = 'llvm'
        target_host = 'llvm'
    else:
        target = 'cuda'
        target_host = 'llvm'

    print("Compiling the model...")
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, mod_params = relay.build(mod, target=target, target_host=target_host, params=params)

    data = [data]

if options.onnx == False and options.tvm == False and options.pytorch == False and options.darknet == False:
    raise Exception("Please choose the framework from which the model come")

relay_graph.construct_op_graph(mod)
parent = os.path.dirname(os.path.realpath(__file__))
tmp = {input_name[i]:data[i] for i in range(len(data))}
file_name = options.model.split(".")[0]
if options.gpu == True:
    device_name = "k"
else:
    device_name = "cpu"
file_name = file_name + '_' + device_name + '_' + 'inputsize' + '_' + str(options.input_size)
relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target, output_file = os.path.join(parent,"output/"+file_name+".csv"))
print(op_statistics.calculate_op_distribution(options.model))
a , b = op_statistics.calculate_op_distribution(options.model.split(".")[0])
out_file = "./data/" + file_name + ".csv"
a = json.loads(a)
b = json.loads(b)
output_list = []
cnt = 0
for key in a.keys():
    if key in b.keys():
        output_list.append({})
        output_list[cnt]["op_name"] = key
        output_list[cnt]["op_proportion"] = a[key]
        output_list[cnt]["op_time"] = b[key]
        cnt += 1
with open(out_file, 'w', newline='', encoding='utf-8') as f:
    header = ["op_name","op_time","op_proportion"]
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(output_list)
