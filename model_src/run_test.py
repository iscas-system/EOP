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
    device = tvm.cpu(0)


"""
define the input_data and input_data_name

attributes
----------
:attr data: input of the model
:attr input_name: keys of the input
"""

data = np.random.uniform(-10, 10, (1, 3, 224, 224)).astype("float32")
data = [data]
input_name = ["input.1"]

if options.onnx == True:
    onnx_model = onnx_profiler.create_onnx_model_from_local_path("./onnx/"+options.model)
    mod, params, intrp = onnx_profiler.compile_onnx_model(onnx_model, data, target=target, input_names=input_name)

if options.tvm == True:
    if options.model == "densenet":
        mod, mod_params = densenet.get_workload(classes=2, batch_size=1, image_shape=(1, 224, 224))
    if options.model == "dcgan":
        mod, params = dcgan.get_workload(1, oshape=(3, 64, 64), ngf=128, random_len=100, layout='NCHW', dtype='float32')
    with tvm.transform.PassContext(opt_level=1):
        intrp = tvm.relay.build_module.create_executor("graph", mod, device, target)
    onnx_profiler.run_relay_mod(data, intrp, params)

if options.pytorch == True:
    pass

if options.darknet == True:
    cfg_path = './darknet/cfg/yolov3.cfg'
    weights_path = './darknet/yolov3.weights'
    lib_path = './darknet/libdarknet.so'

    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    dtype = 'float32'
    batch_size = 1
    data = np.empty([batch_size, net.c, net.h, net.w], dtype)
    print("Converting darknet to relay functions...")
    mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
    print(mod)
    target = 'llvm'
    target_host = 'llvm'

    print("Compiling the model...")
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, mod_params = relay.build(mod, target=target, target_host=target_host, params=params)

relay_graph.construct_op_graph(mod)
parent = os.path.dirname(os.path.realpath(__file__))
tmp = {input_name[i]:data[i] for i in range(len(data))}
file_name = options.model.split(".")[0]
print(file_name)
relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target, output_file = os.path.join(parent,"output/"+file_name+".csv"))
op_statistics.calculate_op_distribution(options.model)