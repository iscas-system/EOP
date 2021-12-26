import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import numpy as np
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
# import time
# from cnn_workload_generator import get_network,compile_with_executor,evaluate_time_with_tvm_evaluator
import logging

logging.basicConfig(level=logging.INFO,filename="/root/github/logs/runpass.log")


dshape = (2048, 2048)
x = relay.var("x", shape=dshape)
y = relay.var("y", shape=dshape)
z = relay.var("z", shape=dshape)
a = relay.var("a", shape=dshape)
b = relay.nn.matmul(x, y)
c = relay.nn.matmul(z, a)
d = relay.add(b,c)
# y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# y1 = relay.add(relay.const(1, "float32"), y)
# y = relay.add(y, y1)
# z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z = relay.add(z2, z3)
func = relay.Function(relay.analysis.free_vars(d), d)
# func = run_opt_pass(func, transform.FuseOps(fuse_opt_level=2))

mod = tvm.IRModule()
mod["main"] = func
mod = relay.transform.InferType()(mod)

target = "llvm"
# target = "cuda"
dtype = "float32"
device = tvm.cpu()
# device = tvm.cuda(0)
model_params = {}
# model_params["w1"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
# model_params["w2"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
# model_params["w3"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
x = np.random.uniform(5,10,dshape).astype("float32")
y = np.random.uniform(5,10,dshape).astype("float32")
z = np.random.uniform(5,10,dshape).astype("float32")
a = np.random.uniform(5,10,dshape).astype("float32")

with tvm.transform.PassContext(opt_level=0, disabled_pass=["AlterOpLayout"]):
   lib = relay.build(mod, target=target, params=model_params)
   print(type(lib.get_graph_json()))
   print(type(lib.get_lib()))
   m = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device, dump_root="/root/github/log")
   m.set_input('x',tvm.nd.array(x.astype(dtype)))
   m.set_input('y',tvm.nd.array(y.astype(dtype)))
   m.set_input('z',tvm.nd.array(z.astype(dtype)))
   m.set_input('a',tvm.nd.array(a.astype(dtype)))
   m.run()
   tvm_out = m.get_output(0, tvm.nd.empty(dshape, dtype)).numpy()
   print(tvm_out)

# lib = relay.build(mod, target=target, params=model_params)
# m = graph_executor.create(lib["get_graph_json"](), lib, device, dump_root="/root/github/logs")


# with tvm.transform.PassContext(opt_level=0, disabled_pass=["AlterOpLayout"]):
   
#    interpreter = relay.build_module.create_executor("graph", lib.ir_mod, device, target)
# fused op are decided by fuseOps transofrm.
# print(lib.ir_mod)
# print(lib.get_graph_json())
# print(lib.function_metadata)

# mod, params, input_shape, output_shape = get_network("resnet-18", 1)
# x = np.random.uniform(5,10,input_shape).astype("float32")
# lib,interpreter,module = compile_with_executor(mod,device,target,params,3, [x])
# print(type(mod.functions.items()[0][1]))
# evaluate_time_with_tvm_evaluator(module,device)

# print(type(interpreter))
# print(type(module))
# t1 = time.time_ns()
# top1_tvm = interpreter.evaluate()(tvm.nd.array(x.astype(dtype)), **params)
# t2 = time.time_ns() - t1
# print(t2/1e9)

# dev = tvm.device(target, 0)
# m = graph_executor.GraphModule(lib["default"](dev))
# m.run(x)