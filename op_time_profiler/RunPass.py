import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import numpy as np
from tvm.contrib import graph_executor
import time

dshape = (1, 16, 64, 64)
x = relay.var("x", shape=dshape)
x = relay.add(x, relay.const(1, "float32"))
y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
y1 = relay.add(relay.const(1, "float32"), y)
y = relay.add(y, y1)
z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z = relay.add(z2, z3)
func = relay.Function(relay.analysis.free_vars(z), z)
# print("before pass: ", func)
# func = run_opt_pass(func, transform.FuseOps(fuse_opt_level=2))
# print("after pass: ", func)
target = "llvm"
dtype = "float32"
device = tvm.cpu()
mod = tvm.IRModule()
mod["main"] = func
mod = relay.transform.InferType()(mod)

model_params = {}
model_params["w1"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
model_params["w2"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
model_params["w3"] = np.random.uniform(0,10,(16, 16, 1, 1)).astype("float32")
x = np.random.uniform(5,10,(1, 16, 64, 64)).astype("float32")

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
   lib = relay.build(mod, target=target, params=model_params)
   interpreter = relay.build_module.create_executor("graph", lib.ir_mod, device, target)

t1 = time.time_ns()
top1_tvm = interpreter.evaluate()(tvm.nd.array(x.astype(dtype)), **model_params)
t2 = time.time_ns() - t1
print(t2/1000)

# dev = tvm.device(target, 0)
# m = graph_executor.GraphModule(lib["default"](dev))
# m.run(x)
# # print(lib.get_graph_json())