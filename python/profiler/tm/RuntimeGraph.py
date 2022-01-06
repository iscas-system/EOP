import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
from tvm.contrib import graph_executor


x = relay.var("x", shape=(10, 10))
w0 = relay.var("w0", shape=(10, 10))
w1 = relay.var("w1", shape=(10, 10))
w2 = relay.var("w2", shape=(10, 10))

w6 = relay.var("w6", shape=(10, 10))
w7 = relay.var("w7", shape=(10, 10))

# subgraph0
x0 = relay.var("x0", shape=(10, 10))
w00 = relay.var("w00", shape=(10, 10))
w01 = relay.var("w01", shape=(10, 10))
w02 = relay.var("w02", shape=(10, 10))
z00 = relay.add(x0, w00)
p00 = relay.subtract(z00, w01)
q00 = relay.multiply(p00, w02)
subgraph = relay.Function([x0, w00, w01, w02], q00)
subgraph = subgraph.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
subgraph = subgraph.with_attr("Compiler", "ccompiler")
subgraph = subgraph.with_attr("global_symbol", "ccompiler_0")
call0 = relay.Call(subgraph, [x, w0, w1, w2])

# Other parts on TVM
z2 = relay.add(x, w6)
q2 = relay.subtract(z2, w7)

r = relay.concatenate((call0, q2), axis=0)
f = relay.Function([x, w0, w1, w2, w6, w7], r)

mod = tvm.IRModule()
mod["main"] = f
mod = relay.transform.InferType()(mod)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
   lib = relay.build(mod, target="llvm")
#    json, lib, params = relay.build(mod, target="llvm")

print(lib.get_graph_json())
func_metadata = lib.function_metadata  
print(func_metadata)  
# for func_name, finfo in func_metadata.items():
#    print(finfo.workspace_sizes.items())
#    print(finfo.tir_primfuncs.items())