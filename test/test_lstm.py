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

# tgt = tvm.target.Target(target="llvm", host="llvm")
# n = te.var("n")
# A = te.placeholder((n,), name='A')
# B = te.placeholder((n,), name='B')
# C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
# s = te.create_schedule(C.op)
#
# fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
# dev = tvm.device(tgt.kind.name, 0)
# n = 1024
# a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
# b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
# c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
# print(type(a))
# print(type(c))

# def funcd(a,begin,end,strides):
#     return nn.strided_slice(a,begin,end,strides)
#
# data_shape = (5,3,3)
# dtype = "float32"
# a = relay.var("a", shape=data_shape, dtype=dtype)
# act =funcd(a,[0],[1],[1])
# func = relay.Function(relay.analysis.free_vars(act),act)
# mod = tvm.ir.IRModule.from_expr(func)
# mod = relay.transform.InferType()(mod)
# shape_dict = {
#     v.name_hint : v.checked_type for v in mod["main"].params}
# np.random.seed(0)
# params = {}
# for k, v in shape_dict.items():
#     if k == "data":
#         continue
#     init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
#     params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))
#
# target = "llvm"
# device = tvm.cpu(0)
#
# with relay.build_config(opt_level=3):
#     graph, lib, params2 = relay.build(mod, target, params=params)
#
# print(mod)
# module = graph_runtime.create(graph, lib, device)
# data_tvm = np.random.uniform(1, 255, size=data_shape).astype(dtype)
# module.set_input('a', data_tvm)
# module.set_input(**params)
# module.run()
#
# construct_op_graph(mod)
# parent = os.path.dirname(os.path.realpath(__file__))
# input_name = ['a']
# data = [data_tvm]
# tmp = {input_name[i]:data[i] for i in range(len(data))}
# profile_resource_usage(params,tmp,input_name=input_name,device = tvm.cpu(0), target = "llvm", output_file = os.path.join(parent,'lstm.csv'))
tgt = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(tgt.kind.name, 0)
n = 1024
dtype = "float32"
a = tvm.nd.array(np.random.uniform(-1,1,(5,3,10)).astype(dtype), dev)
b = tvm.topi.split(a,5)
c = tvm.topi.squeeze(b[0])
print(c)
