import sys
sys.path.append('/root/huyi/TVMProfiler/relayIR')
sys.path.append('/root/huyi/TVMProfiler/memory_profiler')

import tvm
from tvm.relay import transform
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
from relay_graph import construct_op_graph, profile_resource_usage
from std_memory_profiler import operation_memory_profile, operation_time_profile,operation_cuda_memory_profile, profile
from cnn_workload_generator import get_network, compile_without_log, create_graph_executor_on_single_device, evaluate_time_with_tvm_evaluator, create_operator_executor_on_single_device

def batch_norm_infer(data,
                    gamma=None,
                    beta=None,
                    moving_mean=None,
                    moving_var=None,
                    **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                            gamma=gamma,
                            beta=beta,
                            moving_mean=moving_mean,
                            moving_var=moving_var,
                            **kwargs)[0]

def conv2d(data, weight=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)


def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
            padding=(1, 1), epsilon=1e-5):
    conv2d_value = conv2d(
         data=data,
         channels=channels,
         kernel_size=kernel_size,
         strides=strides,
         padding=padding,
         data_layout='NCHW',
         name=name+'_conv')
    # return conv2d_value
    bn = batch_norm_infer(data=conv2d_value, epsilon=epsilon, name=name + '_bn')
    return relay.nn.relu(data=bn)
    # return act
    

data_shape = (1024, 3, 224, 224)

kernel_shape = (32, 3, 3, 3)
# data_shape = (32, 112, 112)
dtype = "float32"
data = relay.var("data", shape=data_shape, dtype=dtype)
act = conv_block(data, "graph", 64, strides=(2, 2))
func = relay.Function(relay.analysis.free_vars(act),act)


mod = tvm.ir.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)
print(mod)
shape_dict = {
    v.name_hint : v.checked_type for v in mod["main"].params}
np.random.seed(0)
params = {}
for k, v in shape_dict.items():
    if k == "data":
        continue
    init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
    params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))

target = "llvm"
device = tvm.cpu(0)

# lib = compile_without_log(mod, target, params)
# module = create_graph_executor_on_single_device(lib,data_shape,target)
# print(evaluate_time_with_tvm_evaluator(module, device))

# print("Relay module function:\n", mod.astext(show_meta_data=False))
# print("TVM parameters:\n", params.keys())

with relay.build_config(opt_level=3):
    graph, lib, params2 = relay.build(mod, target, params=params)

# print("TVM graph:\n", graph)
# print("TVM parameters:\n", params.keys())
# print("TVM compiled target function:\n", lib.get_source())
# print(mod)
module = graph_runtime.create(graph, lib, device)
# batch_norm_input = np.random.uniform(-1, 1, size=(1, 32, 112, 112)).astype(dtype)

#CPU: x and y scale from 64(1089088), 128(2842070),256(8853516),512(41937188),1024(157439060),2048(542466706),4096(1298424103)
#GPU: 
total_nano_time = 0
for i in range(10):
    data_tvm = np.random.uniform(1, 255, size=data_shape).astype(dtype)
    module.set_input('data', data_tvm)
    module.set_input(**params)
    dict = {}
    @operation_time_profile(operation_meta=dict)
    def func(m):
        m.run()
    print(func(module))
    total_nano_time +=dict["op_nano_time"]

print(total_nano_time/10)

# construct_op_graph(mod)
# profile_resource_usage(params,data_tvm, device=device, target = target)

# entrance_tuple = mod.functions.items()[0]
# main_function = entrance_tuple[1]

# temp_body2 = tvm.relay.Call(main_function.body.tuple_value.op, main_function.body.tuple_value.args, attrs=main_function.body.tuple_value.attrs, type_args=main_function.body.tuple_value.type_args)
# temp_body = tvm.relay.expr.TupleGetItem(temp_body2,0)
# call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
# call_functions = {"main": call_function}
# call_ir_module = tvm.ir.IRModule(functions=call_functions)
# with tvm.transform.PassContext(opt_level=1):
#     call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, target)

# print(call_ir_module)
# input_args = []
# input_args.append(data_tvm)
# print(params.keys())
# res = call_interpreter.evaluate()(*input_args, **params)

# print(res)
