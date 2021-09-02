import onnx_profiler
from relay_graph import construct_op_graph, profile_resource_usage
from tvm.relay.testing import densenet
import tvm
import numpy as np
from tvm import relay
from tvm.contrib import graph_runtime
from cnn_workload_generator import get_network, compile_without_log, create_graph_executor_on_single_device, evaluate_time_with_tvm_evaluator, create_operator_executor_on_single_device

def test_copy(data, src_dev=tvm.cpu(0), dst_dev = tvm.cuda(0)):
    res = tvm.relay.device_copy(data, src_dev, dst_dev)
    return res

batch_size = 1
image_shape = (batch_size, 3, 224, 224)
expr_var = tvm.relay.var("data", shape=image_shape, dtype="float32")
output = test_copy(expr_var)
source_data = tvm.relay.var("source_data", shape=image_shape, dtype="float32")
func = relay.Function(relay.analysis.free_vars(output),output)
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

lib = compile_without_log(mod, target, params)
module = create_graph_executor_on_single_device(lib,image_shape,target)
module.run()
print(module.get_input(0))
# print(evaluate_time_with_tvm_evaluator(module, device))