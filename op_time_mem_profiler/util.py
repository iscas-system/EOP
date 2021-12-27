import tvm
from tvm import relay
import tvm.relay.testing
from tvm.contrib import graph_executor

def compile_without_log(mod, target, params):
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target=target, params=params)
    return lib

def compile_with_executor(mod, device, target, params, level, input_args):
    with tvm.transform.PassContext(opt_level=level):
        lib = relay.build(mod, target=target, params=params)
        interpreter = relay.build_module.create_executor("graph", lib.ir_mod, device, target)
        module = graph_executor.GraphModule(lib["default"](device))
    for i in range(len(input_args)):
        module.set_input(i, input_args[i])
    return lib,interpreter,module

def create_graph_executor_on_single_device(lib,input_shape,target,dtype="float32"):
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)
    return module

def create_operator_executor_on_single_device(lib, input_args, target):
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    for i in range(len(input_args)):
        module.set_input(i, input_args[i])
    return module

def evaluate_time_with_tvm_evaluator(module, dev):
    ftimer = module.module.time_evaluator("run", dev, repeat=4, min_repeat_ms=500, number=1)
    prof_res = np.array(ftimer().results) * 1e3* 1e6  # convert to millisecond
    print("Mean inference time (std dev): %f ns (%f ns)" % (np.mean(prof_res), np.std(prof_res)))
    return np.mean(prof_res), np.std(prof_res)