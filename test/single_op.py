import sys
sys.path.append('/root/huyi/TVMProfiler/relayIR')
sys.path.append('/root/huyi/TVMProfiler/memory_profiler')

import tvm
from tvm.relay import transform
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
from relay_graph import construct_op_graph, profile_resource_usage
from std_memory_profiler import profile, operation_time_profile, operation_memory_profile, operation_cuda_memory_profile

target = "llvm"
device = tvm.cpu(0)

for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            #a_shape = (i*10, j*10, k*10)
            a_shape = (i*10, j*10)
            b_shape = (j*10, k*10)
            dtype = "float32"
            a = relay.var('a', shape=a_shape, dtype=dtype)
            b = relay.var('b', shape=b_shape, dtype=dtype)
            data_a = np.random.uniform(1000, 50, size=a_shape).astype(dtype)
            data_b = np.random.uniform(1000, 50, size=b_shape).astype(dtype)
            f = relay.nn.matmul(a,b)
            func = relay.Function(relay.analysis.free_vars(f),f)
            mod = tvm.ir.IRModule.from_expr(func)
            mod = relay.transform.InferType()(mod)
            shape_dict = {
                v.name_hint: v.checked_type for v in mod["main"].params}
            np.random.seed(0)
            params = {}
            for k, v in shape_dict.items():
                if k == "data":
                    continue
                init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
                params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))
            call_functions = {"main": func}
            call_ir_module = tvm.ir.IRModule(functions=call_functions)
            with tvm.transform.PassContext(opt_level=1):
                call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, target)
            input_args = [data_a,data_b]

            # tmp_param = call_interpreter.mod["main"].params
            # cnt = 0
            # out = ""
            # for p in tmp_param:
            #     out = out + str(p.name_hint) + ','
            #     cnt = cnt + 1
            #     if cnt % 10 == 0:
            #         print(out)
            #         out = ""
            # if cnt % 10 != 0:
            #     print(out)
            # print("args:")
            # for p in input_args:
            #     print(p)
            # print("params:")
            # for p in params:
            #     print(p)

            metadata = {}
            @operation_time_profile(stream=sys.stdout, operation_meta=metadata)
            def op_time_forward_profile(call_interpreter, call_intput_args, ir_params):
                return call_interpreter.evaluate()(*call_intput_args, **ir_params)
            op_time_forward_profile(call_interpreter, input_args, {})
