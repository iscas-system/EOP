import sys
import onnx_profiler
from relay_graph import construct_op_graph, profile_resource_usage
from tvm.relay.testing import densenet
import tvm
import numpy as np

#onnx_model = onnx_profiler.create_onnx_model_from_local_path('vgg11.onnx')
#data = onnx_profiler.generate_input_image_data_with_torchvision()
mod, mod_params = densenet.get_workload(classes=2, batch_size=1, image_shape=(1, 224, 224))
#visualize(mod["main"])
print(type(mod))
data = np.random.uniform(-10, 10, (1, 1, 224, 224)).astype("float32")
#mod_params["data"] = data
with tvm.transform.PassContext(opt_level=1):
    intrp = tvm.relay.build_module.create_executor("graph", mod, tvm.cuda(0), 'cuda')
#mod, params,intrp = onnx_profiler.compile_onnx_model(onnx_model,data)
# onnx_profiler.run_relay_mod(data,intrp,mod_params)
construct_op_graph(mod)
profile_resource_usage(mod_params,data)