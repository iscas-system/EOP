import sys
import onnx_profiler
from relay_graph import construct_op_graph, profile_resource_usage
from tvm.relay.testing import densenet
import tvm
import numpy as np

'''
densenet GPU (K40)
('{"model_name": "densenet", "nn.conv2d": "74.092460%", "nn.batch_norm": "13.969248%", "nn.relu": "10.029394%", "nn.max_pool2d": "0.362063%", "tuple": "0.000807%", "concatenate": "0.879271%", "nn.avg_pool2d": "0.504575%", "nn.batch_flatten": "0.046064%", "nn.dense": "0.071491%", "nn.bias_add": "0.044626%"}', '{"nn.conv2d": 6277521.372522876, "total_op_time": 8472550.842301913, "nn.batch_norm": 1183551.6455789702, "nn.relu": 849745.5427416681, "nn.max_pool2d": 30675.99677999811, "tuple": 68.36091714552472, "concatenate": 74496.69323816108, "nn.avg_pool2d": 42750.374205210726, "nn.batch_flatten": 3902.7588412940013, "nn.dense": 6057.097396768401, "nn.bias_add": 3781.0000798148294}')

dcgan GPU (K40)
('{"model_name": "dcgan", "nn.dense": "0.529023%", "nn.relu": "0.102972%", "reshape": "0.000025%", "nn.conv2d_transpose": "99.232683%", "nn.batch_norm": "0.115216%", "tanh": "0.020081%"}', '{"nn.dense": 134307.4431158783, "total_op_time": 25387833.266235735, "nn.relu": 26142.396359540464, "reshape": 6.319237235504548, "nn.conv2d_transpose": 25193028.190798514, "nn.batch_norm": 29250.911951176142, "tanh": 5098.004773391144}')
'''


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