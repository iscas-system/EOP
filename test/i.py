#


#from tvm.relay.visualize import visualize
#relay_graph.profile_resource_usage(params,data,device = tvm.cpu(0), target = "llvm", output_file = os.path.join(parent,'resnet18.csv'))

# onnx_model = onnx_profiler.create_onnx_model_from_local_path('resnet18.onnx')
# data = onnx_profiler.generate_input_image_data_with_torchvision()
#
# mod, params,intrp = onnx_profiler.compile_onnx_model(onnx_model,[data],target = "cuda",input_name=["input.1"])
# onnx_profiler.run_relay_mod([data],intrp,params)
#
# relay_graph.construct_op_graph(mod)
# parent = os.path.dirname(os.path.realpath(__file__))
#
# relay_graph.profile_resource_usage(params,{"input.1":data},input_name=["input.1"],device = tvm.cuda(0), target = "cuda", output_file = os.path.join(parent,'resnet18.csv'))

# onnx_model = onnx_profiler.create_onnx_model_from_local_path('detection.onnx')
# to_tensor = transforms.ToTensor()
# img_rgb = Image.open("cat.png").convert('RGB')
# img_rgb = to_tensor(img_rgb)
# data = [img_rgb.unsqueeze_(0)]
# input_name = ["images"]

# onnx_model = onnx_profiler.create_onnx_model_from_local_path('yolov4.onnx')
# data = np.random.uniform(1, 10, (1, 3, 416, 416)).astype("float32")
# data = [data]
# input_name = ["input_1:0"]
# mod, params,intrp = onnx_profiler.compile_onnx_model(onnx_model,data,target = "cuda",input_names=input_name, freeze_params=True, tvm_mod = "vm")
# onnx_profiler.run_relay_mod(data,intrp,params)
# print("running mod is ok")
# # print(main_function.body)
# # print(type(main_function.body))
# relay_graph.construct_op_graph(mod)
# print("building is ok")
# parent = os.path.dirname(os.path.realpath(__file__))
# tmp = {input_name[i]:data[i] for i in range(len(data))}
# relay_graph.profile_resource_usage(params,tmp,input_name=input_name,device = tvm.cuda(0), target = "cuda", output_file = os.path.join(parent,'yolov3.csv'))
# print("profiling is ok")

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from torch import nn


in_size = 416

input_shape = (1, 3, in_size, in_size)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


from yolort.models import yolov5s


model_func = yolov5s(export_friendly=True, pretrained=True)


model = TraceWrapper(model_func)

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)


img = cv2.imread("cat.png")

img = img.astype("float32")
img = cv2.resize(img, (in_size, in_size))

img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
mod = ToMixedPrecision("float16")(mod)

# print(relay.transform.InferType()(mod))

target = "vulkan -from_device=0"
# target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

ctx = tvm.device(target, 0)
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

with torch.no_grad():
    torch_res = model(torch.from_numpy(img))

for i in range(3):
    print(np.max(np.abs(torch_res[i].numpy() - tvm_res[i].asnumpy())))