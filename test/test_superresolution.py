import onnx
import numpy as np
import tvm

from PIL import Image
from tvm.contrib.download import download_testdata
from relay_graph import construct_op_graph, profile_resource_usage
from onnx_profiler import create_onnx_model_from_web,compile_onnx_model,run_relay_mod

def get_superresolution_input():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
    img_y, img_cb, img_cr = img_ycbcr.split()
    return np.array(img_y)[np.newaxis, np.newaxis, :, :]


onnx_model = create_onnx_model_from_web()
input = get_superresolution_input()
input_name = ["1"]
data = [input]
mod, params, intrp = compile_onnx_model(onnx_model, data, target = "llvm", input_names = input_name, device = tvm.cpu(0))
run_relay_mod(input, intrp, params)
tmp = {input_name[i]:data[i] for i in range(len(data))}
construct_op_graph(mod)
profile_resource_usage(params, tmp, input_name, device=tvm.cpu(0), target="llvm")

