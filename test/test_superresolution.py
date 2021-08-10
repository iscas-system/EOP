import onnx
import numpy as np
import tvm

from PIL import Image
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
from relay_graph import construct_op_graph, profile_memory
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
mod, params, intrp = compile_onnx_model(onnx_model, input, target = "llvm", input_name = "1", device = tvm.cpu(0))
run_relay_mod(input, intrp, params)

construct_op_graph(mod)
profile_memory(params, input)

