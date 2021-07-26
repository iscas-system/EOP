import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata

onnx_model = onnx.load('resnet18.onnx')

from PIL import Image
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
x = np.expand_dims(img, 0)

# 针对cuda进行优化
#target = "llvm"
target = "cuda"
# input_name与onnx模型中的名字一致
input_name = "input.1"
#input_name = "1"
shape_dict = {input_name: x.shape}
#mod为模型表达式。params为参数
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#visualize(mod["main"])
#dev = tvm.cuda()
#evaluator = mod.time_evaluator(mod.entry_name, dev, min_repeat_ms=500)
with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cuda(0), target)
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).numpy()
top1_tvm = np.argmax(tvm_output)
print(top1_tvm)
print(mod)
#建计算图
construct_op_graph(mod)
#profile
profile_memory(params, x)
