"""
realy_graph: key class to read onnx.
:author {xuyuanjia2017,huyi19}@otcaix.iscas.ac.cn
"""
import onnx
import numpy as np
from PIL import Image
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from torchvision import transforms

def create_onnx_model_from_web(name="super_resolution.onnx", url="https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx"):
    """
    create onnx model from web URL.
    ----------
    :name: model name
    :url: web URL
    """
    model_path = download_testdata(url, "super_resolution.onnx", module="onnx")
    return onnx.load(model_path)

def create_onnx_model_from_local_path(abs_path='resnet18.onnx'):
    """
    create onnx model from local absolute path.
    ----------
    :abs_path: local file path
    """
    onnx_model = onnx.load(abs_path)
    return onnx_model

def generate_input_image_data_with_torchvision(img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg", img_name="imagenet_cat.png", module="data",resize1 = (224, 224), resize2 = 256, crop = 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    convert picture to tvm input.

    Paramters:
    ----------
    :param img_url: http path
    :param img_name: image name
    :param module: The defatul is data
    :param resize1: resize for Image
    :param resize2: resize for pytorch
    :param crop: center crop for tensor
    :param mean: mean value for tensor
    :param std: standard derivation value for tensor

    Returns:
    ----------
    :return x: numpy tensor type
    """
    img_path = download_testdata(img_url, img_name, module=module)
    img = Image.open(img_path).resize(resize1)
    my_preprocess = transforms.Compose([
        transforms.Resize(resize2),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    )   
    img = my_preprocess(img)
    x = np.expand_dims(img, 0)
    return x

def compile_onnx_model(onnx_model, x, target = "cuda", input_name = "input.1", device = tvm.cuda(0)):
    """
    compile onnx model

    Paramters:
    ----------
    :param onnx_model: target onnx model
    :param x: input data
    :param target: cuda, llvm, opencv
    :param input_name: onnx input key of x
    :param device: cuda0 as default

    Returns:
    ----------
    :return mod: RelayIRModule
    :return params: pre-trained parameters in onnx model
    :return intrp: model executor in tvm
    """
    shape_dict = {input_name: x.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    with tvm.transform.PassContext(opt_level=1):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    print("relay IR:")
    print(mod)
    return mod, params,intrp

def run_relay_mod(x, intrp, params, dtype="float32"):
    """
    run compiled onnx model

    Paramters:
    ----------
    :param x: input data
    :param intrp: model executor in tvm
    :param params: pre-trained parameters in onnx model
    :param dtype: the default is float32

    Returns:
    ----------
    :return top1_tvm: numpy tensor type
    """
    top1_tvm = np.argmax(intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).numpy())
    print("forward run value:")
    print(top1_tvm)
    return top1_tvm
