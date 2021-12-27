import os
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

class FastSoftmaxMutator(tvm.relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        call = super().visit_call(call)
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.softmax":
            return tvm.relay.nn.fast_softmax(call.args[0], call.attrs.axis)
        return call

@tvm.relay.transform.function_pass(opt_level=1)
def FastSoftmax(fn, mod, device):
    return FastSoftmaxMutator().visit(fn)

def get_network(name, batch_size, layout="NCHW", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'bert':
        import torch
        import transformers  # pip3 install transformers==3.0
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        input_shape = [batch_size, 128]

        if os.path.exists("bert-mod.relay"):
            print("Load relay model from file...")
            with open("bert-mod.relay", "r") as fi:
                mod = tvm.ir.load_json(fi.read())
            with open("bert-params.relay", "rb") as fi:
                params = relay.load_param_dict(fi.read())
        else:
            model_class = transformers.BertModel
            tokenizer_class = transformers.BertTokenizer

            # You can also download them manualy
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
            # Then rename to pytorch_model.bin, vocab.txt & config.json
            # weight = 'path to downloaded model dir'
            weight = 'bert-base-uncased'
            model = model_class.from_pretrained(weight,return_dict=False)
            model.eval()

            # tokenizer = tokenizer_class.from_pretrained(weight)
            # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
            # There is 30522 words in bert-base-uncased's vocabulary list
            input_name = 'input_ids'
            input_dtype = 'int64'
            A = torch.randint(30000, input_shape)
            scripted_model = torch.jit.trace(model, [A], strict=False)
            shape_list = [('input_ids', input_shape)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

            mod = tvm.relay.transform.FastMath()(mod)
            mod = FastSoftmax(mod)
            mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
            BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, device:
                                tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
            mod = BindPass(mod)
            mod = tvm.relay.transform.FoldConstant()(mod)
            mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
            mod = tvm.relay.transform.FoldConstant()(mod)

            with open("bert-mod.relay", "w") as fo:
                fo.write(tvm.ir.save_json(mod))
            with open("bert-params.relay", "wb") as fo:
                fo.write(relay.save_param_dict(params))
            print("Save relay model to file...")
    # elif name == "mxnet":
    #     # an example for mxnet model
    #     from mxnet.gluon.model_zoo.vision import get_model

    #     assert layout == "NCHW"

    #     block = get_model("resnet18_v1", pretrained=True)
    #     mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
    #     net = mod["main"]
    #     net = relay.Function(
    #         net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    #     )
    #     mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape

network = "bert"
batch_size = 128
layout = "NHWC"
#target = tvm.target.Target("llvm -mcpu=core-avx2")
# target = tvm.target.Target("llvm -mcpu=skylake-avx512")
target = tvm.target.Target("cuda")
dtype = "float32"
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": False}):
    lib = relay.build(mod, target=target, params=params)

device = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](device))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
module.set_input("input_ids", data_tvm)

print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", device, repeat=5, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
