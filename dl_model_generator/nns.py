import os
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

os.environ['TVM_BACKTRACE']="1"

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

def get_network(name, batch_size, layout="NCHW", dtype="float32", hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072, max_position_embeddings = 512):
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
    shape2 = ()
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

        # if os.path.exists("bert-mod.relay"):
        #     print("Load relay model from file...")
        #     with open("bert-mod.relay", "r") as fi:
        #         mod = tvm.ir.load_json(fi.read())
        #     with open("bert-params.relay", "rb") as fi:
        #         params = relay.load_param_dict(fi.read())
        # else:
        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer

            # You can also download them manualy
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
            # Then rename to pytorch_model.bin, vocab.txt & config.json
            # weight = 'path to downloaded model dir'
            # weight = 'bert-base-uncased'
            # model = model_class.from_pretrained(weight,return_dict=False)
        configuration = transformers.BertConfig(return_dict=False, hidden_size = hidden_size, num_hidden_layers = num_hidden_layers, num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = max_position_embeddings)
        model = transformers.BertModel(configuration)
        model.eval()

            # tokenizer = tokenizer_class.from_pretrained(weight)
            # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
            # There is 30522 words in bert-base-uncased's vocabulary list
            # input_dtype = 'int64'
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # mod = tvm.relay.transform.FastMath()(mod)
        # mod = FastSoftmax(mod)
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        # BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, device: tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
        # mod = BindPass(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        # mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
    elif name == 'gpt2':
        import torch
        from transformers import GPT2Model, GPT2Config
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        input_shape = [batch_size, 128]
        
        configuration = GPT2Config(return_dict=False)
        model = GPT2Model(configuration)
        input_name = 'input_ids'
        A = torch.randint(50000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False).eval()
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 't5-small':
        return 
        # import torch
        # from transformers import T5Model, T5Config, T5Tokenizer,T5ForConditionalGeneration
        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # input_shape = [batch_size, 128]
        
        # # configuration = T5Config(return_dict=False)
        # # model = T5Model.from_pretrained("t5-small", torchscript=True)
        # input_name = 'input_ids'
        # tokenizer = T5Tokenizer.from_pretrained('t5-small')
        # model = T5ForConditionalGeneration.from_pretrained('t5-small', torchscript =True)
        # input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        # attention_mask = input_ids.ne(model.config.pad_token_id).long()
        # decoder_input_ids = tokenizer('<pad> <extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
        # traced_model = torch.jit.trace(model, (input_ids, attention_mask, decoder_input_ids))
        # # torch.jit.save(traced_model, "traced_t5.pt")
        # input_shape = input_ids.shape
        # shape2 = attention_mask.shape
        # # # ('attention_mask',attention_mask.shape),('decoder_input_ids',decoder_input_ids.shape)
        # shape_list = [
        #     (input_name, input_shape),
        #     ('attention_mask',attention_mask.shape),('decoder_input_ids',decoder_input_ids.shape)]
        # mod, params = relay.frontend.from_pytorch(traced_model, shape_list)
        # mod = relay.transform.DynamicToStatic()(mod)
            # with open("bert-mod.relay", "w") as fo:
            #     fo.write(tvm.ir.save_json(mod))
            # with open("bert-params.relay", "wb") as fo:
            #     fo.write(relay.save_param_dict(params))
            # print("Save relay model to file...")
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

    return mod, params, input_shape, output_shape,shape2

network = "gpt2"
batch_size = 1
layout = "NHWC"
#target = tvm.target.Target("llvm -mcpu=core-avx2")
# target = tvm.target.Target("llvm -mcpu=skylake-avx512")
target = tvm.target.Target("cuda")
dtype = "float32"
mod, params, input_shape, output_shape,shape2 = get_network(network, batch_size, layout, dtype=dtype)

with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
    lib = relay.build(mod, target=target, params=params)

device = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](device))
input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
module.set_input("input_ids", input_ids)
attention_mask = tvm.nd.array((np.random.uniform(size=shape2)).astype("int64"))
# module.set_input("attention_mask", attention_mask)
# module.set_input("decoder_input_ids", input_ids)

print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", device, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
