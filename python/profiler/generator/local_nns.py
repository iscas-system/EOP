import os
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
import neworkx_exporter

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

def get_network(name, batch_size, layout="NCHW", dtype="float32", sequence = 128, hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072, max_position_embeddings = 512):
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
    inputs = {}
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

        input_shape = [batch_size, sequence]

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
        input_shape = [batch_size, sequence]
        
        configuration = GPT2Config(return_dict=False)
        model = GPT2Model(configuration)
        input_name = 'input_ids'
        A = torch.randint(50000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False).eval()
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'roberta':
        import torch
        from transformers import RobertaConfig, RobertaModel
        configuration = RobertaConfig(return_dict=False)
        model = RobertaModel(configuration).eval()
        input_shape = [batch_size, sequence]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'nasnetalarge':
        import torch
        import pretrainedmodels
        from torch.autograd import Variable
        model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').eval()
        input_shape = [batch_size, 3, 331, 331]
        input = torch.randn(batch_size, 3, 331, 331)
        input_name = 'input0'
        scripted_model = torch.jit.trace(model, [input], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name.startswith('dpn'):
        import torch
        import pretrainedmodels
        from torch.autograd import Variable
        # print(pretrainedmodels.model_names)
        model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet').eval()
        input_shape = [128, 3, 224, 224]
        input = torch.randn(128, 3, 224, 224)
        input_name = 'input0'
        scripted_model = torch.jit.trace(model, [input], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'lstm' or name == 'rnn' or name == 'gru':
        import mxnet as mx
        from mxnet import gluon
        from mxnet.gluon.model_zoo import vision
        def verify(
        mode,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        batch=1,
        init_states=True,
        bidirectional=False,
        ):
            if mode == "rnn":
                layer = gluon.rnn.RNN(hidden_size, num_layers, bidirectional=bidirectional)
            elif mode == "gru":
                layer = gluon.rnn.GRU(hidden_size, num_layers, bidirectional=bidirectional)
            else:  # mode == "lstm"
                layer = gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=bidirectional)
            num_states = 2 if mode == "lstm" else 1
            layer.initialize()
            layer.hybridize()

            dtype = "float32"
            directions = 2 if bidirectional else 1
            data_np = np.random.uniform(size=(seq_len, batch, input_size)).astype(dtype)
            data_mx = mx.nd.array(data_np)
            inputs = {}
            if init_states:
                shape_dict = {"data0": data_np.shape}
                inputs = {"data0": data_np}
                state_shape = (num_layers * directions, batch, hidden_size)
                states_np = []
                states_mx = []
                for i in range(num_states):
                    s = np.random.uniform(size=state_shape).astype(dtype)
                    states_np.append(s)
                    states_mx.append(mx.nd.array(s))
                    shape_dict["data%s" % (i + 1)] = s.shape
                    inputs["data%s" % (i + 1)] = s
                mx_out, mx_states = layer(data_mx, states_mx)
                mx_res = [mx_out] + mx_states
            else:
                shape_dict = {"data": data_np.shape}
                inputs = {"data": data_np}
                mx_res = layer(data_mx)

            mx_sym = layer._cached_graph[1]
            mx_params = {}
            for name, param in layer.collect_params().items():
                mx_params[name] = param._reduce()
            mod, params = relay.frontend.from_mxnet(mx_sym, shape=shape_dict, arg_params=mx_params)
            return mod, params, data_np.shape, inputs
        mod, params, input_shape, inputs = verify(name, 128, 1024, 1024, 8,batch = batch_size)
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

    return mod, params, input_shape, output_shape,inputs

network = "roberta"
batch_size = 128
layout = "NHWC"
dtype = "float32"
# target = tvm.target.Target("cuda")
target = 'llvm'
device = tvm.device(str(target), 0)
if network == 'bert' or network == 'gpt2' or network == 'roberta':
    mod, params, input_shape, output_shape,inputs = get_network(network, batch_size, layout, dtype=dtype, sequence=128)
    with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
        lib = relay.build(mod, target=target, params=params)
    module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
    input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
    module.set_input("input_ids", input_ids)
    print("Evaluate inference time cost...")
    module.run()
elif network == "nasnetalarge":
    #target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    mod, params, input_shape, output_shape,inputs = get_network(network, batch_size, layout, dtype=dtype, sequence=128)

    with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
        lib = relay.build(mod, target=target, params=params)
    
    module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
    input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
    module.set_input("input0", input_ids)
    # attention_mask = tvm.nd.array((np.random.uniform(size=shape2)).astype("int64"))
    # module.set_input("attention_mask", attention_mask)
    # module.set_input("decoder_input_ids", input_ids)
    print("Evaluate inference time cost...")
    module.run()

elif network == 'lstm' or network == 'rnn' or network == 'gru':
    mod, params, input_shape, output_shape,inputs = get_network(network, batch_size, layout, dtype=dtype, sequence=128)
    with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
        lib = relay.build(mod, target=target, params=params)
    module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
    for key in inputs:
        module.set_input(key, tvm.nd.array(inputs[key].astype("float32")))
    print("Evaluate inference time cost...")
    module.run()

elif network == 'dpn68':
    mod, params, input_shape, output_shape,inputs = get_network(network, batch_size, layout, dtype=dtype, sequence=128)
    with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
        lib = relay.build(mod, target=target, params=params)
    # module = graph_executor.GraphModule(lib["default"](device))
    module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
    input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
    module.set_input("input0", input_ids)
    print("Evaluate inference time cost...")
    module.run()
    # ftimer = module.module.time_evaluator("run", device, repeat=3, min_repeat_ms=500)
    # prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    # print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
