import onnx_profiler
import test_relay
import op_detectm
from tvm.relay.testing import densenet,dcgan
import tvm
import numpy as np
import os
import op_statistics
import tvm.relay
import torch
import torch.nn as nn
import torchvision
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
from optparse import OptionParser
from cnn_workload_generator import get_network
import tvm.relay as relay
import sys
import csv
import json
from transformers import BertModel, BertTokenizer, BertConfig

#bert

# enc = BertTokenizer.from_pretrained("bert-base-uncased")
#
# # Tokenizing input text
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# # batch_seq = []
# # for i in range(64):
# #     batch_seq.append(text)
# tokenized_text = enc.tokenize(text)
#
# # Masking one of the input tokens
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
#
# # Creating a dummy input
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
# dummy_input = [tokens_tensor, segments_tensors]
#
# # Initializing the model with the torchscript flag
# # Flag set to True even though it is not necessary as this model does not have an LM Head.
# '''
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)
# '''
#
# num_hidden_layer = 60
#
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#     num_hidden_layers=num_hidden_layer, num_attention_heads=12, intermediate_size=3072, torchscript=True)
#
# # Instantiating the model
# model = BertModel(config)
#
# # The model needs to be in evaluation mode
# model.eval()
#
# # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
# model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
#
# # Creating the trace
# traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
# traced_model.eval()
# for p in traced_model.parameters():
#     p.requires_grad_(False)
#
# shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
# mod_bert, params_bert = tvm.relay.frontend.from_pytorch(traced_model,shape_list, default_dtype="float32")
# print(mod_bert)
#
# target = "llvm"
# ctx = tvm.cpu(0)
# # target_host = 'llvm'
# # target = "cuda"
# # ctx = tvm.cuda(0)
# target_host = 'llvm'
#
# tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
# st_a = tvm.nd.array(segments_tensors.numpy(), ctx)
#
# with tvm.transform.PassContext(opt_level=3):
#         graph, lib, params = tvm.relay.build(mod_bert,
#                                      target=target,
#                                      target_host=target_host,
#                                      params=params_bert)
# module = graph_runtime.create(graph, lib, ctx)
#
# module.set_input("input_ids", tt_a)
# module.set_input("attention_mask", st_a)
# module.set_input(**params)
#
# module.run()
#
# print(module.get_output(0))
# relay_graph.construct_op_graph(mod_bert)
# parent = os.path.dirname(os.path.realpath(__file__))
# a = tokens_tensor.numpy()
# b = segments_tensors.numpy()
# data = [a,b]
# input_name = ["input_ids","attention_mask"]
# tmp = {input_name[i]:data[i] for i in range(len(data))}
# relay_graph.profile_resource_usage(params_bert, tmp,input_name, device = tvm.cuda(0), target = "cuda", output_file = os.path.join(parent,'bert.csv'))
def get_op_info(name,model_setting,layout = "NCHW",dtype="float32"):
    batch_size = model_setting[0]
    image_shape = model_setting[1]
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
        #input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)

    op_detectm.construct_op_graph(mod)
    device = tvm.cuda(0)
    target = "cuda"
    # a = tokens_tensor.numpy()
    # b = segments_tensors.numpy()
    # data = [a,b]
    # input_name = ["input_ids","attention_mask"]
    data = [np.random.uniform(-10, 10, model_setting[2]).astype("float32")]
    input_name = ["data"]
    tmp = {input_name[i]: data[i] for i in range(len(data))}
    json_s = op_detectm.profile_resource_usage(params, tmp, input_name, device=device, target=target)
    with open("inceptionv3_T4_1.json",'w') as f:
        json.dump(json.loads(json_s),f)

get_op_info("inception_v3",(5,(3,299,299),[5,3,299,299]))