from transformers import BertModel, BertTokenizer, BertConfig
import torch
import tvm
import tvm.relay
from tvm.contrib import graph_runtime
import onnx_profiler
import relay_graph
import os
import op_statistics
import sys
import csv
import json

'''
bert: CPU
('{"model_name": "bert", "cast": "0.000076%", "strided_slice": "0.000035%", "transpose": "19.338548%", "expand_dims": "0.000010%", "take": "0.015821%", "repeat": "0.000018%", "multiply": "0.201182%", "add": "0.664358%", "subtract": "0.000018%", "nn.layer_norm": "0.694734%", "nn.dropout": "0.000110%", "reshape": "0.001279%", "nn.dense": "77.784594%", "nn.batch_matmul": "0.274035%", "divide": "0.064497%", "nn.softmax": "0.088449%", "erf": "0.867593%", "tanh": "0.004638%", "tuple": "0.000005%"}', '{"cast": 101.01691008994716, "total_op_time": 132487045.55422042, "strided_slice": 46.93220387275787, "transpose": 25621070.37045182, "expand_dims": 12.739240389150503, "take": 20961.236664950036, "repeat": 23.847749260967312, "multiply": 266539.78406513506, "add": 880188.4485154, "subtract": 24.40867608479712, "nn.layer_norm": 920432.8395848701, "nn.dropout": 145.94995260013997, "reshape": 1695.0509930852302, "nn.dense": 103054509.9525598, "nn.batch_matmul": 363060.3535744703, "divide": 85449.51553656165, "nn.softmax": 117183.47923938687, "erf": 1149448.892468539, "tanh": 6144.593092327872, "tuple": 6.142741622195516}')

bert: GPU(K40)
('{"model_name": "bert", "cast": "0.026128%", "strided_slice": "0.013049%", "transpose": "58.419500%", "expand_dims": "0.000019%", "take": "0.029508%", "repeat": "0.006518%", "multiply": "0.290617%", "add": "0.819530%", "subtract": "0.006586%", "nn.layer_norm": "1.716359%", "nn.dropout": "0.000213%", "reshape": "0.002473%", "nn.dense": "37.447621%", "nn.batch_matmul": "0.690844%", "divide": "0.084864%", "nn.softmax": "0.326392%", "erf": "0.113396%", "tanh": "0.006375%", "tuple": "0.000009%"}', '{"cast": 18151.089771884916, "total_op_time": 69469257.59604834, "strided_slice": 9064.977251713633, "transpose": 40583592.85619145, "expand_dims": 12.993658453214131, "take": 20499.224731075043, "repeat": 4527.87672771782, "multiply": 201889.2000140288, "add": 569321.4847191551, "subtract": 4575.346931870142, "nn.layer_norm": 1192342.1756438226, "nn.dropout": 147.79842128658473, "reshape": 1717.665126660619, "nn.dense": 26014583.955047887, "nn.batch_matmul": 479924.10589409026, "divide": 58954.29077181897, "nn.softmax": 226742.1783381147, "erf": 78775.27596735921, "tanh": 4428.9472131579805, "tuple": 6.153626884201612}')
'''

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# batch_seq = []
# for i in range(64):
#     batch_seq.append(text)
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
'''
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)
'''

num_hidden_layer = 60

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=num_hidden_layer, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod_bert, params_bert = tvm.relay.frontend.from_pytorch(traced_model,shape_list, default_dtype="float32")
print(mod_bert)

target = "llvm"
ctx = tvm.cpu(0)
# target_host = 'llvm'
# target = "cuda"
# ctx = tvm.cuda(0)
target_host = 'llvm'

tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
st_a = tvm.nd.array(segments_tensors.numpy(), ctx)

with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = tvm.relay.build(mod_bert,
                                     target=target,
                                     target_host=target_host,
                                     params=params_bert)
module = graph_runtime.create(graph, lib, ctx)
#
# module.set_input("input_ids", tt_a)
# module.set_input("attention_mask", st_a)
# module.set_input(**params)
#
# module.run()
#
# print(module.get_output(0))
relay_graph.construct_op_graph(mod_bert)
parent = os.path.dirname(os.path.realpath(__file__))
a = tokens_tensor.numpy()
b = segments_tensors.numpy()
data = [a,b]
input_name = ["input_ids","attention_mask"]
tmp = {input_name[i]:data[i] for i in range(len(data))}
relay_graph.profile_resource_usage(params_bert, tmp,input_name, device = tvm.cuda(0), target = "cuda", output_file = os.path.join(parent,'bert.csv'))
print(op_statistics.calculate_op_distribution("bert"))
# device_name = 'gpu'
device_name = 'cpu'
file_name = 'bert' + '_' + device_name + '_' + 'num_layers-' + str(num_hidden_layer)
a , b = op_statistics.calculate_op_distribution("bert")
out_file = "./data/" + file_name + ".csv"
a = json.loads(a)
b = json.loads(b)
output_list = []
cnt = 0
for key in a.keys():
    if key in b.keys():
        output_list.append({})
        output_list[cnt]["op_name"] = key
        output_list[cnt]["op_proportion"] = a[key]
        output_list[cnt]["op_time"] = b[key]
        cnt += 1
with open(out_file, 'w', newline='', encoding='utf-8') as f:
    header = ["op_name","op_time","op_proportion"]
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(output_list)
