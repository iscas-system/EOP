#This code aims at profiling the performance of operators in several deep learning workloads
#resnet, vgg, lstm, and gru are from onnx
#yolov3 are from darknet
#densenet, dcgan are from tvm.relay.testing
#

# import onnx_profiler
# import relay_graph
# from tvm.relay.testing import densenet,dcgan
# import tvm.relay.testing
# import tvm
# import numpy as np
# import os
# import op_statistics
# import tvm.relay
# import torch
# import torch.nn as nn
# import torchvision
# from tvm.contrib.download import download_testdata
# from tvm.relay.testing.darknet import __darknetffi__
# from optparse import OptionParser
# import tvm.relay as relay
# import sys
# import csv
# import json
import os
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor

"""
Example
----------
python run_test.py -o -m resnet18.onnx -g
python run_test.py -o -m gru_4.onnx -b 4 -g
"""

target = "cuda"
device = tvm.cuda(0)
target_host = "llvm"


"""
define the input_data and input_data_name

attributes
----------
:attr data: input of the model
:attr input_name: keys of the input

description
----------
resnet18: {"input.1":(1,3,224,224)} 
vgg11: {"input.1":(1,3,224,224)}
lstm: {"input":(5,3,10),"h0":(2,3,20),"c0":(2,3,20)}
gru: {"input":(5,1,10),"h0":(2,1,20)}
densenet: {"data":(1,1,224,224)}
dcgan: {"data":(1,100)}
yolov3: {"data":}
"""

def main(model_name):
    if model_name == "mobile_net":
        mod, params = tvm.relay.testing.mobilenet.get_workload()
        data = [np.random.uniform(-10, 10, (1,3,224,224)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1,3,224,224)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "mobile_net")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()
        # relay_graph.construct_op_graph(mod)
        # parent = os.path.dirname(os.path.realpath(__file__))
        #
        # file_name = "mobile_net"
        # relay_graph.profile_resource_usage(params, tmp,input_name, device = device, target = target, output_file = os.path.join(parent,"output/"+file_name+".csv"))
        # a , b = op_statistics.calculate_op_distribution(file_name)
        # out_file = "./datat/" + file_name + ".csv"
        # a = json.loads(a)
        # b = json.loads(b)
        # output_list = []
        # cnt = 0
        # for key in a.keys():
        #     if key in b.keys():
        #         output_list.append({})
        #         output_list[cnt]["op_name"] = key
        #         output_list[cnt]["op_proportion"] = a[key]
        #         #output_list[cnt]["op_time"] = b[key]
        #         cnt += 1
        # with open(out_file, 'w', newline='', encoding='utf-8') as f:
        #     header = ["op_name","op_proportion"]
        #     writer = csv.DictWriter(f, fieldnames=header)
        #     writer.writeheader()
        #     writer.writerows(output_list)

    elif model_name == "vgg16":
        mod, params = tvm.relay.testing.vgg.get_workload(batch_size=1,num_layers=16)
        data = [np.random.uniform(-10, 10, (1, 3, 224, 224)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1, 3, 224, 224)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "vgg16")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "resnet101":
        mod, params = tvm.relay.testing.resnet.get_workload(num_layers=101)
        data = [np.random.uniform(-10, 10, (1, 3, 224, 224)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1, 3, 224, 224)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "resnet101")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "resnet152":
        mod, params = tvm.relay.testing.resnet.get_workload(num_layers=152)
        data = [np.random.uniform(-10, 10, (1, 3, 224, 224)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1, 3, 224, 224)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "resnet152")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "inceptionv2":
        pass

    elif model_name == "densenet201":
        mod, params = tvm.relay.testing.resnet.get_workload(densenet_size=201)
        data = [np.random.uniform(-10, 10, (4, 3, 224, 224)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1, 3, 224, 224)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "densenet201")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "bert_large":
        import torch
        import transformers  # pip3 install transformers==3.0
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        sequence = 128
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 3072
        max_position_embeddings = 512
        input_shape = [1, sequence]


        #tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')
        model = transformers.BertModel.from_pretrained("bert-large-uncased", torchscript=True)
        model.eval()


        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        data = [A]
        input_name = [input_name]
        tmp = {input_name[i]: data[i] for i in range(len(data))}


        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target,target_host=target_host, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "bert_large")
        #input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
        input_ids = tvm.nd.array(A.numpy())
        module.set_input("input_ids", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "inceptionv3":
        mod, params = relay.testing.inception_v3.get_workload()
        data = [np.random.uniform(-10, 10, (1, 3, 299, 299)).astype("float32")]
        input_name = ["data"]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        input_shape = (1, 3, 299, 299)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "inceptionv3")
        input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("data", input_ids)
        print("Evaluate inference time cost...")
        module.run()




    elif model_name == "roberta":
        import torch
        from transformers import RobertaConfig, RobertaModel
        sequence = 128
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 3072
        max_position_embeddings = 512
        configuration = RobertaConfig(return_dict=False)
        model = RobertaModel(configuration).eval()
        input_shape = [1, sequence]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        data = [A]
        input_name = [input_name]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "roberta")
        input_ids = tvm.nd.array(A.numpy())
        module.set_input("input_ids", input_ids)
        print("Evaluate inference time cost...")
        module.run()



    elif model_name == "transformer":
        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F
        sequence = 128
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', torchscript=True)
        input_shape = [1, sequence]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)

        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        data = [A]
        input_name = [input_name]
        tmp = {input_name[i]: data[i] for i in range(len(data))}

        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.create(lib.get_graph_json(), lib.get_lib(), device,
                                       dump_root='/root/github/TVMProfiler/deprecated/model_src/data0/' + "transformer")
        input_ids = tvm.nd.array(A.numpy())

        module.set_input("input_ids", input_ids)
        print("Evaluate inference time cost...")
        module.run()



model_list = ["mobile_net","vgg16","resnet101","resnet152","densenet201","bert_large","inceptionv3","roberta","transformer"]

for m in model_list:
    main(m)