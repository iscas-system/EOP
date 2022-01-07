import subprocess
import os
import sys
import posixpath
from typing import Sequence
from six.moves.urllib.request import urlretrieve
import glob

import onnx
import numpy as np
import tvm
import tvm.relay as relay
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
from tvm.runtime.vm import VirtualMachine
from onnx.tools import update_model_dims

base_path = '/root/github/onnx-models/'

batch = 1
sequence = 128

def get_value_info_shape(value_info):
    return tuple([max(d.dim_value, 1) for d in value_info.type.tensor_type.shape.dim])

urls = [
    # 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.tar.gz',
    # 'https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz',
    # 'https://github.com/onnx/models/raw/master/text/machine_comprehension/roberta/model/roberta-base-11.tar.gz',
    # XXX: Often segfaults
    # 'https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz',
    # checked:
    'https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz'
    # 'https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-encoder-12.tar.gz'
]
target = tvm.target.Target("cuda")
# target = "llvm"
dev = tvm.device(str(target), 0)
ctx = tvm.device(str(target), 0)
network = ''
name = ''
summary = []
for url in urls:
    print(f'==> {url} <==')

    archive = posixpath.basename(url)
    archive = base_path + archive
    if not os.path.exists(archive):
        print(f'Downloading {url} ...')
        urlretrieve(url, archive)
        assert os.path.exists(archive)
    else:
        print('Already downloaded.')

    import tarfile
    tar = tarfile.open(archive, 'r:gz')
    for n in tar.getnames():
        if n.endswith('.onnx'):
            model_file = base_path + n.split('.onnx')[0] +'/' + n
            name = os.path.dirname(model_file)
            network = n.split('.')[0]
            print('onnx location:' + model_file)
            print('extraction path:' + name)
            break

    if not os.path.exists(model_file):
        print(f'Extracting {archive} ...')
        #subprocess.call(['tar', 'xzf', archive])
        tar.extractall(name)
        assert os.path.exists(model_file)

    print(f'Loading {model_file} ...')
    onnx_model = onnx.load(model_file)

    initializers = set()
    for initializer in onnx_model.graph.initializer:
        initializers.add(initializer.name)
    shape_dict = {}
    input_values = []
    inputs = {}
    # print("input data:")
    # for input in onnx_model.graph.input:
    #     print(input)
    # print("output data:")
    # for out in onnx_model.graph.output:
    #     print(out)
    if(len(glob.glob(os.path.join(name, 'test_data_set_*'))) > 0):
        test_data_set = glob.glob(os.path.join(name, 'test_data_set_*'))[0]
        assert os.path.exists(test_data_set)
        for input in onnx_model.graph.input:
            if input.name not in initializers:
                i = len(input_values)
                input_data = os.path.join(test_data_set, f'input_{i}.pb')
                tensor = onnx.TensorProto()
                input_data = open(input_data, 'rb').read()
                tensor.ParseFromString(input_data)
                x = onnx.numpy_helper.to_array(tensor)
                input_values.append(x)
                shape_dict[input.name] = x.shape
                inputs[input.name] = tvm.nd.array(x, ctx)
        print(f'Input shapes: {shape_dict}')
    elif name.endswith('t5-decoder-with-lm-head'):
        onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, {'input_ids': [batch, sequence],'encoder_hidden_states': [batch, sequence,768]},{'hidden_states':[batch, sequence, 32128]})
        # shape_dict['input_ids'] = (batch,sequence)
        # shape_dict['encoder_hidden_states'] = (batch,sequence,768)
        # shape_dict['hidden_states'] = (batch,sequence,32128)
        inputs['input_ids'] = tvm.nd.array((np.random.uniform(size=(batch,sequence))).astype("int64"))
        inputs['encoder_hidden_states'] = tvm.nd.array((np.random.uniform(size=(batch,sequence,768))).astype("float32"))
    elif name.endswith('t5-encoder'):
        # print(dir(onnx_model.graph))
        onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, {'input_ids': [batch, sequence]},{'hidden_states':[batch, sequence, 768]})
        inputs['input_ids'] = tvm.nd.array((np.random.uniform(size=(batch,sequence))).astype("int64"))
    try:
        print(f'Importing graph from ONNX to TVM Relay IR ...')
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
        mod = relay.transform.DynamicToStatic()(mod)

        print(f'Compiling graph from Relay IR to {target} ...')
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(mod, target=target, params=params)
            # vm_exec = relay.vm.compile(mod, target, params=params)
            # module = graph_executor.GraphModule(lib["default"](dev))
            module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), dev, dump_root = '/root/github/debug_dump/' + network)
        # vm = VirtualMachine(vm_exec, dev)
        if name.endswith('t5-decoder-with-lm-head'):
            module.set_input("input_ids", inputs['input_ids'])
            module.set_input("encoder_hidden_states", inputs['encoder_hidden_states'])
        elif name.endswith('t5-encoder'):
            module.set_input("input_ids", inputs['input_ids'])
        print(f"Running inference...")
        module.run()
        # ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        # prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(f'Caught an exception {ex}')
        result = 'not ok'
    else:
        print(f'Succeeded!')
        result = 'ok'
    summary.append((result, url))
    print()

print('Summary:')
for result, url in summary:
    print(f'{result}\t- {url}')