import torch
import torch.nn as nn
import numpy as np
import torchvision
from tvm.contrib.download import download_testdata
import tvm
import tvm.relay

batch_size_list = [4,8,16,32,64]

for batch_size in batch_size_list:

    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, batch_size, 10)
    h0 = torch.randn(2, batch_size, 20)
    c0 = torch.randn(2, batch_size, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    
    
    
    # export onnx
    onnx_name = "lstm_" + str(batch_size) + ".onnx"
    with torch.no_grad():
        torch.onnx.export(rnn, (input, (h0, c0)), './onnx/' + onnx_name, input_names=['input', 'h0', 'c0'],output_names=['output', 'hn', 'cn'])

    # rnn = nn.GRU(10, 20, 2)
    # input = torch.randn(5, batch_size, 10)
    # h0 = torch.randn(2, batch_size, 20)
    # output, hn = rnn(input, h0)

    # onnx_name = "gru_" + str(batch_size) + ".onnx"
    # torch.onnx.export(rnn, (input, h0), './onnx/' + onnx_name,input_names=['input', 'h0'],output_names=['output', 'hn'])