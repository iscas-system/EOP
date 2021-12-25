#convert pytorch model to onnx

import torch
import torch.nn as nn
import numpy as np
import torchvision
from tvm.contrib.download import download_testdata
import tvm
import tvm.relay

#lstm

rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))



# export onnx
#with torch.no_grad():
torch.onnx.export(rnn, (input, (h0, c0)), 'lstm.onnx',input_names=['input', 'h0', 'c0'],output_names=['output', 'hn', 'cn'])

#gru

# rnn = nn.GRU(10, 20, 2)
# input = torch.randn(5, 1, 10)
# h0 = torch.randn(2, 1, 20)
# output, hn = rnn(input, h0)
#
# torch.onnx.export(rnn, (input, h0), 'gru.onnx',input_names=['input', 'h0'],output_names=['output', 'hn'])