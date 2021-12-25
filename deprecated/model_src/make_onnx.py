import torch
import torch.nn as nn
import numpy as np
import torchvision
from tvm.contrib.download import download_testdata
import tvm
import tvm.relay

# batch_size_list = [48,80,112,144,176,208,240,272,304,336,368,400]
# inputsize_list = [20,40,60,80,120,140,160,180,200]

seq_len = [5,10,20,30,40,50,60,70]
for sl in seq_len:
# # for batch_size in batch_size_list:
# for inputsize in inputsize_list:
    # rnn = nn.LSTM(128, 256, 2)
    # input = torch.randn(50, batch_size, 128)
    # h0 = torch.randn(2, batch_size, 256)
    # c0 = torch.randn(2, batch_size, 256)
    # output, (hn, cn) = rnn(input, (h0, c0))
    # rnn = nn.LSTM(inputsize, 20, 2)
    # input = torch.randn(5, 4, inputsize)
    # h0 = torch.randn(2, 4, 20)
    # c0 = torch.randn(2, 4, 20)
    # output, (hn, cn) = rnn(input, (h0, c0))
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(sl, 4, 10)
    h0 = torch.randn(2, 4, 20)
    c0 = torch.randn(2, 4, 20)
    output, (hn, cn) = rnn(input, (h0, c0))


    # export onnx
    onnx_name = "lstm_seqlen_" + str(sl) + ".onnx"
    with torch.no_grad():
        torch.onnx.export(rnn, (input, (h0, c0)), './onnx/' + onnx_name, input_names=['input', 'h0', 'c0'],output_names=['output', 'hn', 'cn'])

    # rnn = nn.GRU(10, 20, 2)
    # input = torch.randn(5, batch_size, 10)
    # h0 = torch.randn(2, batch_size, 20)
    # output, hn = rnn(input, h0)
    #
    # onnx_name = "gru_" + str(batch_size) + ".onnx"
    # torch.onnx.export(rnn, (input, h0), './onnx/' + onnx_name,input_names=['input', 'h0'],output_names=['output', 'hn'])


