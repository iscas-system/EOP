import sys

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import autotvm, relay, te
from tvm.contrib import utils
from tvm.ir.module import IRModule
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.topi.cuda.conv3d_winograd import _infer_tile_size

n, c, w = te.var("n"), 10, 224
x = relay.var("x", relay.ty.TensorType((n, c, w), "float32"))
w = relay.var("w")
y = relay.nn.conv1d(x, w, kernel_size=3, padding=(1, 1), channels=2)
yy = run_infer_type(y)
print(yy.args)
print(yy.checked_type)
assert yy.checked_type == relay.TensorType((n, 2, 224), "float32")
assert yy.args[1].checked_type == relay.TensorType((2, 10, 3), "float32")