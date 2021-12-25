import numpy as np
import tvm
from tvm import te

comp = lambda a, b: a * b
init = lambda dtype: tvm.tir.const(1, dtype=dtype)
product = te.comm_reducer(comp, init)

n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='a')
k = te.reduce_axis((0, m), name='k')
B = te.compute((n,), lambda i: product(A[i, k], axis=k), name='b')
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B])
b = tvm.nd.array(np.empty((3,), dtype='float32'))
a = np.random.normal(size=(3,4)).astype('float32')
mod(tvm.nd.array(a), b)
np.testing.assert_allclose(a.prod(axis=1), b.asnumpy(), atol=1e-5)

print(b)