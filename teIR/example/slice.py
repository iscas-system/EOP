import numpy as np
import tvm
from tvm import te

n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='a')
bi, bj, si, sj = [te.var(name) for name in ['bi', 'bj', 'si', 'sj']]
B = te.compute(((n-bi)//si, (m-bj)//sj), lambda i, j: A[i*si+bi, j*sj+bj], name='b')
s = te.create_schedule(B.op)
mod = tvm.build(s, [A, B, bi, si, bj, sj])

a = tvm.nd.array(np.arange(12, dtype='float32').reshape((3,4)))
b = tvm.nd.array(np.empty((1, 3), dtype='float32'))
mod(a, b, 1, 2, 1, 1)
np.testing.assert_equal(b.asnumpy(), a.asnumpy()[1::2, 1::1])

print(a)
print(b)

b = tvm.nd.array(np.empty((1, 2), dtype='float32'))
mod(a, b, 2, 1, 0, 2)
np.testing.assert_equal(b.asnumpy(), a.asnumpy()[2::1, 0::2])
print(a)
print(b)