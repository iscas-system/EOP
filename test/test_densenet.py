import sys
from tvm.relay.testing import densenet
import tvm
import numpy as np
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download_testdata

import numpy as np
import cv2

target = "llvm"
dev = tvm.cpu()
mod, params = densenet.get_workload(classes=2, batch_size=1, image_shape=(1, 224, 224))
exe = relay.vm.compile(mod, target, params=params)
vm = profiler_vm.VirtualMachineProfiler(exe, dev)
data = tvm.nd.array(np.random.rand(1, 1, 224, 224).astype("float32"), device=dev)
report = vm.profile(
    [data],
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
)
print(report)
print("ok")
