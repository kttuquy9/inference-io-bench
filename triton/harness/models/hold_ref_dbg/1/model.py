# Như hold_ref, thêm log mỗi request (in ra log của Triton) để soi vùng nhớ bị dùng lại:
#   seq | địa chỉ mảng mới | địa chỉ mảng đang giữ | mảng giữ còn nguyên không | refcount của mảng giữ
import sys
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.prev = None
        self.prev_sum = None
        self.seq = 0

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            bad = 0.0
            held_addr = "-"
            if self.prev is not None:
                held_addr = hex(self.prev.ctypes.data)
                if float(self.prev.sum(dtype=np.float64)) != self.prev_sum:
                    bad = 1.0
            print(f"HOLDDBG seq={self.seq} new={hex(arr.ctypes.data)} held={held_addr} "
                  f"held_changed={int(bad)} held_refcnt={sys.getrefcount(self.prev) if self.prev is not None else 0} "
                  f"base={type(arr.base).__name__} got_first={float(arr.reshape(-1)[0])} "
                  f"held_first={float(self.prev.reshape(-1)[0]) if self.prev is not None else -1}", flush=True)
            self.prev = arr
            self.prev_sum = float(arr.sum(dtype=np.float64))
            self.seq += 1
            out = np.full((arr.shape[0], 1), bad, dtype=np.float32)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUT", out)]))
        return responses
