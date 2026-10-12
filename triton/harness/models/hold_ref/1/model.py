# Giữ mảng numpy qua request (không copy) rồi kiểm nó còn nguyên không.
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.prev = None
        self.prev_sum = None

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            bad = 0.0
            if self.prev is not None and float(self.prev.sum(dtype=np.float64)) != self.prev_sum:
                bad = 1.0
            self.prev = arr
            self.prev_sum = float(arr.sum(dtype=np.float64))
            out = np.full((arr.shape[0], 1), bad, dtype=np.float32)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUT", out)]))
        return responses
