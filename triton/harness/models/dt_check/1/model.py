# Trả lại nguyên input và kiểm mảng của request trước (đang cầm) còn nguyên không.
# Giữ thêm một bản copy riêng của mảng trước để so: copy này chỉ phục vụ phép kiểm, không phải đường đang đo.
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.prev = None
        self.prev_copy = None

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            ok = 1.0
            if self.prev is not None and not np.array_equal(self.prev, self.prev_copy):
                ok = 0.0
            self.prev, self.prev_copy = arr, arr.copy()
            held = np.full((arr.shape[0], 1), ok, dtype=np.float32)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("OUT", np.ascontiguousarray(arr)), pb_utils.Tensor("HELD", held)]))
        return responses
