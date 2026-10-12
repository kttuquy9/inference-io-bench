# Nhận ảnh dưới dạng numpy, gọi model GPU (GPU_MODEL) một lần mỗi execute qua BLS, trả 1 số mỗi mẫu.
# GPU_MODEL rỗng = không gọi GPU (mốc 0 ms). Model GPU nhận 1 số (không chuyển ảnh lên GPU):
# đo đúng "chuyển ảnh qua ranh giới python_backend + thời gian GPU", không lẫn PCIe của ảnh.
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        self.gpu_model = cfg["parameters"]["GPU_MODEL"]["string_value"]

    def execute(self, requests):
        if self.gpu_model:
            sub = pb_utils.InferenceRequest(
                model_name=self.gpu_model, requested_output_names=["Y"],
                inputs=[pb_utils.Tensor("X", np.ones((1, 1), dtype=np.float32))],
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0))
            resp = sub.exec()
            if resp.has_error():
                raise pb_utils.TritonModelException(resp.error().message())
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            out = np.ascontiguousarray(arr[:, :1])
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUT", out)]))
        return responses
