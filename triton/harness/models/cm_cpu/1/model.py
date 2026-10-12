# Nhận ảnh dưới dạng numpy (đi qua đúng đường chép/không chép đang đo), đốt CPU đúng WORK_MS ms mỗi execute,
# trả 1 số mỗi mẫu. Đốt bằng vòng lặp theo đồng hồ: giữ GIL như tiền xử lý Python thật.
import json, time
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        self.work_s = float(cfg["parameters"]["WORK_MS"]["string_value"]) / 1000.0

    def execute(self, requests):
        responses = []
        end = time.perf_counter() + self.work_s
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            out = np.ascontiguousarray(arr[:, :1])
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUT", out)]))
        while time.perf_counter() < end:
            pass
        return responses
