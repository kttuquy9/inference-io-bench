# Input-only. Chạm input dưới dạng numpy rồi trả 1 số mỗi mẫu. Không compute.
# Mọi ms phía server = đưa input qua ranh giới (chép lần 1 ở core, lần 2 ở pb_tensor.cc:725).
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            # đọc 1 phần tử mỗi mẫu: chắc chắn chạm dữ liệu, không thêm chi phí đáng kể
            out = np.ascontiguousarray(arr[:, :1])
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUT", out)]))
        return responses
