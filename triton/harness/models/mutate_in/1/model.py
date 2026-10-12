# Ghi vào input numpy rồi xem vùng nhớ gốc của tensor có đổi theo không.
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_t = pb_utils.get_input_tensor_by_name(request, "IN")
            arr = in_t.as_numpy()
            before = float(arr[0, 0])
            arr *= 2
            try:
                raw = np.from_dlpack(in_t)
                seen = 1.0 if float(raw.reshape(arr.shape)[0, 0]) != before else 0.0
            except Exception:
                seen = -1.0   # không đọc được vùng gốc qua DLPack -> ghi nhận, không đoán
            out = np.full((arr.shape[0], 1), seen, dtype=np.float32)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUT", out)]))
        return responses
