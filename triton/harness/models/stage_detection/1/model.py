# Leaf stage. Nhận FRAME, trả BOXES. Compute tối thiểu để chi phí ranh giới chiếm ưu thế.
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        out = []
        for request in requests:
            frame = pb_utils.get_input_tensor_by_name(request, "FRAME").as_numpy()
            boxes = frame.reshape(-1)[: max(4, frame.size // 16)].astype(np.float32)
            out.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("BOXES", boxes)]))
        return out
