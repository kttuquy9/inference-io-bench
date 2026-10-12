# Stage 3 OCR. TRACKS_IN(float32) -> LABELS(BYTES). Sinh N chuỗi để ép serialize_byte_tensor.
import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        out=[]
        for request in requests:
            t=pb_utils.get_input_tensor_by_name(request,"TRACKS_IN").as_numpy()
            n=max(1,min(64,t.size//64))
            labels=np.array([f"{i%100:02d}".encode("utf-8") for i in range(n)],dtype=object)
            out.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("LABELS",labels)]))
        return out
