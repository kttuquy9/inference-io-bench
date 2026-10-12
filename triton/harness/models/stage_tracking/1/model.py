# Stage 2 tracking. BOXES_IN -> TRACKS.
import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        out=[]
        for request in requests:
            b=pb_utils.get_input_tensor_by_name(request,"BOXES_IN").as_numpy()
            t=(b.reshape(-1)+1.0).astype(np.float32)
            out.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRACKS",t)]))
        return out
