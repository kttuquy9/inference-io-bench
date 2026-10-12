# Nhận ảnh dưới dạng numpy, chuyển nguyên cho gpu_conv qua BLS, trả kết quả. Không tiền xử lý thêm.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            sub = pb_utils.InferenceRequest(
                model_name="gpu_conv", requested_output_names=["Y"],
                inputs=[pb_utils.Tensor("X", arr)],
                # gpu_conv (onnxruntime GPU) trả output trên GPU; as_numpy chỉ đọc được CPU
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0))
            resp = sub.exec()
            if resp.has_error():
                raise pb_utils.TritonModelException(resp.error().message())
            y = pb_utils.get_output_tensor_by_name(resp, "Y").as_numpy()
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUT", y)]))
        return responses
