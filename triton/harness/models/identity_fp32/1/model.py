# Python backend identity. Vai trò: chạm vào input dưới dạng numpy (ép ranh giới B kích hoạt),
# rồi trả về. Không compute. Chi phí đo được = ranh giới trung chuyển, không phải model.
#
# Cờ đo (đặt qua parameters trong config, đọc bằng biến môi trường HARNESS_MODE):
#   touch  : gọi as_numpy() (mặc định) — kích hoạt py::array copy ở pb_tensor.cc:725
#   notouch: KHÔNG gọi as_numpy(), trả thẳng — dùng để đo phần copy tách khỏi phần còn lại (delta)
import os
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.mode = os.environ.get("HARNESS_MODE", "touch")

    def execute(self, requests):
        responses = []
        for request in requests:
            in_t = pb_utils.get_input_tensor_by_name(request, "IN")
            if self.mode == "touch":
                # as_numpy() trả &numpy_array_ đã dựng ở LoadFromSharedMemory (điểm B đã copy).
                arr = in_t.as_numpy()
                out_t = pb_utils.Tensor("OUT", arr)
            else:
                # Trả lại chính tensor object, tránh đọc numpy phía Python.
                out_t = pb_utils.Tensor("OUT", in_t.as_numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
        return responses
