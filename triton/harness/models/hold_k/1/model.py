# Cầm HOLD_COUNT mảng input gần nhất; mỗi request kiểm MỘT mảng đang cầm (xoay vòng), để chi phí kiểm
# không tăng theo HOLD_COUNT (kiểm hết mỗi request: 100 mảng x 6 MB = ~600 MB crc/request, quá chậm).
import collections, json, zlib
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        n = int(json.loads(args["model_config"])["parameters"]["HOLD_COUNT"]["string_value"])
        self.held = collections.deque(maxlen=n)   # (mảng, crc32 của các byte lúc nhận)
        self.turn = 0

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            # so theo byte (crc32): dữ liệu ngẫu nhiên của perf_analyzer có NaN, so theo tổng thì NaN != NaN -> báo sai
            ok = True
            if self.held:
                a, c = self.held[self.turn % len(self.held)]
                ok = zlib.crc32(a.tobytes()) == c
                self.turn += 1
            if not ok:
                print("HOLDK_BAD", flush=True)   # perf_analyzer không đọc output -> báo qua log Triton
            self.held.append((arr, zlib.crc32(arr.tobytes())))
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("HELD", np.full((arr.shape[0], 1), 1.0 if ok else 0.0, dtype=np.float32))]))
        return responses
