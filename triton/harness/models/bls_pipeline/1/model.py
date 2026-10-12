# BLS chain detection -> tracking -> ocr. Mỗi bước gọi InferenceRequest().exec():
# đi từ stub xuống core rồi lên stub kế — kích hoạt điểm C (message + 2 wait + future.get chặn thread).
# Mỗi tensor qua ranh giới cũng kích hoạt điểm B (py::array copy) và bước cuối điểm A (BYTES serialize).
import triton_python_backend_utils as pb_utils


def out(resp, name):
    # request con lỗi -> báo đúng lỗi của nó, không để rơi xuống thành None.as_numpy()
    if resp.has_error():
        raise pb_utils.TritonModelException(resp.error().message())
    return pb_utils.get_output_tensor_by_name(resp, name)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            frame = pb_utils.get_input_tensor_by_name(request, "FRAME")

            det = pb_utils.InferenceRequest(
                model_name="stage_detection",
                requested_output_names=["BOXES"],
                inputs=[pb_utils.Tensor("FRAME", frame.as_numpy())],
            ).exec()
            boxes = out(det, "BOXES")

            trk = pb_utils.InferenceRequest(
                model_name="stage_tracking",
                requested_output_names=["TRACKS"],
                inputs=[pb_utils.Tensor("BOXES_IN", boxes.as_numpy())],
            ).exec()
            tracks = out(trk, "TRACKS")

            ocr = pb_utils.InferenceRequest(
                model_name="stage_ocr",
                requested_output_names=["LABELS"],
                inputs=[pb_utils.Tensor("TRACKS_IN", tracks.as_numpy())],
            ).exec()
            labels = out(ocr, "LABELS")

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("LABELS", labels.as_numpy())]))
        return responses
