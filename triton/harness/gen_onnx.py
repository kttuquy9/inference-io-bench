#!/usr/bin/env python3
"""Sinh model ONNX có compute GPU thật cho bài 'workload thật'.

Ảnh 224x224x3 float32 (150528 phần tử) -> Conv 3->64 -> Relu -> Conv 64->64 -> Relu -> mean -> [N,1].
Trọng số ngẫu nhiên cố định: không cần đúng, chỉ cần GPU phải làm việc thật vài ms mỗi batch.
Dùng: python3 gen_onnx.py <đường dẫn model.onnx>   (cần gói onnx; setup.sh cài trong container)
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(0)
w1 = numpy_helper.from_array(rng.standard_normal((64, 3, 3, 3)).astype(np.float32) * 0.1, "w1")
w2 = numpy_helper.from_array(rng.standard_normal((64, 64, 3, 3)).astype(np.float32) * 0.05, "w2")
shape = numpy_helper.from_array(np.array([-1, 3, 224, 224], dtype=np.int64), "shape")
nodes = [
    helper.make_node("Reshape", ["X", "shape"], ["img"]),
    helper.make_node("Conv", ["img", "w1"], ["c1"], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c1"], ["r1"]),
    helper.make_node("Conv", ["r1", "w2"], ["c2"], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c2"], ["r2"]),
    helper.make_node("ReduceMean", ["r2"], ["m"], axes=[1, 2, 3], keepdims=0),
    helper.make_node("Unsqueeze", ["m", "ax"], ["Y"]),
]
ax = numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
graph = helper.make_graph(
    nodes, "gpu_conv",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 150528])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 1])],
    initializer=[w1, w2, shape, ax])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
print("saved", sys.argv[1])
