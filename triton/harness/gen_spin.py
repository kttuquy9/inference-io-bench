#!/usr/bin/env python3
"""Sinh model ONNX GPU tốn thời gian cố định cho bản đồ chi phí (cm_gpu gọi qua BLS).

Nhận X [1,1] (một số), nhân với ma trận 2048x2048 rồi chạy N lần MatMul 2048x2048 + Relu, trả trung bình [1,1].
Mỗi MatMul ~17 GFLOP. Thời gian thật trên GPU đo hiệu chuẩn bằng harness (bài costmap đo thẳng cm_spin_d*),
không lấy theo con số tính tay này.
Dùng: python3 gen_spin.py <thư mục models> <N1> [N2 ...]   -> models/cm_spin_d<N>/1/model.onnx + config.pbtxt
"""
import os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root, depths = sys.argv[1], [int(x) for x in sys.argv[2:]]
rng = np.random.default_rng(0)
A = numpy_helper.from_array((rng.standard_normal((2048, 2048)) * 0.01).astype(np.float32), "A")
W = numpy_helper.from_array((rng.standard_normal((2048, 2048)) * 0.02).astype(np.float32), "W")
for n in depths:
    nodes = [helper.make_node("Mul", ["A", "X"], ["h0"])]
    for i in range(n):
        nodes.append(helper.make_node("MatMul", [f"h{i}", "W"], [f"m{i}"]))
        nodes.append(helper.make_node("Relu", [f"m{i}"], [f"h{i+1}"]))
    nodes.append(helper.make_node("ReduceMean", [f"h{n}"], ["Y"], keepdims=1))
    g = helper.make_graph(nodes, f"cm_spin_d{n}",
                          [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
                          initializer=[A, W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    d = os.path.join(root, f"cm_spin_d{n}")
    os.makedirs(os.path.join(d, "1"), exist_ok=True)
    onnx.save(m, os.path.join(d, "1", "model.onnx"))
    with open(os.path.join(d, "config.pbtxt"), "w") as f:
        f.write(f'# Sinh bởi gen_spin.py: {n} MatMul 2048x2048 trên GPU (bản đồ chi phí)\n'
                'backend: "onnxruntime"\nmax_batch_size: 0\n'
                'input [ { name: "X" data_type: TYPE_FP32 dims: [ 1, 1 ] } ]\n'
                'output [ { name: "Y" data_type: TYPE_FP32 dims: [ 1, 1 ] } ]\n'
                'instance_group [ { count: 1 kind: KIND_GPU } ]\n')
    print("saved", d)
