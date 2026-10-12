#!/usr/bin/env python3
"""Kiểm rò shared memory, mở rộng nguyên lý ShmLeakDetector của Triton (server/qa/common/shm_util.py).
Đo bộ nhớ ĐÃ DÙNG = kích thước vùng - bộ nhớ trống, trong từng vùng triton_python_backend_shm_region_*,
trước và sau một loạt request. Sau nhiều hơn trước = rò. Dùng module triton_shm_monitor.
Không chỉ so bộ nhớ trống như ShmLeakDetector: đối chứng dương 4/10 cho thấy vùng shm tự nở ra khi bị giữ thêm
khối (7.3 -> 38.8 MB) và bộ nhớ trống lại TĂNG -> chỉ so bộ nhớ trống sẽ bỏ sót.

Workload chỉ gồm model KHÔNG cầm input lại (như bài của Triton). Model cố ý cầm mảng (hold_ref, dt_check,
hold_k) giữ khối shm là đúng thiết kế, không phải rò -> đo riêng ở bài `hold`.
Chạy trong container SDK (--ipc=host). Ghi JSON ra $OUT.
"""
import json, os, sys, time
import numpy as np
import tritonclient.http as http

sys.path.insert(0, os.environ.get("MONITOR_DIR", "/m"))
import triton_shm_monitor  # noqa: E402

URL = os.environ.get("TRITON_URL", "localhost:8000")
PREFIX = "triton_python_backend_shm_region"
EXIT_DELAY = float(os.environ.get("LEAK_EXIT_DELAY", "2"))
# (model, số phần tử mỗi mẫu, batch, số request, tên input)
WORKLOAD = [
    ("identity_fp32", 1555200, 1, 200, "IN"),
    ("identity_fp32", 1555200, 16, 20, "IN"),
    ("identity_in", 1555200, 1, 200, "IN"),
    ("identity_in", 12288, 16, 200, "IN"),
    ("py_gpu", 150528, 4, 100, "IN"),
]
# Đối chứng dương (LEAK_POSITIVE_CONTROL=1): thêm model CỐ Ý cầm dồn mảng 1080p (hold_k_100, không trượt trong
# 6 request). Làm nóng cầm 1, workload thêm 5 -> với bản zero-copy phải giữ thêm 5 khối shm (~31 MB), thước PHẢI
# thấy bộ nhớ trống giảm. Không thấy = thước không nhạy, kết quả "không rò" vô giá trị.
# (Lần đầu dùng hold_ref: nó chỉ cầm 1 mảng trượt -> trước/sau đều giữ 1 khối, không đổi -> đối chứng sai.)
if os.environ.get("LEAK_POSITIVE_CONTROL") == "1":
    WORKLOAD.append(("hold_k_100", 1555200, 1, 5, "IN"))


def free_sizes():
    """{vùng: (bộ nhớ trống, kích thước file)} — kích thước để thấy vùng có tự nở ra giữa chừng không."""
    out = {}
    for name in sorted(os.listdir("/dev/shm")):
        if name.startswith(PREFIX):
            out[name] = (triton_shm_monitor.SharedMemoryManager(name).free_memory(),
                         os.stat("/dev/shm/" + name).st_size)
    return out


def main():
    cli = http.InferenceServerClient(URL, network_timeout=300)
    # 1 lượt làm nóng cho mọi model: vùng shm đã nở tới kích thước làm việc trước khi đo
    for model, elems, batch, _, iname in WORKLOAD:
        x = np.ones((batch, elems), dtype=np.float32)
        i = http.InferInput(iname, list(x.shape), "FP32"); i.set_data_from_numpy(x, binary_data=True)
        cli.infer(model, [i])
    time.sleep(EXIT_DELAY)
    before = free_sizes()
    sent = 0
    for model, elems, batch, n, iname in WORKLOAD:
        for s in range(n):
            x = np.full((batch, elems), float(s % 97), dtype=np.float32)
            i = http.InferInput(iname, list(x.shape), "FP32"); i.set_data_from_numpy(x, binary_data=True)
            cli.infer(model, [i]); sent += 1
    time.sleep(EXIT_DELAY)
    after = free_sizes()
    regions = []
    for r in sorted(set(before) | set(after)):
        (b, bs), (a, as_) = before.get(r, (None, None)), after.get(r, (None, None))
        used_b = (bs - b) if None not in (bs, b) else None
        used_a = (as_ - a) if None not in (as_, a) else None
        regions.append({"region": r, "free_before": b, "free_after": a, "size_before": bs, "size_after": as_,
                        "used_before": used_b, "used_after": used_a,
                        "leak_bytes": (used_a - used_b) if None not in (used_a, used_b) and used_a > used_b else 0})
    report = {"requests": sent, "workload": WORKLOAD, "regions": regions,
              "leak_detected": any(x["leak_bytes"] > 0 for x in regions)}
    with open(os.environ.get("OUT", "/dev/stdout"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
