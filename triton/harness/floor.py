#!/usr/bin/env python3
"""Sàn vật lý: một lần chép CPU tốn bao nhiêu trên chính máy đo này.

Chạy trong container tritonserver (cùng numpy, cùng CPU với stub). In JSON ra stdout.
Hai cách chép, tách phần cứng khỏi phần thư viện:
  reuse : chép vào buffer có sẵn        -> băng thông RAM thuần (sàn của 1 lần chép)
  fresh : cấp buffer mới rồi chép        -> thêm page fault / zero-fill, giống PyArray_NewCopy
Nếu fresh chậm hơn reuse nhiều ở cỡ lớn -> mức phạt khi batch lớn đến từ việc cấp bộ nhớ mới
mỗi request, không phải từ băng thông.
"""
import json, os, platform, statistics, sys, time
import numpy as np

# cỡ tính bằng số phần tử float32; khớp config.env (crop, 224x224, 1080p) và batch 16 của 1080p
SIZES = [int(x) for x in os.environ.get("FLOOR_ELEMS", "12288 150528 1555200 24883200").split()]
REPEAT = int(os.environ.get("FLOOR_REPEAT", "30"))


def timed(fn, repeat):
    fn()  # warmup
    xs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t)
    return statistics.median(xs), statistics.pstdev(xs)


def main():
    out = {"cpu": platform.processor() or platform.machine(), "numpy": np.__version__,
           "repeat": REPEAT, "sizes": []}
    for n in SIZES:
        src = np.random.rand(n).astype(np.float32)
        dst = np.empty_like(src)
        nbytes = src.nbytes
        reuse, reuse_sd = timed(lambda: np.copyto(dst, src), REPEAT)
        fresh, fresh_sd = timed(lambda: src.copy(), REPEAT)
        out["sizes"].append({
            "elems": n, "bytes": nbytes,
            "reuse_us": round(reuse * 1e6, 1), "reuse_sd_us": round(reuse_sd * 1e6, 1),
            "reuse_gbps": round(nbytes / reuse / 1e9, 2),
            "fresh_us": round(fresh * 1e6, 1), "fresh_sd_us": round(fresh_sd * 1e6, 1),
            "fresh_gbps": round(nbytes / fresh / 1e9, 2),
            "fresh_over_reuse": round(fresh / reuse, 2),
        })
    # ghi thẳng ra file: stdout của container bị banner NVIDIA trộn vào -> JSON hỏng
    with open(os.environ.get("OUT", "/dev/stdout"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
