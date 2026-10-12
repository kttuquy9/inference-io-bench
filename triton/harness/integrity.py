#!/usr/bin/env python3
"""Chất lượng + rủi ro: dữ liệu có còn đúng từng byte qua ranh giới không.

Chạy trong container SDK (có tritonclient + numpy). In JSON ra stdout. Ba bài:
  echo      identity_fp32: gửi dữ liệu biết trước, nhiều luồng song song, output phải khớp tuyệt đối.
            Bắt lỗi request này thấy dữ liệu của request khác (vùng nhớ tái dùng sớm).
  hold_ref  model giữ mảng numpy qua request; mọi OUT phải = 0.
  mutate_in model ghi vào input; OUT cho biết vùng nhớ gốc có đổi theo không (hợp đồng hành vi).
Không phán PROVEN. Chỉ đếm và ghi.
"""
import json, os, sys, threading, time
import numpy as np
import tritonclient.http as http

URL = os.environ.get("TRITON_URL", "localhost:8000")
N = int(os.environ.get("INTEGRITY_REQUESTS", "2000"))      # mỗi cấu hình
THREADS = int(os.environ.get("INTEGRITY_THREADS", "8"))
SIZES = [int(x) for x in os.environ.get("INTEGRITY_ELEMS", "12288 1555200").split()]
BATCHES = [int(x) for x in os.environ.get("INTEGRITY_BATCH", "1 16").split()]
# trần dữ liệu mỗi cấu hình: tensor lớn thì ít request hơn, nhưng không dưới 200
MAX_BYTES = float(os.environ.get("INTEGRITY_BYTES", "20e9"))


def n_requests(elems, batch):
    return min(N, max(200, int(MAX_BYTES / (elems * 4 * batch))))


def pattern(seq, batch, elems):
    # giá trị khác nhau theo request và theo mẫu, bắt đầu từ 1 (mutate_in cần phần tử khác 0).
    # float32 biểu diễn chính xác số nguyên < 2^24 -> so sánh tuyệt đối được.
    base = np.arange(1, elems + 1, dtype=np.float32) % 4096
    rows = [(base + ((seq * batch + r) % 4096)) for r in range(batch)]
    return np.stack(rows).astype(np.float32)


def run(model, elems, batch, check):
    """check(seq, sent, out) -> True nếu đúng. Trả số request, số sai, lỗi, thời gian."""
    total = n_requests(elems, batch)
    counter = {"next": 0, "bad": 0, "err": 0, "outs": []}
    lock = threading.Lock()

    def worker():
        cli = http.InferenceServerClient(URL, connection_timeout=60, network_timeout=300)
        while True:
            with lock:
                seq = counter["next"]
                counter["next"] += 1
            if seq >= total:
                return
            x = pattern(seq, batch, elems)
            inp = http.InferInput("IN", list(x.shape), "FP32")
            inp.set_data_from_numpy(x, binary_data=True)
            try:
                res = cli.infer(model, [inp], outputs=[http.InferRequestedOutput("OUT", binary_data=True)])
                out = res.as_numpy("OUT")
                ok = check(seq, x, out)
                with lock:
                    if not ok:
                        counter["bad"] += 1
                    if len(counter["outs"]) < 5000:
                        counter["outs"].append(float(out.reshape(-1)[0]))
            except Exception as e:
                with lock:
                    counter["err"] += 1
                    counter.setdefault("first_error", str(e)[:300])

    t = time.time()
    ts = [threading.Thread(target=worker) for _ in range(THREADS)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    vals = counter["outs"]
    return {"model": model, "elems": elems, "batch": batch, "requests": total, "threads": THREADS,
            "wrong": counter["bad"], "errors": counter["err"],
            "first_error": counter.get("first_error"),
            "out_values": sorted(set(vals))[:10],
            "seconds": round(time.time() - t, 1)}


NP = {"FP16": np.float16, "FP32": np.float32, "FP64": np.float64, "INT8": np.int8, "UINT8": np.uint8,
      "INT16": np.int16, "INT32": np.int32, "INT64": np.int64, "BOOL": np.bool_}
DT_REQUESTS = int(os.environ.get("DT_REQUESTS", "300"))


def dt_pattern(seq, elems, t):
    # giá trị đổi theo request, nằm trong miền của mọi kiểu (int8: 0..99), biểu diễn chính xác ở fp16
    v = (np.arange(elems, dtype=np.int64) + seq) % 100
    return (v % 2 == 0).reshape(1, -1) if t == "BOOL" else v.astype(NP[t]).reshape(1, -1)


def run_dtype(t, elems):
    """dt_check_<kiểu>: OUT phải khớp tuyệt đối input; HELD phải = 1 (mảng đang cầm không bị đổi)."""
    cli = http.InferenceServerClient(URL, connection_timeout=60, network_timeout=300)
    model = "dt_check_" + t.lower()
    wrong = held_bad = err = 0
    first_error = None
    for seq in range(DT_REQUESTS):
        x = dt_pattern(seq, elems, t)
        inp = http.InferInput("IN", list(x.shape), t)
        inp.set_data_from_numpy(x, binary_data=True)
        try:
            r = cli.infer(model, [inp], outputs=[http.InferRequestedOutput("OUT", binary_data=True),
                                                 http.InferRequestedOutput("HELD", binary_data=True)])
            if not np.array_equal(r.as_numpy("OUT"), x):
                wrong += 1
            if float(r.as_numpy("HELD")[0, 0]) != 1.0:
                held_bad += 1
        except Exception as e:
            err += 1
            first_error = first_error or str(e)[:300]
    return {"model": model, "dtype": t, "elems": elems, "requests": DT_REQUESTS,
            "wrong": wrong, "held_changed": held_bad, "errors": err, "first_error": first_error}


def main():
    report = {"url": URL, "tests": []}
    for e in SIZES:
        for b in BATCHES:
            report["tests"].append(run("identity_fp32", e, b,
                                       lambda s, x, o: o.shape == x.shape and np.array_equal(o, x)))
            report["tests"].append(run("hold_ref", e, b, lambda s, x, o: bool((o == 0).all())))
    # mutate_in: không có đúng/sai cố định — ghi giá trị quan sát (0 = vùng gốc không đổi, 1 = có đổi)
    report["tests"].append(run("mutate_in", SIZES[0], 1, lambda s, x, o: True))
    dtypes = os.environ.get("DTYPES", "").split()
    report["dtype_tests"] = [run_dtype(t, e) for t in dtypes for e in SIZES]
    report["all_dtypes_ok"] = all(d["wrong"] == 0 and d["held_changed"] == 0 and d["errors"] == 0
                                  for d in report["dtype_tests"]) if dtypes else None
    report["all_echo_exact"] = all(t["wrong"] == 0 and t["errors"] == 0
                                   for t in report["tests"] if t["model"] in ("identity_fp32", "hold_ref"))
    # ghi thẳng ra file: stdout của container bị banner NVIDIA trộn vào -> JSON hỏng
    with open(os.environ.get("OUT", "/dev/stdout"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
