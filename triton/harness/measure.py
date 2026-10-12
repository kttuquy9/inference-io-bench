#!/usr/bin/env python3
"""Tổng hợp số đo -> results/<mode>.json.

Thước phải tự chứng minh (sàn, nhiễu, warmup, tự kiểm) trước khi dùng số của nó.
- noise floor: gộp MEASURE_RUNS lần, tính mean/std/p99. Delta nhỏ hơn noise = KHÔNG có số.
- calibration: đọc GPU/driver/container từ results/_env.txt.
- tự-kiểm: tỉ lệ compute_input_duration / tổng — chỉ ra ranh giới chiếm bao nhiêu %.
Chỉ đọc stdlib. Không sửa gì, không quyết PROVEN thay người — chỉ tính và gắn cờ.
"""
import csv, glob, json, os, re, statistics, sys, hashlib, time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"


def env():
    p = os.path.join(R, "_env.txt")
    if not os.path.exists(p):
        return {"gpu": "?", "driver": "?"}
    line = open(p).read().strip().splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    return {"gpu": parts[0], "vram": parts[1] if len(parts) > 1 else "?",
            "driver": parts[2] if len(parts) > 2 else "?"}


def read_pa(tag):
    """perf_analyzer CSV: lấy p50/p90/p99 (us) + breakdown compute_input nếu có."""
    f = os.path.join(R, f"pa_{tag}.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    row = rows[-1]
    g = lambda k: float(row[k]) if k in row and row[k] not in ("", None) else None
    return {
        "p50_us": g("p50 latency"), "p90_us": g("p90 latency"),
        "p99_us": g("p99 latency"), "throughput": g("Inferences/Second"),
        "compute_input_us": g("Server Compute Input"),
        "compute_infer_us": g("Server Compute Infer"),
        "compute_output_us": g("Server Compute Output"),
        "queue_us": g("Server Queue"),
        "client_send_us": g("Client Send"), "network_us": g("Network+Server Send/Recv"),
        "client_recv_us": g("Client Recv"),
    }


def agg(tags):
    """Gộp nhiều lần chạy -> mean/std/p99 cho từng metric (noise floor)."""
    keys = ["p50_us", "p99_us", "throughput", "compute_input_us",
            "compute_infer_us", "compute_output_us", "queue_us",
            "client_send_us", "network_us", "client_recv_us"]
    vals = {k: [] for k in keys}
    n = 0
    for t in tags:
        r = read_pa(t)
        if not r:
            continue
        n += 1
        for k in keys:
            if r.get(k) is not None:
                vals[k].append(r[k])
    out = {}
    for k, xs in vals.items():
        if xs:
            out[k] = {"mean": round(statistics.mean(xs), 2),
                      "std": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0,
                      "n": len(xs)}
    out["_runs"] = n
    return out


def boundary_ratio(a):
    """Tự-kiểm: compute_input (chứa chi phí ranh giới B) trên tổng compute."""
    ci = a.get("compute_input_us", {}).get("mean")
    inf = a.get("compute_infer_us", {}).get("mean")
    co = a.get("compute_output_us", {}).get("mean")
    if ci is None:
        return None
    total = sum(x for x in [ci, inf, co] if x)
    return {"compute_input_us": ci, "compute_infer_us": inf,
            "compute_output_us": co,
            "input_pct_of_compute": round(100 * ci / total, 1) if total else None}


def collect(prefix):
    """Tìm mọi tag khớp prefix (mọi run) để gộp noise."""
    tags = set()
    for f in glob.glob(os.path.join(R, f"pa_{prefix}*.csv")):
        m = re.search(r"pa_(.+)\.csv$", os.path.basename(f))
        if m:
            tags.add(m.group(1))
    return sorted(tags)


def usd_per_hour():
    p = os.path.join(R, "_cost.txt")
    if os.path.exists(p):
        m = re.search(r"USD_PER_HOUR=([\d.]+)", open(p).read())
        if m:
            return float(m.group(1))
    return None


def floor():
    """phần cứng: sàn memcpy đo trên chính máy này (floor.py). Trả {elems: row}."""
    p = os.path.join(R, "floor.json")
    if not os.path.exists(p):
        return {}
    try:
        return {s["elems"]: s for s in json.load(open(p))["sizes"]}
    except (ValueError, KeyError):
        return {}


def shape_of(group):
    """Lấy (elems, batch) từ tên nhóm. Nhóm cũ idshm_b*_c* / inst_* dùng frame 1080p."""
    large = int(os.environ.get("TENSOR_ELEMS_LARGE", "1555200"))
    m = re.search(r"_s(\d+)_b(\d+)$", group)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_b(\d+)_c\d+$", group)
    if m:
        return large, int(m.group(1))
    if group.startswith(("inst_", "asym_large", "bls", "blsshm")):
        return large, 1
    if group.startswith("asym_small"):
        return int(os.environ.get("SWEEP_SMALL", "12288")), 1
    m = re.search(r"^gpu_\w+_b(\d+)$", group)
    if m:
        return 150528, int(m.group(1))
    return None, None


def faults_per_inf(tags, thr):
    """page fault của stub mỗi lần suy luận = delta fault / (thời gian x throughput)."""
    xs = []
    for t in tags:
        p = os.path.join(R, f"flt_{t}.txt")
        if not os.path.exists(p):
            continue
        lines = [l.split() for l in open(p).read().split("\n") if l.strip()]
        if len(lines) < 2 or not thr:
            continue
        dt = float(lines[1][0]) - float(lines[0][0])
        df = float(lines[1][1]) - float(lines[0][1])
        if dt > 0:
            xs.append(df / (dt * thr))
    return round(statistics.mean(xs), 1) if xs else None


def gpu_util(tags):
    """phần cứng: GPU bận bao nhiêu % trong lúc đo (nvidia-smi). GPU rỗi = ranh giới đang bỏ đói GPU."""
    xs = []
    for t in tags:
        p = os.path.join(R, f"gpu_{t}.csv")
        if not os.path.exists(p):
            continue
        for row in csv.reader(open(p)):
            try:
                xs.append(float(row[1]))
            except (IndexError, ValueError):
                pass
    return round(statistics.mean(xs), 1) if xs else None


def physics(group, a, fl, tags, price):
    """Khoảng cách tới sàn + tiền cho một nhóm. Sàn = 1 lần chép reuse của đúng số byte mỗi mẫu."""
    elems, batch = shape_of(group)
    thr = a.get("throughput", {}).get("mean")
    srv = sum(a.get(k, {}).get("mean") or 0 for k in
              ("compute_input_us", "compute_infer_us", "compute_output_us"))
    out = {"faults_per_inf": faults_per_inf(tags, thr), "gpu_util_pct": gpu_util(tags)}
    if price and thr:
        # thông lượng tính theo mẫu (inferences/s đã nhân batch trong perf_analyzer)
        out["usd_per_1k"] = round(price * 1000 / (3600 * thr), 6)
    if elems and batch and srv:
        per_sample = srv / batch
        out["server_us_per_sample"] = round(per_sample, 1)
        row = fl.get(elems)
        if row:
            out["floor_one_copy_us"] = row["reuse_us"]
            out["copies_equivalent"] = round(per_sample / row["reuse_us"], 2)
    return out


def build():
    e = env()
    fl = floor()
    price = usd_per_hour()
    report = {
        "boundary": "triton/python_backend/src/pb_tensor.cc:725 (candidate B) + utils.py:49-92 (A) + BLS (C)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrument": {
            "gpu": e["gpu"], "driver": e["driver"], "vram": e.get("vram"),
            "container": os.environ.get("TRITON_TAG", "24.08-py3"),
            "measure_runs_configured": int(os.environ.get("MEASURE_RUNS", "10")),
        },
        "results": {},
    }
    # Mỗi cấu hình (id/idshm/bls/blsshm × batch × conc) là một điểm; gộp các run _rN để lấy noise.
    # Bỏ soak_* (đọc riêng từ soak.log).
    groups = sorted(set(re.sub(r"_r\d+$", "", t) for t in collect("")
                        if not t.startswith("soak_") and re.search(r"_r\d+$", t)))
    for g in groups:
        tags = [t for t in collect(g + "_r") if re.fullmatch(re.escape(g) + r"_r\d+", t)]
        a = agg(tags)
        if a["_runs"] == 0:
            continue
        report["results"][g] = {"agg": a, "self_check": boundary_ratio(a),
                                "physics": physics(g, a, fl, tags, price)}

    # cờ tự-kiểm cho toàn report: thước chưa đủ thì nói rõ thiếu gì, không phán
    ok_noise = all(v["agg"].get("p99_us", {}).get("n", 0) >= 10
                   for v in report["results"].values()) if report["results"] else False
    report["instrument"]["noise_floor_ok"] = ok_noise
    report["instrument"]["floor_measured"] = bool(fl)
    report["instrument"]["floor"] = list(fl.values())
    report["instrument"]["usd_per_hour"] = price
    sp = os.path.join(R, "_stub.txt")
    report["instrument"]["stub"] = open(sp).read().strip() if os.path.exists(sp) else "?"
    report["instrument"]["gpu_sampled"] = any(
        v["physics"].get("gpu_util_pct") is not None for v in report["results"].values())
    report["instrument"]["soak_done"] = os.path.exists(os.path.join(R, "soak.log"))
    have = set(report["results"])
    report["instrument"]["coverage"] = {
        "sweep": any(g.startswith("sw_") for g in have),
        "instances": any(g.startswith("inst_") for g in have),
        "asym": any(g.startswith("asym_") for g in have),
        "bls": any(g.startswith(("bls", "blsshm")) for g in have),
        "gpu_workload": any(g.startswith("gpu_") for g in have),
        "integrity": os.path.exists(os.path.join(R, "integrity.json")),
        "chaos": os.path.exists(os.path.join(R, "chaos.txt")),
    }
    ip = os.path.join(R, "integrity.json")
    if os.path.exists(ip):
        try:
            report["quality"] = json.load(open(ip))
        except ValueError:
            report["quality"] = {"error": "integrity.json hỏng"}
    report["instrument"]["note"] = (
        "status PROVEN chỉ đặt khi có bản before/after cùng harness và "
        "|after-before| > noise std. File này là 1 phía (baseline hoặc after)."
    )
    report["status"] = "NOT_PROVEN"  # luôn NOT_PROVEN cho tới khi có so sánh before/after
    blob = json.dumps(report, ensure_ascii=False, indent=2).encode()
    report_hash = hashlib.sha256(blob).hexdigest()[:16]
    out = os.path.join(R, f"{MODE}_{report['generated_utc'].replace(':','').replace('-','')}.json")
    report["artifact_hash"] = report_hash
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"-> {out}")
    print(f"   runs điểm: {len(report['results'])}, noise_floor_ok={ok_noise}, hash={report_hash}")
    for k, v in report["results"].items():
        ph = v.get("physics") or {}
        thr = v["agg"].get("throughput", {})
        print(f"   {k}: thr={thr.get('mean')}±{thr.get('std')}/s "
              f"srv/mẫu={ph.get('server_us_per_sample')}us "
              f"= {ph.get('copies_equivalent')} lần chép sàn, "
              f"fault/inf={ph.get('faults_per_inf')}, gpu={ph.get('gpu_util_pct')}%, "
              f"$/1k={ph.get('usd_per_1k')}")
    print(f"   thước: {json.dumps(report['instrument']['coverage'])} "
          f"floor={report['instrument']['floor_measured']} gpu={report['instrument']['gpu_sampled']} "
          f"soak={report['instrument']['soak_done']}")


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    build()
