#!/usr/bin/env python3
"""Bản đồ chi phí: so stub gốc với bản vá trên lưới cỡ ảnh x batch x thời gian model.

Dùng: python3 costmap.py <thư mục kết quả stub gốc> <thư mục kết quả bản vá> [USD_PER_HOUR]
Đọc thẳng CSV perf_analyzer (pa_cm_*_r*.csv), không qua measure.py. In bảng markdown:
  - thr        : ảnh/giây (trung bình các lần lặp), ± độ lệch chuẩn
  - tiết kiệm  : 1 - thr_gốc / thr_vá  = phần tiền máy bớt được cho cùng số ảnh
  - phí chuyển : (1/thr_gốc - 1/thr_vá) x batch, ms mỗi batch = thời gian bản vá bỏ được
  - model thật : thời gian GPU đo hiệu chuẩn của cm_spin_d<N> (gọi thẳng, không qua python_backend)
Ô nào chênh lệch nhỏ hơn 2 lần độ lệch chuẩn (gộp) bị đánh dấu "~" = trong nhiễu, không tính là có lợi.
"""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict

base_dir, patch_dir = sys.argv[1], sys.argv[2]
price = float(sys.argv[3]) if len(sys.argv) > 3 else 0.526


def load(d):
    """{(kind, work, elems, batch): [thr, ...]} và {depth: [ms, ...]} từ CSV perf_analyzer."""
    cells, cal = defaultdict(list), defaultdict(list)
    for f in glob.glob(os.path.join(d, "pa_cm_*_r*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        row = rows[-1]
        name = os.path.basename(f)
        m = re.match(r"pa_cm_(cpu|gpu)_w(\d+)_s(\d+)_b(\d+)_r\d+\.csv", name)
        if m:
            cells[(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))].append(
                float(row["Inferences/Second"]))
            continue
        m = re.match(r"pa_cm_cal_d(\d+)_r\d+\.csv", name)
        if m:  # thời gian GPU thật mỗi lần gọi = p50 latency phía server (us -> ms)
            cal[int(m.group(1))].append(float(row["p50 latency"]) / 1000.0)
    return cells, cal


def ms(x):
    return f"{st.mean(x):.2f}" if x else "?"


bc, bcal = load(base_dir)
pc, pcal = load(patch_dir)
print(f"Giá máy: ${price}/giờ. Stub gốc: {base_dir}. Bản vá: {patch_dir}\n")
print("Hiệu chuẩn GPU (cm_spin, gọi thẳng, p50 ms): " +
      ", ".join(f"d{d}={ms(bcal[d])}" for d in sorted(bcal)) + "\n")
print("| kiểu | model ms | cỡ ảnh (phần tử) | batch | thr gốc | thr vá | tiết kiệm tiền | phí chuyển ms/batch | n |")
print("|---|---|---|---|---|---|---|---|---|")
for key in sorted(set(bc) & set(pc)):
    kind, work, elems, batch = key
    a, b = bc[key], pc[key]
    ma, mb = st.mean(a), st.mean(b)
    sa = st.pstdev(a) if len(a) > 1 else 0.0
    sb = st.pstdev(b) if len(b) > 1 else 0.0
    noise = 2 * max(sa, sb)
    save = 1 - ma / mb
    fee = (1 / ma - 1 / mb) * batch * 1000
    flag = "~" if abs(mb - ma) <= noise else ""
    print(f"| {kind} | {work} | {elems} | {batch} | {ma:.1f}±{sa:.1f} | {mb:.1f}±{sb:.1f} | "
          f"{flag}{100 * save:+.1f}% | {fee:.2f} | {min(len(a), len(b))} |")
missing = sorted(set(bc) ^ set(pc))
if missing:
    print(f"\nÔ thiếu ở một bên (không so): {len(missing)}")
