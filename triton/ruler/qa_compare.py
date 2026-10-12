#!/usr/bin/env python3
"""So kết quả gom /qa giữa bản gốc và một bản vá: python3 qa_compare.py qa_base qa_on [qa_off ...]
In bảng markdown từng đơn vị: rc / số bài qua / số bài trượt mỗi bản, và đánh dấu:
  HỒI QUY   = gốc qua (rc 0) mà bản vá trượt  -> lỗi do bản vá, phải sửa
  CẢ HAI    = gốc cũng trượt -> do môi trường máy đo, đơn vị này CHƯA kiểm được bản vá (ghi rõ, không tính là qua)
"""
import csv, sys, os

def load(d):
    rows = {}
    with open(os.path.join(d, "summary.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["unit"]] = r
    return rows

dirs = sys.argv[1:]
data = [load(d) for d in dirs]
base = data[0]
units = list(base) + [u for d in data[1:] for u in d if u not in base]
for d in dirs:
    print(f"- `{d}`: " + open(os.path.join(d, "_run.txt")).read().strip().replace("\n", " | "))
print()
print("| đơn vị | " + " | ".join(os.path.basename(d) for d in dirs) + " | kết luận |")
print("|---|" + "---|" * len(dirs) + "---|")
count = {"OK": 0, "HỒI QUY": 0, "CẢ HAI": 0, "THIẾU": 0}
for u in units:
    cells, verdict = [], "OK"
    for i, d in enumerate(data):
        r = d.get(u)
        cells.append("-" if r is None else f"rc={r['rc']} ({r['passed']}✓/{r['failed']}✗)")
    b = base.get(u)
    if b is None or any(d.get(u) is None for d in data):
        verdict = "THIẾU"
    elif b["rc"] != "0":
        verdict = "CẢ HAI" if any(d[u]["rc"] != "0" for d in data[1:]) else "gốc trượt, vá qua"
    elif any(d[u]["rc"] != "0" for d in data[1:]):
        verdict = "HỒI QUY"
    count[verdict] = count.get(verdict, 0) + 1
    print(f"| {u} | " + " | ".join(cells) + f" | {verdict} |")
print("\nTổng: " + ", ".join(f"{k} {v}" for k, v in count.items() if v))
