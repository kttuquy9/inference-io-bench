#!/usr/bin/env bash
# Bản đồ chi phí: cùng lưới (cỡ ảnh x batch x thời gian model) đo trên stub gốc của image và trên bản vá C.
# Máy tắt giữa chừng: chạy lại đúng lệnh này là làm tiếp (lượt xong có _round_done; lượt dở dùng tiếp results/).
# Dùng (trên máy đo): bash run_costmap.sh <thư mục harness>
# Kết quả: <harness>/results_cm_base, <harness>/results_cm_C ; so bằng: python3 costmap.py results_cm_base results_cm_C
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
H="${1:?thiếu đường dẫn harness}"
cd "$H"

run_round() {  # tên lượt, file stub ("" = stub gốc của image)
  local name="$1" stub="$2"
  if [ -f "results_$name/_round_done" ]; then echo "=== lượt $name đã xong -> bỏ qua ==="; return; fi
  echo "=== lượt $name: ${stub:-stub gốc của image} ==="
  if [ -d results ] && [ "$(cat results/_round.txt 2>/dev/null)" != "$name" ]; then
    mv results "results_before_${name}_$(date +%s)"
  fi
  mkdir -p results
  echo "$name" > results/_round.txt
  STUB="$stub" bash run.sh costmap
  local errs=0
  [ -f results/_bench_errors.txt ] && errs=$(wc -l < results/_bench_errors.txt)
  [ -d "results_$name" ] && mv "results_$name" "results_${name}_old_$(date +%s)"
  mv results "results_$name"
  if [ "$errs" -gt 0 ]; then echo "!! lượt $name có $errs lần đo lỗi -> DỪNG"; exit 1; fi
  date -u +%FT%TZ > "results_$name/_round_done"
}

run_round cm_base ""
run_round cm_C "$HERE/out/stub_C"
python3 costmap.py results_cm_base results_cm_C | tee results_cm_C/costmap.md
echo "=== xong bản đồ chi phí ==="
