#!/usr/bin/env bash
# Đo trước/sau bản vá bằng ĐÚNG harness đã đo stub gốc (cùng cái cân).
#   Lượt A: stub_A (r24.08 tự build, chưa vá) -> sweep. Phải khớp số stub gốc của image (lượt sạch 1/10),
#           nếu lệch thì cách build có vấn đề -> dừng, không đo B.
#   Lượt B: stub_B (đã vá dòng 660) -> integrity, sweep, inst, gpu, bls, chaos.
# Soak cho B làm riêng sau (60 phút).
# Máy tắt giữa chừng: chạy lại đúng lệnh này là làm tiếp. Lượt đã xong (results_<lượt>/_round_done) bỏ qua;
# lượt đang dở dùng tiếp results/ (_round.txt ghi tên lượt) để run.sh bỏ qua các lần đo đã có CSV.
# Dùng (trên máy đo): bash run_ab.sh <thư mục harness>
# Kết quả: <harness>/results_A, <harness>/results_B
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
H="${1:?thiếu đường dẫn harness}"
cd "$H"

run_round() {  # tên lượt, file stub, danh sách bài
  local name="$1" stub="$2"; shift 2
  if [ -f "results_$name/_round_done" ]; then echo "=== lượt $name đã xong -> bỏ qua ==="; return; fi
  echo "=== lượt $name: $stub ($(sha256sum "$stub" | cut -c1-16)) bài: $* ==="
  # harness luôn ghi vào results/. results/ của đúng lượt này (đang dở) -> dùng tiếp; của lượt khác -> cất đi
  if [ -d results ] && [ "$(cat results/_round.txt 2>/dev/null)" != "$name" ]; then
    mv results "results_before_${name}_$(date +%s)"
  fi
  mkdir -p results
  echo "$name" > results/_round.txt
  # sàn vật lý đo trên cùng máy -> dùng lại, không đo lại
  [ -f "$HERE/floor.json" ] && cp "$HERE/floor.json" results/floor.json
  for st in "$@"; do STUB="$stub" bash run.sh "$st"; done
  python3 measure.py "ab_$name" | tail -40
  local errs=0
  [ -f results/_bench_errors.txt ] && errs=$(wc -l < results/_bench_errors.txt)
  [ -d "results_$name" ] && mv "results_$name" "results_${name}_old_$(date +%s)"   # không bao giờ xoá kết quả cũ
  mv results "results_$name"
  # fail-closed: lượt có lần đo lỗi thì số không dùng được -> dừng hẳn, không chạy lượt sau
  if [ "$errs" -gt 0 ]; then echo "!! lượt $name có $errs lần đo lỗi -> DỪNG (xem results_$name/_bench_errors.txt)"; exit 1; fi
  date -u +%FT%TZ > "results_$name/_round_done"
}

run_round A "$HERE/out/stub_A" sweep
run_round B "$HERE/out/stub_B" integrity sweep inst gpu bls chaos
echo "=== xong A/B ==="
