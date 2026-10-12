#!/usr/bin/env bash
# Kiểm một chênh lệch nhỏ có thật không: đo XEN KẼ stub gốc / bản vá nhiều vòng trên cùng vài ô,
# để độ lệch giữa các phiên (máy khởi động lại, nhiệt độ...) rơi đều cho cả hai bên.
# Dùng: bash run_interleave.sh <thư mục harness> <số vòng>
# Ô đo lấy từ biến môi trường COSTMAP_* (mặc định: GPU 4.3 ms, 224² và 1080p, batch 1, 10 lần/vòng).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
H="${1:?thiếu đường dẫn harness}"; ROUNDS="${2:-2}"
cd "$H"
export COSTMAP_ELEMS="${COSTMAP_ELEMS:-150528 1555200}" COSTMAP_BATCH="${COSTMAP_BATCH:-1}"
export COSTMAP_WORK="${COSTMAP_WORK:-2}" COSTMAP_SPIN="${COSTMAP_SPIN:-2:1}" COSTMAP_RUNS="${COSTMAP_RUNS:-10}"
for i in $(seq 1 "$ROUNDS"); do
  for side in base C; do
    out="results_il_${side}_${i}"
    [ -f "$out/_done" ] && { echo "bỏ qua $out"; continue; }
    # máy tắt giữa vòng: results/ của đúng vòng này -> dùng tiếp (run.sh bỏ qua lần đo đã có CSV);
    # results/ của vòng khác -> cất đi, không trộn số
    if [ -d results ] && [ "$(cat results/_round.txt 2>/dev/null)" != "$out" ]; then
      mv results "results_before_il_$(date +%s)"
    fi
    mkdir -p results
    echo "$out" > results/_round.txt
    stub=""; [ "$side" = C ] && stub="$HERE/out/stub_C"
    STUB="$stub" bash run.sh costmap
    mv results "$out"; date -u +%FT%TZ > "$out/_done"
  done
done
echo "=== xong xen kẽ ==="
