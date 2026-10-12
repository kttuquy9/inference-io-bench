#!/usr/bin/env bash
# Đo bản vá trên Triton mới (tritonserver:26.09-py3, python_backend r26.09 = main ở src/).
# Bộ bắn tải giữ SDK 24.08: cùng cái cân với mọi số đo trước.
#   def   = stub có sẵn trong image 26.09 (gốc NVIDIA)
#   A     = r26.09 tự build, chưa vá        -> phải ≈ def (thước tự chứng minh cách build)
#   Doff  = r26.09 + bản vá sau cờ, cờ TẮT  -> phải ≈ def (không đổi hành vi khi không bật)
#   Don   = như trên, cờ BẬT                -> zero-copy
# Xen kẽ 2 vòng để lệch giữa các phiên rơi đều. Máy tắt giữa chừng: chạy lại đúng lệnh là làm tiếp
# (vòng đã xong có _done; vòng dở dùng tiếp results/ nhờ _round.txt, run.sh bỏ qua lần đo đã có CSV).
# Dùng: bash run_2609.sh <thư mục harness> [số vòng] [danh sách bài]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
H="${1:?thiếu đường dẫn harness}"; ROUNDS="${2:-2}"; STAGES="${3:-integrity sweep hold}"
cd "$H"
export TRITON_TAG=26.09-py3 SRV_ARGS="--allow-client-shm=true"
S=$HERE/out/r26.09
for i in $(seq 1 "$ROUNDS"); do
  for v in def A Doff Don; do
    out="results_2609${OUTTAG-F}_${v}_${i}"   # OUTTAG=F: lượt đo bản sửa F (kết quả bản D cũ giữ nguyên)
    [ -f "$out/_done" ] && { echo "bỏ qua $out"; continue; }
    if [ -d results ] && [ "$(cat results/_round.txt 2>/dev/null)" != "$out" ]; then
      mv results "results_before_2609_$(date +%s)"
    fi
    mkdir -p results
    echo "$out" > results/_round.txt
    [ -f "$HERE/floor.json" ] && cp "$HERE/floor.json" results/floor.json
    case $v in
      def)  stub=""; env="" ;;
      A)    stub="$S/stub_A"; env="" ;;
      Doff) stub="$S/${FIX:-stub_F}"; env="" ;;
      Don)  stub="$S/${FIX:-stub_F}"; env="TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1" ;;
    esac
    echo "=== $out stub=${stub:-image} env=${env:-none} ==="
    for st in $STAGES; do STUB="$stub" SRV_ENV="$env" bash run.sh "$st"; done
    # fail-closed: vòng có lần đo lỗi -> không đánh dấu xong, dừng hẳn (lần 4/10 quên chặn: 8 vòng "xong" mà 0 CSV)
    if [ -s results/_bench_errors.txt ]; then
      echo "!! $out có $(wc -l < results/_bench_errors.txt) lần đo lỗi -> DỪNG (xem results/_bench_errors.txt)"; exit 1
    fi
    mv results "$out"; date -u +%FT%TZ > "$out/_done"
  done
done
echo "=== xong 26.09 ==="
