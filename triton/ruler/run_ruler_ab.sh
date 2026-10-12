#!/usr/bin/env bash
# Thước ở chế độ gác cổng trên 3 bản, xen kẽ 2 vòng: gốc image / vá cờ tắt / vá cờ bật.
# Kỳ vọng: gốc ĐỎ (rc=1), cờ tắt ĐỎ (= gốc), cờ bật XANH (rc=0). Lượt đã xong (out_*/_run.txt có rc=) bỏ qua.
HERE="$(cd "$(dirname "$0")" && pwd)"
export BOUNDARY_MAX_FAULTS_PER_SAMPLE=${BOUNDARY_MAX_FAULTS_PER_SAMPLE:-5}
D=$HERE/../patch/out/r26.09/${FIX:-stub_F}
for i in 1 2; do
  for v in base off on; do
    n=gate${OUTTAG-F}_${v}_$i
    grep -q "^rc=" "$HERE/out_$n/_run.txt" 2>/dev/null && { echo "bỏ qua $n"; continue; }
    case $v in
      base) bash "$HERE/run_ruler.sh" $n ;;
      off)  bash "$HERE/run_ruler.sh" $n $D ;;
      on)   bash "$HERE/run_ruler.sh" $n $D TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1 ;;
    esac > "$HERE/out_$n.log" 2>&1
    echo "$n rc=$?"
  done
done
echo "=== xong thước ==="
