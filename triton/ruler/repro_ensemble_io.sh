#!/usr/bin/env bash
# Dựng lại riêng IOTest.test_ensemble_io (trial decoupled) của NVIDIA, lặp N lần trong 1 server, đếm số lần sai.
# Lỗi 5/10: chỉ hỏng khi bật cờ zero-copy, nhưng không phải lần nào cũng hỏng (6/10 chạy lại với stub_E thì qua)
# -> phải đếm tỉ lệ, không kết luận từ 1 lần.
# Dùng: bash repro_ensemble_io.sh <tên> [stub] [env server] [N=10] [trial=decoupled]
# Kết quả: repro_<tên>/{result.txt, client.log, server.log}
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?thiếu tên}"; STUB="${2:-}"; SENV="${3:-}"; N="${4:-10}"; TRIAL="${5:-decoupled}"
IMG=nvcr.io/nvidia/tritonserver:26.09-py3
Q="$HERE/server_r2609"; OUT="$HERE/repro_$NAME"; sudo rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$Q/qa" "$OUT/qa"; cp "$HOME/patch/out/r26.09/"triton_shm_monitor*.so "$OUT/qa/common/"
src=dlpack_io_identity; [ "$TRIAL" = decoupled ] && src=dlpack_io_identity_decoupled
cat > "$OUT/qa/L0_backend_python/io/_repro.sh" <<EOF
set -u
pip3 install -q "tritonclient[all]" pytest ml_dtypes >/dev/null 2>&1
pip3 install -q torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html >/dev/null 2>&1
rm -rf models
for i in 1 2 3; do
  m=dlpack_io_identity_\$i; mkdir -p models/\$m/1
  cp ../../python_models/$src/model.py models/\$m/1/; cp ../../python_models/$src/config.pbtxt models/\$m/
  sed -i "s/^name:.*/name: \"\$m\"/" models/\$m/config.pbtxt
done
mkdir -p models/ensemble_io/1; cp ../../python_models/ensemble_io/config.pbtxt models/ensemble_io/
tritonserver --model-repository=\$PWD/models --log-verbose=1 > /o/server.log 2>&1 &
for i in \$(seq 1 120); do curl -sf localhost:8000/v2/health/ready >/dev/null && break; sleep 1; done
fail=0
for r in \$(seq 1 $N); do
  if TRIAL=$TRIAL python3 -m pytest -q io_test.py::IOTest::test_ensemble_io >> /o/client.log 2>&1; then
    echo "lần \$r: qua"; else echo "lần \$r: SAI"; fail=\$((fail+1)); fi
done
echo "sai=\$fail/$N"
kill %1; wait
EOF
stub_args=(); [ -n "$STUB" ] && stub_args=(-v "$(realpath "$STUB"):/opt/tritonserver/backends/python/triton_python_backend_stub:ro")
env_args=(); for kv in $SENV; do env_args+=(-e "$kv"); done
echo "stub=${STUB:-image} sha=$( [ -n "$STUB" ] && sha256sum "$STUB" | cut -c1-16) env=${SENV:-none} trial=$TRIAL N=$N" > "$OUT/result.txt"
docker run --rm --gpus all --shm-size=1g "${env_args[@]}" "${stub_args[@]}" -v "$OUT/qa:/opt/tritonserver/qa" -v "$OUT:/o" \
  -w /opt/tritonserver/qa/L0_backend_python/io $IMG bash _repro.sh >> "$OUT/result.txt" 2>&1
sudo rm -rf "$OUT/qa"
cat "$OUT/result.txt"
