#!/usr/bin/env bash
# Chạy thước L0_python_backend_boundary_perf trên máy đo, giống cách CI của họ chạy:
# server tritonserver (image 26.09) + client cùng máy, chung không gian tiến trình để đọc page fault của stub.
# Dùng: bash run_ruler.sh <tên lượt> [stub] [env server]
#   bash run_ruler.sh base                                   -> stub gốc của image
#   bash run_ruler.sh fix ~/patch/out/r26.09/stub_D TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1
# Biến BOUNDARY_* truyền thẳng vào test (vd BOUNDARY_MAX_BATCH_COST_RATIO=1.5 để bật chế độ gác cổng).
# Kết quả: out_<tên lượt>/{boundary_perf.json, client.log, server.log}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?thiếu tên lượt}"; STUB="${2:-}"; SENV="${3:-}"
IMG=nvcr.io/nvidia/tritonserver:${TRITON_TAG:-26.09-py3}
SDK=nvcr.io/nvidia/tritonserver:${TRITON_SDK_TAG:-24.08-py3-sdk}
OUT="$HERE/out_$NAME"; mkdir -p "$OUT"
QA="$HERE/server/qa"
M="$OUT/models"; rm -rf "$M"; mkdir -p "$M/input_only_fp32/1" "$M/identity_fp32/1"
cp "$QA/python_models/input_only_fp32/model.py" "$M/input_only_fp32/1/"
cp "$QA/python_models/input_only_fp32/config.pbtxt" "$M/input_only_fp32/"
# identity_fp32 lấy nguyên bản của họ (repos/triton/server/qa/python_models) đã chép sẵn vào đây
cp "$HERE/identity_fp32/model.py" "$M/identity_fp32/1/"
cp "$HERE/identity_fp32/config.pbtxt" "$M/identity_fp32/"

docker rm -f ruler_srv >/dev/null 2>&1 || true
stub_args=(); [ -n "$STUB" ] && stub_args=(-v "$(realpath "$STUB"):/opt/tritonserver/backends/python/triton_python_backend_stub:ro")
env_args=(); for kv in $SENV; do env_args+=(-e "$kv"); done
docker run -d --gpus all --name ruler_srv --ipc=host --network host -e PYTHONDONTWRITEBYTECODE=1 \
  "${env_args[@]}" "${stub_args[@]}" -v "$M:/models" \
  $IMG tritonserver --model-repository=/models --allow-client-shm=true >/dev/null
for i in $(seq 1 90); do curl -sf localhost:8000/v2/health/ready >/dev/null && break; sleep 2; done
curl -sf localhost:8000/v2/health/ready >/dev/null || { docker logs ruler_srv | tail -20; exit 1; }
echo "stub=${STUB:-image} sha=$( [ -n "$STUB" ] && sha256sum "$STUB" | cut -c1-16) env=${SENV:-none} image=$IMG" > "$OUT/_run.txt"

# client chung pid namespace với server (đọc /proc của stub), chung ipc (shared memory), chạy root như CI
benv=(); while IFS='=' read -r k v; do [[ $k == BOUNDARY_* ]] && benv+=(-e "$k=$v"); done < <(env)
set +e
docker run --rm --pid=container:ruler_srv --ipc=host --network host "${benv[@]}" \
  -e BOUNDARY_RESULT=/o/boundary_perf.json -v "$OUT:/o" -v "$QA/L0_python_backend_boundary_perf:/t:ro" \
  $SDK python3 /t/boundary_perf_test.py > "$OUT/client.log" 2>&1
rc=$?
set -e
docker logs ruler_srv > "$OUT/server.log" 2>&1
docker stop -t 30 ruler_srv >/dev/null; docker rm ruler_srv >/dev/null
echo "rc=$rc" >> "$OUT/_run.txt"
cat "$OUT/client.log"
exit $rc
