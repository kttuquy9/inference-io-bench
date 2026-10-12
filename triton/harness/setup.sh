#!/usr/bin/env bash
# Bootstrap harness trên EC2 (g4dn, T4, driver>=535, Docker + nvidia runtime có sẵn ở DLAMI).
# Chạy 1 lần. Không cài gì tay ngoài container.
set -euo pipefail
cd "$(dirname "$0")"
source config.env

echo "== Kiểm tra môi trường (ghi lại đúng máy đo) =="
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee results/_env.txt
docker info --format '{{.Runtimes}}' | grep -q nvidia || { echo "THIẾU nvidia runtime"; exit 1; }

echo "== Kéo Triton server + SDK (perf_analyzer) =="
docker pull nvcr.io/nvidia/tritonserver:${TRITON_TAG}
docker pull nvcr.io/nvidia/tritonserver:${TRITON_SDK_TAG}

echo "== DCGM exporter (GPU idle, PCIe copy) =="
docker rm -f dcgm >/dev/null 2>&1 || true
# Không dùng --rm: lần trước DCGM sập thì container tự biến mất, file dcgm_*.txt rỗng mà không ai biết.
# SYS_ADMIN: cần để đọc metric profiling (PCIe bytes, SM active) trên T4.
docker run -d --gpus all --cap-add SYS_ADMIN --restart unless-stopped --name dcgm -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04 >/dev/null || true
# exporter cần vài chục giây mới có số đầu tiên -> thử lại tới 30s thay vì kết luận sau 8s
dcgm_ok=0
for i in $(seq 1 15); do
  # không dùng "curl | grep -q": grep thoát sớm -> curl bị SIGPIPE -> pipefail báo lỗi dù đã thấy metric
  m=$(curl -sf localhost:9400/metrics || true)
  case "$m" in *DCGM_FI_DEV_GPU_UTIL*) dcgm_ok=1; break ;; esac
  sleep 2
done
if [ "$dcgm_ok" = 1 ]; then
  echo "DCGM ok"
else
  echo "WARN: DCGM không trả metric. Log:"; docker logs --tail 20 dcgm 2>&1 || true
  echo "      run.sh vẫn lấy mẫu GPU bằng nvidia-smi; DCGM chỉ bổ sung PCIe."
fi

echo "== Sinh model ONNX có compute GPU (workload thật) =="
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/h" nvcr.io/nvidia/tritonserver:${TRITON_TAG} \
  sh -c "pip install -q --user onnx==1.16.2 && python3 /h/gen_onnx.py /h/models/gpu_conv/1/model.onnx"

echo "== Xong. Chạy: bash start.sh full =="
