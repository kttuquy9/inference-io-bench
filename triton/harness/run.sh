#!/usr/bin/env bash
# Ép tải + đo phần mềm, phần cứng, chi phí. Mỗi cấu hình chạy MEASURE_RUNS lần để lấy noise floor.
# Không đọc code khi chạy — đọc bảng chia khúc của perf_analyzer, mẫu GPU, page fault của stub.
#
# Dùng: bash run.sh <mode> [shm|http]
#   floor    phần cứng: sàn memcpy của máy (reuse vs fresh) -> results/floor.json
#   identity phần mềm: identity_fp32, batch x concurrency (baseline cũ, giữ để so)
#   sweep    phần mềm: cỡ tensor x batch x {chỉ chiều vào, vào+ra}
#   inst     hạ tầng: 1/2/4 instance, frame 1080p
#   asym     phần mềm: tải bất đối xứng, request nhỏ và lớn chạy cùng lúc
#   bls      phần mềm: BLS chain 3 model
#   soak     hạ tầng: tải dài SOAK_MINUTES, theo dõi RSS/VRAM
#   full     floor + sweep + inst + asym + bls + soak
#   shm (mặc định): tensor đi qua system shared memory -> loại HTTP khỏi phép đo.
#   http          : gửi tensor qua HTTP — dùng để đối chiếu.
set -euo pipefail
cd "$(dirname "$0")"
source config.env
MODE="${1:-full}"
TRANSPORT="${2:-shm}"
case "$TRANSPORT" in shm|http) ;; *) echo "transport phải là shm hoặc http"; exit 1 ;; esac
NET=host
SRV=triton_srv
IMG=nvcr.io/nvidia/tritonserver:${TRITON_TAG}
SDK=nvcr.io/nvidia/tritonserver:${TRITON_SDK_TAG}
# ghi lại máy đo mỗi lần chạy (calibration) — measure.py đọc các file này
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > results/_env.txt
lscpu > results/_cpu.txt 2>/dev/null || true
echo "USD_PER_HOUR=${USD_PER_HOUR}" > results/_cost.txt
# stub nào đang được đo: hash để kết quả truy được về đúng bản (gốc / A / B)
if [ -n "${STUB:-}" ]; then echo "stub=$STUB sha256=$(sha256sum "$STUB" | cut -c1-16)" > results/_stub.txt
else echo "stub=image-default" > results/_stub.txt; fi
echo "srv_env=${SRV_ENV:-}" >> results/_stub.txt
echo "image=$IMG sdk=$SDK" >> results/_stub.txt
P_ID=$([ "$TRANSPORT" = shm ] && echo idshm || echo id)
P_BLS=$([ "$TRANSPORT" = shm ] && echo blsshm || echo bls)

# identity_in_i2 / _i4 sinh từ identity_in, chỉ đổi số instance
gen_instances() {
  for i in $INSTANCES; do
    [ "$i" = 1 ] && continue
    local d="models/identity_in_i$i"
    rm -rf "$d"; cp -r models/identity_in "$d"
    sed -i "s/count: 1 /count: $i /" "$d/config.pbtxt"
  done
}

# tắt êm (docker stop) để Triton tự gỡ vùng shm của stub; rm -f thẳng thì để lại rác trong /dev/shm
stop_server() {
  docker stop -t 30 $SRV >/dev/null 2>&1 || true
  docker rm -f $SRV >/dev/null 2>&1 || true
}

# /dev/shm phải sạch trước khi đo. Lần 1/10: vùng /OUT, /IN_0_0 thuộc root (perf_analyzer bị giết ngang)
# làm mọi lần đo sau lỗi "unable to get shared memory descriptor"; ~1.2 GB vùng stub mồ côi giữ RAM.
clean_shm() {
  pgrep -x tritonserver >/dev/null && { echo "!! đang có tritonserver chạy, không dọn /dev/shm"; return; }
  local junk; junk=$(ls /dev/shm 2>/dev/null | grep -E '^(triton_python_backend_shm_region_|IN|OUT|FRAME|X)' || true)
  [ -z "$junk" ] && return
  echo "dọn /dev/shm mồ côi: $(echo "$junk" | wc -l) vùng"
  (cd /dev/shm && sudo rm -f $junk)
}

# dt_check_<kiểu>: cùng model kiểm, chỉ đổi kiểu dữ liệu của IN/OUT
gen_dtype_models() {
  for t in $DTYPES; do
    local d="models/dt_check_$(echo $t | tr A-Z a-z)"
    rm -rf "$d"; cp -r models/dt_check "$d"
    sed -i "/name: \"IN\"/s/TYPE_FP32/TYPE_$t/; /name: \"OUT\"/s/TYPE_FP32/TYPE_$t/" "$d/config.pbtxt"
  done
}

start_server() {
  stop_server
  clean_shm
  # --ipc=host: server và perf_analyzer dùng chung /dev/shm của máy -> shared memory hoạt động
  # PYTHONDONTWRITEBYTECODE: container chạy root, không cho sinh __pycache__ (thuộc root) vào models/
  # STUB=<đường dẫn>: thay stub python_backend trong container bằng stub tự build (đo trước/sau bản vá).
  # Không đặt STUB = dùng stub gốc của image.
  local stub_mount=()
  if [ -n "${STUB:-}" ]; then
    [ -f "$STUB" ] || { echo "không thấy STUB=$STUB"; exit 1; }
    stub_mount=(-v "$(realpath "$STUB"):/opt/tritonserver/backends/python/triton_python_backend_stub:ro")
  fi
  # SRV_ENV="KEY=VAL KEY2=VAL2": biến môi trường thêm cho server (stub thừa hưởng), vd thử bộ cấp phát khác
  local env_args=()
  for kv in ${SRV_ENV:-}; do env_args+=(-e "$kv"); done
  docker run -d --gpus all --name $SRV --network $NET --ipc=host \
    -e PYTHONDONTWRITEBYTECODE=1 "${env_args[@]}" \
    -v "$PWD/models:/models" "${stub_mount[@]}" \
    $IMG tritonserver --model-repository=/models --log-verbose=0 ${SRV_ARGS:-} >/dev/null
  echo -n "chờ server READY"
  for i in $(seq 1 90); do
    if curl -sf localhost:8000/v2/health/ready >/dev/null; then echo " ok"; return; fi
    sleep 2; echo -n "."
  done
  echo " FAIL"; docker logs --tail 40 $SRV; exit 1
}

# perf_analyzer chạy trong MỘT container SDK sống suốt lượt đo, mỗi lần đo chỉ docker exec.
# Trước đây mỗi lần đo bật container mới (~2-3s x ~270 lần = 10-13 phút phí). Lệnh đo giữ nguyên.
PA=pa_sdk
PA_FRESH=0
ensure_sdk() {
  # lần gọi đầu trong mỗi run.sh: luôn tạo mới (container sống từ lượt trước còn mount thư mục results cũ)
  if [ "$PA_FRESH" = 0 ]; then docker rm -f $PA >/dev/null 2>&1 || true; PA_FRESH=1; fi
  [ -n "$(docker ps -q -f name=^${PA}$)" ] && return
  docker rm -f $PA >/dev/null 2>&1 || true
  docker run -d --name $PA --network $NET --ipc=host --user "$(id -u):$(id -g)" \
    -v "$PWD/results:/r" $SDK sleep infinity >/dev/null
}

# server còn sống không; chết -> lưu log sập vào results/crash_*.log rồi dựng lại để bài sau vẫn chạy
ensure_server() {
  local stage="$1"
  if curl -sf localhost:8000/v2/health/ready >/dev/null; then return; fi
  local f="results/crash_$(date +%H%M%S)_before_${stage}.log"
  echo "!! SERVER KHÔNG SỐNG trước bài ${stage} -> lưu $f, dựng lại"
  docker ps -a --filter name=$SRV --format "{{.Status}}" > "$f" 2>&1 || true
  docker logs --tail 80 $SRV >> "$f" 2>&1 || true
  start_server
}

# tổng minor page fault của mọi tiến trình stub (phần cứng: cấp bộ nhớ mới mỗi request -> fault tăng)
stub_faults() {
  docker exec $SRV sh -c 'for d in /proc/[0-9]*; do
      grep -q triton_python_backend_stub $d/cmdline 2>/dev/null && cut -d" " -f10 $d/stat; done' \
    2>/dev/null | awk '{s+=$1} END{print s+0}'
}

# perf_analyzer cho 1 model, 1 batch, 1 concurrency; CSV kết quả vào results/
# out_bytes: kích thước vùng shm cho output (chỉ dùng khi TRANSPORT=shm)
bench() {
  local model="$1" shape="$2" batch="$3" conc="$4" tag="$5" out_bytes="${6:-67108864}"
  BENCH_N=$(( ${BENCH_N:-0} + 1 ))
  # đã có CSV hợp lệ (header + ít nhất 1 dòng số) -> bỏ qua. Máy tắt giữa chừng thì chạy lại
  # chỉ đo phần còn thiếu. FORCE=1 để đo lại tất cả.
  if [ "${FORCE:-0}" != 1 ] && [ -s "results/pa_${tag}.csv" ] && [ "$(wc -l < "results/pa_${tag}.csv")" -ge 2 ]; then
    echo "[bỏ qua] #${BENCH_N} ${tag} đã có kết quả"; return
  fi
  echo "[$(date +%H:%M:%S)] #${BENCH_N} ${model} ${shape} batch=${batch} conc=${conc} ${TRANSPORT} -> ${tag}"
  : > "results/pa_${tag}.log"   # lần đo dở trước (không có CSV) không được trộn log vào lần này
  ensure_sdk
  local shm_flags=()
  if [ "$TRANSPORT" = shm ]; then
    shm_flags=(--shared-memory system --output-shared-memory-size "$out_bytes")
  fi
  # batch lớn: ít request hơn mỗi cửa sổ đo, tránh một lần chạy kéo dài hàng chục phút
  local reqs=${BENCH_REQS:-$([ "$batch" -gt 1 ] && echo 200 || echo 500)}
  # phần cứng: lấy mẫu GPU suốt lần đo + page fault của stub trước/sau
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits -lms "$GPU_SAMPLE_MS" > "results/gpu_${tag}.csv" &
  local smi=$!
  echo "$(date +%s.%N) $(stub_faults)" > "results/flt_${tag}.txt"
  docker exec $PA perf_analyzer -m "$model" -b "$batch" --concurrency-range "$conc" \
      --shape "$shape" --measurement-mode count_windows --measurement-request-count "$reqs" \
      --input-data random --stability-percentage 15 "${shm_flags[@]}" \
      -f "/r/pa_${tag}.csv" >> "results/pa_${tag}.log" 2>&1 || true
  echo "$(date +%s.%N) $(stub_faults)" >> "results/flt_${tag}.txt"
  kill $smi 2>/dev/null || true
  if [ ! -s "results/pa_${tag}.csv" ]; then
    echo "        !! LỖI: lần đo không ra CSV -> xem results/pa_${tag}.log"
    echo "$tag" >> results/_bench_errors.txt
  fi
  # in tóm tắt ngay: throughput + latency + lỗi nếu có
  grep -E "Throughput|p99 latency|error|Error|failed" "results/pa_${tag}.log" | tail -3 | sed 's/^/        /'
}

snapshot_metrics() {  # Triton /metrics + DCGM trước/sau để đối chiếu
  local tag="$1"
  curl -s localhost:8002/metrics > "results/tmetrics_${tag}.txt" 2>/dev/null || true
  curl -s localhost:9400/metrics > "results/dcgm_${tag}.txt"    2>/dev/null || true
}

out_size() {  # byte vùng output: model chỉ-chiều-vào trả 1 số/mẫu, còn lại trả cả tensor
  local model="$1" elems="$2" b="$3"
  case "$model" in identity_in*) echo $(( 4 * b + 1048576 )) ;;
                   *)            echo $(( elems * 4 * b + 1048576 )) ;; esac
}

run_floor() {
  echo "== sàn memcpy =="
  grep -qs '"sizes"' results/floor.json && { echo "[bỏ qua] floor.json đã có"; return; }
  docker run --rm --ipc=host --user "$(id -u):$(id -g)" -v "$PWD:/h" -e OUT=/h/results/floor.json \
    -e FLOOR_ELEMS="$SWEEP_ELEMS $(( TENSOR_ELEMS_LARGE * 16 ))" \
    $IMG python3 /h/floor.py > results/floor.log 2>&1
  cat results/floor.json
}

run_identity() {
  echo "== identity (cô lập ranh giới B), transport=$TRANSPORT =="
  for b in $BATCH_SIZES; do
    for c in $CONCURRENCY; do
      for run in $(seq 1 $MEASURE_RUNS); do
        local tag="${P_ID}_b${b}_c${c}_r${run}"
        snapshot_metrics "${tag}_pre"
        bench identity_fp32 "IN:${TENSOR_ELEMS_LARGE}" "$b" "$c" "$tag" \
          "$(out_size identity_fp32 $TENSOR_ELEMS_LARGE $b)"
        snapshot_metrics "${tag}_post"
      done
    done
  done
}

run_sweep() {
  echo "== quét cỡ tensor x batch x chiều =="
  for m in $SWEEP_MODELS; do
    for e in $SWEEP_ELEMS; do
      for b in $SWEEP_BATCH; do
        for run in $(seq 1 $MEASURE_RUNS); do
          bench "$m" "IN:${e}" "$b" 1 "sw_${m}_s${e}_b${b}_r${run}" "$(out_size $m $e $b)"
        done
      done
    done
  done
}

run_inst() {
  echo "== nhiều instance, frame 1080p =="
  for i in $INSTANCES; do
    local m=$([ "$i" = 1 ] && echo identity_in || echo "identity_in_i$i")
    local c=$(( i * 2 ))   # đủ request để mọi instance cùng bận
    for run in $(seq 1 $MEASURE_RUNS); do
      bench "$m" "IN:${TENSOR_ELEMS_LARGE}" 1 "$c" "inst_i${i}_c${c}_r${run}" "$(out_size $m 0 1)"
    done
  done
}

run_asym() {
  echo "== tải bất đối xứng: nhỏ + lớn cùng lúc, 2 instance =="
  local small=$(echo $SWEEP_ELEMS | awk '{print $1}')
  for run in $(seq 1 $MEASURE_RUNS); do
    # perf_analyzer đặt tên vùng shm cố định (IN_0_0): hai tiến trình dùng shm cùng lúc sẽ gỡ vùng
    # của nhau (lần khói 27/9 làm server segfault). Luồng nhỏ đi HTTP (48 KB), luồng lớn giữ shm.
    TRANSPORT=http bench identity_in_i2 "IN:${small}" 1 2 "asym_small_r${run}" "$(out_size identity_in 0 1)" &
    local p1=$!
    bench identity_in_i2 "IN:${TENSOR_ELEMS_LARGE}" 1 2 "asym_large_r${run}" "$(out_size identity_in 0 1)"
    wait $p1 || true
  done
}

run_bls() {
  echo "== BLS chain 3 model, transport=$TRANSPORT =="
  for c in $CONCURRENCY; do
    for run in $(seq 1 $MEASURE_RUNS); do
      local tag="${P_BLS}_c${c}_r${run}"
      snapshot_metrics "${tag}_pre"
      bench bls_pipeline "FRAME:${TENSOR_ELEMS_LARGE}" 1 "$c" "$tag" 1048576
      snapshot_metrics "${tag}_post"
    done
  done
}

run_integrity() {
  echo "== toàn vẹn dữ liệu + rủi ro =="
  grep -qs '"all_echo_exact"' results/integrity.json && { echo "[bỏ qua] integrity.json đã có"; return; }
  docker run --rm --network $NET --ipc=host --user "$(id -u):$(id -g)" -v "$PWD:/h" \
    -e OUT=/h/results/integrity.json -e INTEGRITY_ELEMS="$(echo $SWEEP_ELEMS | awk '{print $1, $NF}')" \
    -e DTYPES="$DTYPES" \
    $SDK python3 /h/integrity.py > results/integrity.err 2>&1 || true
  python3 -c "import json;d=json.load(open('results/integrity.json'));print('  echo khớp tuyệt đối:',d['all_echo_exact']);[print('  ',t['model'],t['elems'],'b',t['batch'],'sai',t['wrong'],'lỗi',t['errors'],'OUT',t['out_values']) for t in d['tests']]" \
    || { echo "  integrity lỗi:"; tail -5 results/integrity.err; }
}

run_gpu() {
  echo "== workload có compute GPU: gọi thẳng vs qua python_backend =="
  for b in 1 16; do
    for run in $(seq 1 $MEASURE_RUNS); do
      bench gpu_conv "X:150528" "$b" 2 "gpu_direct_b${b}_r${run}" $(( 4 * b + 1048576 ))
      bench py_gpu   "IN:150528" "$b" 2 "gpu_py_b${b}_r${run}"    $(( 4 * b + 1048576 ))
    done
  done
}

run_chaos() {
  echo "== giết stub giữa tải: server có sống, model có tự hồi phục không =="
  local log=results/chaos.txt
  grep -qs stub_pids_after $log && { echo "[bỏ qua] chaos.txt đã có"; return; }
  : > $log
  # tải phải còn chạy lúc giết: 30000 request ở ~600/s = hàng chục giây (lần khói 27/9 tải xong sau 3s,
  # giết stub lúc không còn tải -> bài vô hiệu)
  BENCH_REQS=30000 bench identity_in "IN:${TENSOR_ELEMS_LARGE}" 1 2 "chaos_load" "$(out_size identity_in 0 1)" &
  local load=$!
  sleep 10
  # PID stub của đúng model identity_in (không lẫn identity_in_i2/_i4) trước và sau khi giết
  local pids='for d in /proc/[0-9]*; do tr "\0" " " < $d/cmdline 2>/dev/null |
      grep -q "triton_python_backend_stub /models/identity_in/1/model.py" && echo ${d#/proc/}; done'
  echo "$(date +%s.%N) stub_pids_before=$(docker exec $SRV sh -c "$pids" | paste -sd,)" >> $log
  docker exec $SRV sh -c "for p in \$($pids); do kill -9 \$p; done" || true
  echo "$(date +%s.%N) killed; load_still_running=$(kill -0 $load 2>/dev/null && echo yes || echo no)" >> $log
  for i in $(seq 1 60); do
    echo "$(date +%s.%N) server=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready)" \
         "model=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/models/identity_in/ready)" >> $log
    sleep 1
  done
  echo "$(date +%s.%N) stub_pids_after=$(docker exec $SRV sh -c "$pids" | paste -sd,)" >> $log
  wait $load || true
  docker logs --tail 30 $SRV >> $log 2>&1
  # grep -c trả mã 1 khi đếm được 0 -> với set -e sẽ giết cả script
  echo "  giây server còn sống: $(grep -c "server=200" $log || true)"
  echo "  giây model sẵn sàng: $(grep -c "model=200" $log || true)"
}

# sinh cm_cpu_w<K> (đốt CPU K ms) và cm_gpu_w<K> (gọi cm_spin_d<N> qua BLS); cm_spin_d<N> sinh bằng gen_spin.py
gen_costmap_models() {
  local depths=""
  for kv in $COSTMAP_SPIN; do local n=${kv#*:}; [ "$n" != 0 ] && depths="$depths $n"; done
  local need=0
  for n in $depths; do [ -f "models/cm_spin_d$n/1/model.onnx" ] || need=1; done
  if [ "$need" = 1 ]; then
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/h" $IMG \
      sh -c "pip install -q --user onnx==1.16.2 && python3 /h/gen_spin.py /h/models $depths" > results/gen_spin.log 2>&1
  fi
  for kv in $COSTMAP_SPIN; do
    local k=${kv%%:*} n=${kv#*:} gm=""
    [ "$n" != 0 ] && gm="cm_spin_d$n"
    rm -rf "models/cm_cpu_w$k" "models/cm_gpu_w$k"
    cp -r models/cm_cpu "models/cm_cpu_w$k"; cp -r models/cm_gpu "models/cm_gpu_w$k"
    sed -i "s/string_value: \"0\"/string_value: \"$k\"/" "models/cm_cpu_w$k/config.pbtxt"
    sed -i "s/string_value: \"\"/string_value: \"$gm\"/" "models/cm_gpu_w$k/config.pbtxt"
  done
}

run_costmap() {
  echo "== bản đồ chi phí: cỡ ảnh x batch x thời gian model, CPU và GPU =="
  # hiệu chuẩn: model GPU spin tốn thật bao nhiêu (gọi thẳng, không qua python_backend)
  for kv in $COSTMAP_SPIN; do
    local n=${kv#*:}; [ "$n" = 0 ] && continue
    for run in $(seq 1 $COSTMAP_RUNS); do
      BENCH_REQS=100 bench "cm_spin_d$n" "X:1,1" 1 1 "cm_cal_d${n}_r${run}" 1048576
    done
  done
  for kind in cpu gpu; do
    for k in $COSTMAP_WORK; do
      for e in $COSTMAP_ELEMS; do
        for b in $COSTMAP_BATCH; do
          [ "$e" -ge 6220800 ] && [ "$b" -ge 16 ] && continue
          for run in $(seq 1 $COSTMAP_RUNS); do
            BENCH_REQS=100 bench "cm_${kind}_w${k}" "IN:${e}" "$b" 1 "cm_${kind}_w${k}_s${e}_b${b}_r${run}" $(( 4 * b + 1048576 ))
          done
        done
      done
    done
  done
}

# hold_k_<N>: model cầm N mảng input gần nhất
gen_hold_models() {
  for n in $HOLD_COUNTS; do
    rm -rf "models/hold_k_$n"; cp -r models/hold_k "models/hold_k_$n"
    sed -i "s/string_value: \"10\"/string_value: \"$n\"/" "models/hold_k_$n/config.pbtxt"
  done
}

# Model cầm nhiều mảng: /dev/shm + RSS có phình không, dữ liệu đang cầm còn nguyên không,
# và tắt server ngay lúc model còn cầm có sập không.
run_hold() {
  echo "== cầm nhiều mảng + tắt server lúc đang cầm =="
  for n in $HOLD_COUNTS; do
    start_server
    local pid; pid=$(pgrep -f "^/opt/tritonserver/backends/python/triton_python_backend_stub /models/hold_k_$n/" | head -1)
    ( while :; do
        # RssAnon = RAM riêng, RssShmem = trang shm đã chạm (đếm chung với /dev/shm, không cộng hai lần)
        local sp; sp=$(pgrep -x tritonserver)
        echo "$(date +%s) shm_bytes=$(du -sb /dev/shm 2>/dev/null | cut -f1)" \
             "stub_anon_kb=$(sudo awk '/RssAnon/{print $2}' /proc/$pid/status 2>/dev/null)" \
             "stub_shmem_kb=$(sudo awk '/RssShmem/{print $2}' /proc/$pid/status 2>/dev/null)" \
             "srv_anon_kb=$(sudo awk '/RssAnon/{print $2}' /proc/$sp/status 2>/dev/null)" \
             "srv_shmem_kb=$(sudo awk '/RssShmem/{print $2}' /proc/$sp/status 2>/dev/null)"
        sleep 2
      done ) > "results/hold_mem_$n.log" 2>&1 &
    local smp=$!
    BENCH_REQS=$HOLD_REQS bench "hold_k_$n" "IN:${TENSOR_ELEMS_LARGE}" 1 1 "hold_k${n}_r1" "$(out_size identity_in 0 1)"
    sleep 4; kill $smp 2>/dev/null || true
    # tắt server NGAY khi model vẫn đang cầm n mảng
    local f="results/shutdown_$n.txt"
    echo "held_bad=$(docker logs $SRV 2>&1 | grep -c HOLDK_BAD)" > "$f"
    local t0; t0=$(date +%s.%N)
    docker stop -t 30 $SRV >/dev/null 2>&1 || true
    echo "stop_seconds=$(echo "$(date +%s.%N) - $t0" | bc)" >> "$f"
    echo "exit_code=$(docker inspect -f '{{.State.ExitCode}}' $SRV 2>/dev/null)" >> "$f"
    echo "oom_killed=$(docker inspect -f '{{.State.OOMKilled}}' $SRV 2>/dev/null)" >> "$f"
    echo "crash_lines=$(docker logs $SRV 2>&1 | grep -ciE 'segmentation|signal \(|core dump|terminate called|double free|corrupt')" >> "$f"
    docker logs $SRV 2>&1 | grep -iE 'segmentation|signal \(|core dump|terminate called|double free|corrupt' | sed 's/^/  crash_match: /' >> "$f"
    docker logs --tail 15 $SRV >> "$f" 2>&1
    docker rm -f $SRV >/dev/null 2>&1 || true
    echo "  N=$n: $(head -5 "$f" | tr '\n' ' ')"
  done
}

# Rò shm theo nguyên lý ShmLeakDetector của Triton. Cần module triton_shm_monitor (patch/build_stub.sh BUILD_MONITOR=1).
run_leak() {
  echo "== rò shared memory (bộ nhớ trống trước/sau workload) =="
  local mon=${MONITOR_DIR_HOST:-$HOME/patch/out}
  ls "$mon"/triton_shm_monitor*.so >/dev/null 2>&1 || { echo "thiếu $mon/triton_shm_monitor*.so"; return 1; }
  # chạy root: vùng shm của Triton thuộc root, module phải mở được để đọc bộ nhớ trống
  docker run --rm --network $NET --ipc=host -v "$mon:/m:ro" -v "$PWD:/h" -e OUT=/h/results/leak.json \
    -e LEAK_POSITIVE_CONTROL="${LEAK_POSITIVE_CONTROL:-0}" \
    $SDK python3 /h/leak_check.py > results/leak.err 2>&1 || true
  sudo chown "$(id -u):$(id -g)" results/leak.json 2>/dev/null || true
  python3 -c "import json;d=json.load(open('results/leak.json'));print('  request:',d['requests'],'rò:',d['leak_detected']);[print('  ',r['region'][-12:],r['free_before'],'->',r['free_after'],'rò',r['leak_bytes']) for r in d['regions']]" \
    || { echo "  leak_check lỗi:"; tail -5 results/leak.err; }
}

run_soak() {
  echo "== soak ${SOAK_MINUTES}m (lộ leak/drift), transport=$TRANSPORT =="
  [ -f results/soak.done ] && { echo "[bỏ qua] soak đã xong"; return; }
  # soak dở (máy tắt giữa chừng) không nối tiếp được: khoảng trống làm sai số trôi -> cất riêng, chạy lại đủ
  [ -f results/soak.log ] && mv results/soak.log "results/soak_partial_$(date +%s).log"
  local end=$(( $(date +%s) + SOAK_MINUTES*60 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    local ts=$(date +%s)
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    # RSS của tritonserver (pid 1) + tổng RSS mọi stub, đơn vị kB
    local rss=$(docker exec $SRV sh -c 'grep VmRSS /proc/1/status | awk "{print \$2}";
        for d in /proc/[0-9]*; do grep -q triton_python_backend_stub $d/cmdline 2>/dev/null &&
        grep VmRSS $d/status | awk "{print \$2}"; done' 2>/dev/null | paste -sd' ')
    echo "$ts vram_mb=$vram rss_kb=$rss" >> results/soak.log
    bench bls_pipeline "FRAME:${TENSOR_ELEMS_LARGE}" 1 4 "soak_$ts" 1048576
    sleep 30
  done
  echo "${SOAK_MINUTES}" > results/soak.done
}

# Quan sát: ghi tải nền của máy suốt lượt đo. Việc khác chạy chen (lần 29/9 có job analyze song song)
# làm lẫn số tốc độ -> phải có bằng chứng máy sạch, không đoán.
( while :; do
    echo "$(date +%s) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | paste -sd';')"
    sleep 5
  done ) >> results/_host_load.log 2>&1 &
HOSTLOAD=$!
trap 'kill $HOSTLOAD 2>/dev/null || true' EXIT

gen_instances
gen_dtype_models
gen_hold_models
[ "$MODE" = costmap ] && gen_costmap_models
# fail-closed: thiếu model ONNX thì tritonserver không lên -> dừng sớm, nói rõ vì sao
[ -f models/gpu_conv/1/model.onnx ] || { echo "thiếu models/gpu_conv/1/model.onnx -> chạy bash setup.sh"; exit 1; }
[ "$MODE" = floor ] || start_server
case "$MODE" in
  floor)     run_floor ;;
  identity)  run_identity ;;
  sweep)     run_sweep ;;
  inst)      run_inst ;;
  asym)      run_asym ;;
  bls)       run_bls ;;
  integrity) run_integrity ;;
  gpu)       run_gpu ;;
  chaos)     run_chaos ;;
  costmap)   run_costmap ;;
  hold)      run_hold ;;
  leak)      run_leak ;;
  soak)      run_soak ;;
  # chaos chạy cuối trước soak: giết stub có thể làm model hỏng cho các bài sau
  full)      run_floor
             for st in integrity sweep inst asym gpu bls chaos; do ensure_server $st; run_$st; done
             start_server; run_soak ;;
  *) echo "dùng: bash run.sh floor|identity|sweep|inst|asym|bls|integrity|gpu|chaos|soak|full [shm|http]"; exit 1 ;;
esac
stop_server; docker rm -f $PA >/dev/null 2>&1 || true
echo "== đo xong. Tổng hợp: python3 measure.py $MODE =="
python3 measure.py "$MODE"
