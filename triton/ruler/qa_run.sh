#!/usr/bin/env bash
# Chạy GOM mọi bài trong server/qa (r26.09) có dùng Python backend, trên một stub, song song, nhóm dễ hỏng trước.
# Mỗi đơn vị chạy trong container riêng (mạng riêng -> không tranh cổng, /dev/shm riêng 1 GiB, bản qa sạch riêng).
# Bằng chứng: qa_<tên>/{_run.txt (stub sha, cờ, commit qa, ảnh), summary.tsv, <đơn vị>.log}
#
# Dùng: bash qa_run.sh <tên> [stub] [env server] [đơn vị...]
#   bash qa_run.sh base
#   bash qa_run.sh on ~/patch/out/r26.09/stub_D TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1
#   bash qa_run.sh dbg ~/patch/out/r26.09/stub_E "TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1 TRITON_PB_ZC_DEBUG=1" bp:io
# Đơn vị: bp:<subtest> = qa/L0_backend_python/<subtest>; bp:core = phần lõi python_test.py; L0_xxx = thư mục qa.
# Biến: QA_PAR (mặc định 2 container cùng lúc, máy 4 vCPU), QA_TIMEOUT (giây mỗi đơn vị, mặc định 5400)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?thiếu tên}"; STUB="${2:-}"; SENV="${3:-}"; shift $(( $# < 3 ? $# : 3 ))
[ -f "$HERE/qa_prep.done" ] || echo "CẢNH BÁO: chưa chạy qa_prep.sh -> bài cần model mẫu / kho code nội bộ sẽ trượt ở cả gốc"
IMG=nvcr.io/nvidia/tritonserver:${TRITON_TAG:-26.09-py3}
PAR=${QA_PAR:-2}; TMO=${QA_TIMEOUT:-5400}
Q="$HERE/server_r2609"; OUT="$HERE/qa_$NAME"; mkdir -p "$OUT"

# nhóm có ảnh đi vào model (đụng bản vá) chạy trước; nhóm không liên quan sau; env (build lâu) cuối
DEFAULT_UNITS="bp:io bp:bls bp:decoupled bp:ensemble bp:response_sender bp:core bp:variants bp:async_execute
 bp:python_based_backends L0_io L0_infer L0_large_payload L0_string_io L0_nullchar_string L0_shared_memory
 L0_cuda_shared_memory L0_simple_ensemble L0_decoupled L0_batcher L0_sequence_batcher L0_sequence_corrid_batcher
 L0_dlpack_multi_gpu L0_backend_output_detail L0_trace L0_metrics L0_parameters L0_warmup L0_lifecycle
 L0_model_update L0_memory L0_memory_growth L0_grpc L0_http L0_model_config L0_sagemaker L0_vertex_ai
 bp:lifecycle bp:argument_validation bp:logging bp:custom_metrics bp:parameters bp:model_control
 bp:examples bp:request_rescheduling bp:model_readiness bp:env"
UNITS=("$@"); [ ${#UNITS[@]} -eq 0 ] && read -r -a UNITS <<< "$(echo $DEFAULT_UNITS)"

PBTAG=r26.09; [ -n "$STUB" ] && PBTAG=${QA_BRANCH:-zcF}   # test tự build python_backend thì build đúng code đang thử (khớp qa_prep)
{
  echo "name=$NAME image=$IMG qa=r26.09@$(git -C "$Q" rev-parse --short HEAD) python_backend_tag=$PBTAG"
  echo "stub=${STUB:-image} sha256=$( [ -n "$STUB" ] && sha256sum "$STUB" | cut -c1-16) env=${SENV:-none}"
  echo "start=$(date -u +%FT%TZ) par=$PAR units=${#UNITS[@]}"
} >> "$OUT/_run.txt"
[ -f "$OUT/summary.tsv" ] || printf "unit\trc\tseconds\tpassed\tfailed\tnote\n" > "$OUT/summary.tsv"

run_unit() {
  local u="$1" w log rc t0 t1 cmd
  local tag=${u//[:\/]/_}
  [ -f "$OUT/$tag.done" ] && return 0          # máy tắt giữa chừng: đơn vị xong rồi thì bỏ qua
  w="$OUT/work_$tag"; sudo rm -rf "$w"; mkdir -p "$w"; cp -r "$Q/qa" "$w/qa"
  cp -r "$HERE/server/qa/." "$w/qa/"   # thước của mình (L0_python_backend_boundary_perf[_pr2], python_models/input_only_fp32)
  cp "$HERE/../patch/out/r26.09/"triton_shm_monitor*.so "$w/qa/common/"   # build_stub.sh BUILD_MONITOR=1
  # QA_KEEP_LOGS=1: log server vẫn ghi đè như cũ (test grep file này) nhưng nối thêm vào <log>.all -> không mất lần nào
  [ "${QA_KEEP_LOGS:-0}" = 1 ] && sed -i 's|\$SERVER \$SERVER_ARGS > \$SERVER_LOG 2>&1 &|$SERVER $SERVER_ARGS > >(tee -a ${SERVER_LOG}.all > $SERVER_LOG) 2>\&1 \&|' "$w/qa/common/util.sh"
  log="$OUT/$tag.log"
  local stub_args=() env_args=()
  [ -n "$STUB" ] && stub_args=(-v "$(realpath "$STUB"):/opt/tritonserver/backends/python/triton_python_backend_stub:ro")
  for kv in $SENV; do env_args+=(-e "$kv"); done
  # lệnh của đơn vị ghi ra file trong bản qa riêng (khỏi lồng dấu nháy)
  case $u in
    bp:core)
      awk '/^SERVER_ARGS="\$BASE_SERVER_ARGS --backend-config=python,shm-default-byte-size=5242880"$/{exit} {print}' \
        "$w/qa/L0_backend_python/test.sh" > "$w/qa/L0_backend_python/t_core.sh"
      echo 'exit $RET' >> "$w/qa/L0_backend_python/t_core.sh"
      printf '%s\n' 'cd L0_backend_python && bash -x t_core.sh 26.09' > "$w/qa/_unit.sh" ;;
    bp:*)
      # phần đầu test.sh của họ (biến môi trường + source common) rồi chạy đúng 1 subtest như vòng lặp của họ
      sed -n '/^export TEST_WINDOWS=0/,/^source .\/common.sh/p' "$w/qa/L0_backend_python/test.sh" > "$w/qa/L0_backend_python/t_env.sh"
      cat > "$w/qa/_unit.sh" <<EOS
cd L0_backend_python
source ./t_env.sh
pip3 install -q virtualenv >/dev/null 2>&1
setup_virtualenv
(cd ${u#bp:} && bash -ex test.sh)
rc=\$?
deactivate_virtualenv 2>/dev/null
exit \$rc
EOS
      ;;
    L0_*) printf '%s\n' "cd $u && bash -x test.sh 26.09" > "$w/qa/_unit.sh" ;;
  esac
  cmd="bash /opt/tritonserver/qa/_unit.sh"
  t0=$(date +%s)
  timeout "$TMO" docker run --rm --gpus all --shm-size=1g "${env_args[@]}" "${stub_args[@]}" \
    -v "$w/qa:/opt/tritonserver/qa" -v "$HERE/data:/data/inferenceserver:ro" -v "$HERE/mirror:/mirror:ro" \
    -w /opt/tritonserver/qa \
    -e NVIDIA_TRITON_SERVER_VERSION=26.09 -e REPO_VERSION=26.09 -e BACKENDS=python \
    -e TRITON_REPO_ORGANIZATION=file:///mirror -e PYTHON_BACKEND_REPO_TAG=$PBTAG \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e "GIT_CONFIG_VALUE_0=*" \
    -e TRITON_BACKEND_REPO_TAG=r26.09 -e TRITON_COMMON_REPO_TAG=r26.09 -e TRITON_CORE_REPO_TAG=r26.09 \
    $IMG bash -c "pip3 install -q 'tritonclient[all]' pytest requests ml_dtypes perf-analyzer >/dev/null 2>&1; $cmd" \
    > "$log" 2>&1
  rc=$?; t1=$(date +%s)
  local p f note=""
  p=$(grep -aoE "[0-9]+ passed" "$log" | awk '{s+=$1} END{print s+0}')
  f=$(grep -aoE "[0-9]+ failed" "$log" | awk '{s+=$1} END{print s+0}')
  [ $rc -eq 124 ] && note="timeout ${TMO}s"
  note="$note $(grep -aE '^E +[A-Za-z]*Error|Test FAILED|FAILED\.|cannot stat|No module named' "$log" | head -2 | tr '\t\n' '  ' | cut -c1-160)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$u" "$rc" "$((t1-t0))" "$p" "$f" "$note" >> "$OUT/summary.tsv"
  # giữ log server (bằng chứng), bỏ phần nặng (model, venv)
  mkdir -p "$OUT/srv_$tag"
  find "$w" -name "*server*.log*" | while read -r f; do cp "$f" "$OUT/srv_$tag/$(echo "${f#$w/qa/}" | tr / _)"; done
  sudo rm -rf "$w"
  touch "$OUT/$tag.done"
}

# đơn vị nặng RAM chạy MỘT MÌNH: 12/10 bp:core (test_growth_error gửi 1 GiB, client ăn ~8 GB) chạy cùng bp:bls
# -> RAM trống 8.4 GB -> 0.26 GB trong 1 phút, máy 16 GB đứng hình. Chốt RAM chỉ kiểm lúc mở đơn vị nên không đủ.
SOLO_UNITS=" ${QA_SOLO_UNITS:-bp:core L0_large_payload} "
# hàng đợi song song PAR container
for u in "${UNITS[@]}"; do
  if [[ "$SOLO_UNITS" == *" $u "* ]]; then
    wait                      # chờ mọi đơn vị đang chạy xong
    run_unit "$u"             # chạy một mình, không mở đơn vị khác trong lúc này
    continue
  fi
  # chốt RAM: 6/10 hai bộ test song song làm máy 16 GB đứng hình -> chỉ mở đơn vị mới khi còn đủ RAM trống
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || \
        { [ "$(jobs -rp | wc -l)" -gt 0 ] && [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -lt "${QA_MIN_AVAIL_MB:-6000}" ]; }; do
    sleep 10
  done
  run_unit "$u" &
done
wait
echo "end=$(date -u +%FT%TZ)" >> "$OUT/_run.txt"
column -t -s $'\t' "$OUT/summary.tsv" | cut -c1-200
