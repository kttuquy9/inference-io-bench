#!/usr/bin/env bash
# Chuẩn bị máy đo cho bộ test /qa của NVIDIA giống máy CI của họ (chạy 1 lần, chạy lại thì bỏ qua phần đã có):
#  1. mirror/ : kho git nội bộ (bare) common, core, backend, python_backend ở r26.09.
#     python_backend có thêm nhánh $QA_BRANCH (mặc định zcF) = r26.09 + bản vá $QA_PATCH (mặc định bản sửa F: chỉ
#     input zero-copy). Nhánh cũ `zc` (bản D) giữ nguyên. Test nào tự clone + build python_backend (env, examples...)
#     sẽ lấy đúng code đang thử khi PYTHON_BACKEND_REPO_TAG=$QA_BRANCH.
#  2. data/26.09/ : model mẫu NVIDIA để sẵn trên máy CI nội bộ (/data/inferenceserver/<bản>), không công khai.
#     Tự sinh bằng chính script của họ (qa/common/gen_qa_models.py, gen_qa_implicit_models.py): onnx + libtorch.
# Dùng: bash qa_prep.sh   -> ghi qa_prep.done khi xong
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=r26.09
M="$HERE/mirror"; DATA="$HERE/data/26.09"; Q="$HERE/server_r2609"
PATCH="${QA_PATCH:-$HERE/../patch/fix_660_inputs_only_r26.09.patch}"; BR="${QA_BRANCH:-zcF}"
[ -d "$Q/.git" ] || git clone -q --depth 1 -b $TAG https://github.com/triton-inference-server/server "$Q"

mkdir -p "$M"
for r in common core backend python_backend; do
  if [ ! -d "$M/$r.git" ]; then
    git clone -q --bare --depth 1 -b $TAG https://github.com/triton-inference-server/$r "$M/$r.git"
    ln -sfn "$r.git" "$M/$r"     # test clone "<org>/python_backend", cmake lấy "<org>/common.git"
  fi
done
# kho thuộc root (khớp quyền root trong hộp test, khỏi lỗi "dubious ownership") -> phía máy chủ cũng dùng git bằng root
# (git 2.25 của Ubuntu 20.04 không nhận safe.directory qua -c)
G="sudo git"
if ! $G -C "$M/python_backend.git" rev-parse -q --verify "$BR" >/dev/null; then
  W=$(mktemp -d); $G clone -q "$M/python_backend.git" -b $TAG "$W/pb"
  (cd "$W/pb" && sudo patch -p1 -s < "$PATCH" && $G -c user.name=zc -c user.email=zc@local commit -qam "zero-copy CPU model inputs behind TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS ($(basename "$PATCH"))" \
     && $G push -q origin HEAD:refs/heads/"$BR")
  sudo rm -rf "$W"
fi
echo "mirror: $($G -C "$M/python_backend.git" log -1 --format='%h %s' "$BR") (nhánh $BR)"
sudo chown -R root:root "$M"

if [ ! -f "$DATA/.done" ]; then
  mkdir -p "$DATA"
  docker run --rm -v "$Q/qa/common:/gen:ro" -v "$DATA:/out" --entrypoint bash nvcr.io/nvidia/tritonserver:26.09-py3 -c '
    set -e
    pip3 install -q onnx==1.22.0 torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple >/dev/null
    cd /gen
    python3 gen_qa_models.py --onnx --onnx_opset=26 --libtorch --models_dir=/out/qa_model_repository
    python3 gen_qa_implicit_models.py --onnx --onnx_opset=26 --models_dir=/out/qa_sequence_implicit_model_repository
    chmod -R a+rX /out' > "$HERE/qa_prep_gen.log" 2>&1
  for m in qa_model_repository/libtorch_float32_float32_float32 qa_model_repository/libtorch_nobatch_float32_float32_float32 \
           qa_model_repository/onnx_int32_int32_int32 qa_sequence_implicit_model_repository/onnx_nobatch_sequence_int32; do
    [ -d "$DATA/$m" ] || { echo "thiếu model mẫu $m (xem qa_prep_gen.log)"; exit 1; }
  done
  touch "$DATA/.done"
fi
echo "data: $(ls "$DATA/qa_model_repository" | wc -l) model, $(ls "$DATA/qa_sequence_implicit_model_repository" | wc -l) model sequence"
date -u +%FT%TZ > "$HERE/qa_prep.done"
