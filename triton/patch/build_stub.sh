#!/usr/bin/env bash
# Build 2 stub python_backend r24.08 để đo trước/sau, trong đúng image tritonserver:24.08-py3 đang dùng đo
# (cùng compiler, cùng thư viện, cùng Python 3.10 với stub gốc của container).
#   stub_A = r24.08 nguyên bản, tự build  -> so với stub có sẵn trong container: phải khớp (thước tự chứng minh)
#   stub_B = r24.08 + fix_660_r24.08.patch -> zero-copy thẳng (đo 1/10: hỏng dữ liệu hold_ref, chỉ là mức trần)
#   stub_C = r24.08 + fix_660_pin_r24.08.patch -> zero-copy, mảng numpy tự giữ tham chiếu vào khối shm
#   stub_D = như C nhưng sau cờ TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS (mặc định tắt = hành vi cũ)
#   stub_E = D + log ZCDBG (bật bằng TRITON_PB_ZC_DEBUG=1): ghim/thả khối shm, to_dlpack, huỷ tensor kèm checksum
#            -> tìm lỗi test_ensemble_io (decoupled) chỉ hỏng khi bật cờ (5/10)
#   stub_F = chỉ INPUT của request được zero-copy; output BLS vẫn chép (7/10: output BLS bị giữ 44.8 s -> test rò
#            720 B). Build thêm libtriton_python.so (phía lõi, -Werror) -> out/<tag>/libtriton_python_F.so
#   stub_G = F + log ZCDBG (giờ ms, thời gian giữ khối)
# Dùng: VARIANTS="C" bash build_stub.sh   (mặc định A B C; chạy trên máy đo, cần mạng)
# Kết quả: out/stub_<biến thể>, out/build.log
# Bản mới: SRC_TAG=r26.09 IMG_TAG=26.09 VARIANTS="A D" bash build_stub.sh
#   -> build nguồn r26.09 trong image tritonserver:26.09-py3, dùng fix_660_pin_flag_r26.09.patch (src/ r26.09 = main, cùng nội dung),
#      ra out/r26.09/stub_<biến thể> (không đè stub r24.08)
# BUILD_MONITOR=1: build thêm out/triton_shm_monitor*.so (module Triton dùng trong test rò shm của họ)
set -euo pipefail
cd "$(dirname "$0")"
SRC_TAG=${SRC_TAG:-r24.08}
IMG=nvcr.io/nvidia/tritonserver:${IMG_TAG:-24.08}-py3
VARIANTS=${VARIANTS:-"A B C"}
OUT=out; [ "$SRC_TAG" != r24.08 ] && OUT=out/$SRC_TAG
mkdir -p $OUT
docker run --rm -v "$PWD:/w" -w /w -e VARIANTS="$VARIANTS" -e BUILD_MONITOR="${BUILD_MONITOR:-0}" \
  -e SRC_TAG="$SRC_TAG" -e OUT="$OUT" --entrypoint bash $IMG -c '
  set -euo pipefail
  # image runtime có sẵn nvcc + header CUDA + g++ 11.4, chỉ thiếu cmake và vài thư viện dev
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rapidjson-dev libarchive-dev zlib1g-dev git >/dev/null
  pip install -q "cmake>=3.17,<4"
  for v in $VARIANTS; do
    rm -rf /tmp/src_$v
    git clone -q --depth 1 -b $SRC_TAG https://github.com/triton-inference-server/python_backend /tmp/src_$v
    case $v in
      B) (cd /tmp/src_B && git apply /w/fix_660_r24.08.patch) ;;
      C) (cd /tmp/src_C && patch -p1 < /w/fix_660_pin_r24.08.patch) ;;
      D) (cd /tmp/src_D && patch -p1 < /w/fix_660_pin_flag_$SRC_TAG.patch) ;;
      E) (cd /tmp/src_E && patch -p1 < /w/fix_660_pin_flag_debug_$SRC_TAG.patch) ;;
      F) (cd /tmp/src_F && patch -p1 < /w/fix_660_inputs_only_$SRC_TAG.patch) ;;
      G) (cd /tmp/src_G && patch -p1 < /w/fix_660_inputs_only_debug_$SRC_TAG.patch) ;;
    esac
  done
  for v in $VARIANTS; do
    mkdir -p /tmp/src_$v/build && cd /tmp/src_$v/build
    cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release \
          -DTRITON_BACKEND_REPO_TAG=$SRC_TAG -DTRITON_COMMON_REPO_TAG=$SRC_TAG -DTRITON_CORE_REPO_TAG=$SRC_TAG ..
    make -j"$(nproc)" triton-python-backend-stub
    cp triton_python_backend_stub /w/$OUT/stub_$v
    # bản sẽ nộp: build cả phía lõi với đúng cờ của họ (-Werror) để chắc không vỡ build
    if [ "$v" = F ]; then
      make -j"$(nproc)" triton-python-backend
      cp libtriton_python.so /w/$OUT/libtriton_python_F.so
    fi
    if [ "$BUILD_MONITOR" = 1 ]; then
      make -j"$(nproc)" triton-shm-monitor && cp src/shm_monitor/triton_shm_monitor*.so /w/$OUT/ 2>/dev/null \
        || cp $(find . -name "triton_shm_monitor*.so" | head -1) /w/$OUT/
      BUILD_MONITOR=0
    fi
    cd /w
  done
  # ghi phiên bản nguồn để kết quả đo truy được về đúng mã
  v0=$(echo $VARIANTS | cut -d" " -f1)
  git -C /tmp/src_$v0 log -1 --format="$SRC_TAG %H %ci" > /w/$OUT/source.txt
  chown -R '"$(id -u):$(id -g)"' /w/out
' > $OUT/build.log 2>&1
ls -la $OUT/
for v in $VARIANTS; do sha256sum $OUT/stub_$v; done
