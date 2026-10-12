#!/usr/bin/env bash
# Dừng phép đo đang chạy nền và dọn container.
cd "$(dirname "$0")"
if [ -f results/.run.pid ] && kill -0 "$(cat results/.run.pid)" 2>/dev/null; then
  pkill -P "$(cat results/.run.pid)" 2>/dev/null || true
  kill "$(cat results/.run.pid)" 2>/dev/null || true
  echo "đã dừng pid $(cat results/.run.pid)"
else
  echo "không có phép đo nào đang chạy"
fi
rm -f results/.run.pid
docker rm -f triton_srv >/dev/null 2>&1 && echo "đã tắt triton_srv" || true
