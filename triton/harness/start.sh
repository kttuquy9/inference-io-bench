#!/usr/bin/env bash
# Chạy harness tách khỏi phiên SSH: mất mạng / đóng terminal thì phép đo vẫn chạy tiếp.
# Dùng giống run.sh:  bash start.sh identity        (shared memory, mặc định)
#                     bash start.sh identity http
#                     bash start.sh all
# Màn hình sẽ hiện log trực tiếp; Ctrl+C chỉ thoát màn xem, KHÔNG dừng phép đo.
# Dừng thật sự: bash stop.sh
cd "$(dirname "$0")"
mkdir -p results
LOG="results/run_$(date +%Y%m%d_%H%M%S)_$1_${2:-shm}.log"
nohup bash run.sh "$@" > "$LOG" 2>&1 &
echo $! > results/.run.pid
echo "đang chạy nền, pid=$(cat results/.run.pid), log=$LOG"
echo "Ctrl+C để thoát màn xem (phép đo vẫn chạy). Xem lại: tail -f $LOG"
sleep 1
# --pid: màn xem tự thoát khi phép đo kết thúc, trả lại dấu nhắc lệnh
tail -f --pid="$(cat results/.run.pid)" "$LOG"
echo "== phép đo đã kết thúc =="
