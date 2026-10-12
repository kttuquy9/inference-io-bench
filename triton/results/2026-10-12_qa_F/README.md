# Gom /qa với bản sửa F (12/10)

- `qa_compare_F.md`: 46 đơn vị qa (r26.09@8187625), gốc / F bật cờ / F tắt cờ. OK 18, HỒI QUY 1 (custom_metrics, F bật, 1 lần), CẢ HAI 27 (thiếu hạ tầng nội bộ NVIDIA, chưa kiểm được).
- `qa_compare.md`: lượt trước với bản D (7/10): HỒI QUY 1 = bls rò 720 B -> dẫn tới bản sửa F (output BLS giữ 44.8 s).
- `repeats.md`: custom_metrics lặp 6 lần gốc/F/D đều qua (model không có input -> đường code F không chạy tới); bls lặp 3 lần gốc/bản D đều qua.
