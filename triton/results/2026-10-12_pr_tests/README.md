# Thước theo khuôn qa NVIDIA (test.sh trong hộp test), 12/10

PR1 (L0_python_backend_boundary_perf, chỉ in số): gốc XANH, F XANH.
PR2 (thêm lượt TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1 + gác cổng fault/ảnh <= 5): gốc ĐỎ (17.8 faults/sample > 5), F XANH.
Chi tiết: pr_tests.md.
