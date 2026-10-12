## Tốc độ (ảnh/giây, trung bình 2 vòng × 10 lần) — Triton 26.09, T4

| ô | gốc NVIDIA | tự build | vá, cờ TẮT | vá, cờ BẬT | bật/gốc |
|---|---|---|---|---|---|
| identity_fp32_s12288_b1 | 3610±51 | 3532±73 | 3596±59 | 3692±33 | 1.02× | 
| identity_fp32_s12288_b16 | 28786±303 | 28338±581 | 28690±268 | 32092±371 | 1.11× | 
| identity_fp32_s150528_b1 | 2049±17 | 2027±36 | 2045±21 | 2259±24 | 1.10× | 
| identity_fp32_s150528_b16 | 2636±56 | 2578±218 | 2489±133 | 3507±148 | 1.33× | 
| identity_fp32_s1555200_b1 | 320±14 | 313±22 | 294±26 | 417±13 | 1.30× | 
| identity_fp32_s1555200_b16 | 160±0 | 161±1 | 159±1 | 270±2 | 1.68× | 
| identity_in_s12288_b1 | 3732±54 | 3664±83 | 3708±50 | 3805±60 | 1.02× | 
| identity_in_s12288_b16 | 38448±428 | 38117±514 | 38245±249 | 49398±164 | 1.28× | 
| identity_in_s150528_b1 | 2631±27 | 2622±42 | 2632±25 | 3266±9 | 1.24× | 
| identity_in_s150528_b16 | 6721±240 | 5566±947 | 6105±112 | 13537±290 | 2.01× | 
| identity_in_s1555200_b1 | 710±10 | 618±89 | 666±23 | 1248±6 | 1.76× | 
| identity_in_s1555200_b16 | 263±4 | 261±4 | 261±4 | 832±10 | 3.16× | 

n mỗi ô: {'def': 20, 'A': 20, 'Doff': 20, 'Don': 20}

## Toàn vẹn dữ liệu

- def_1: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- def_2: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- A_1: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- A_2: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- Doff_1: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- Doff_2: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- Don_1: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]
- Don_2: keys=['url', 'tests', 'dtype_tests', 'all_dtypes_ok', 'all_echo_exact'] vấn đề=[]

## Hành vi + độ an toàn (cả 8 lượt)

- 9 kiểu dữ liệu × 2 cỡ, echo khớp tuyệt đối, `hold_ref` sai 0: cả 4 bản.
- `mutate_in` (model ghi vào input): gốc / tự build / vá cờ TẮT = 0.0 (hành vi cũ); vá cờ BẬT = 1.0 (ghi thẳng vào shm, đúng như tài liệu cờ).
- Cầm 10/100 mảng 1080p rồi tắt server: `held_bad=0`, exit 0 ở cả 4 bản.
- Page fault mỗi lần đo 1080p batch 16: gốc 340,592 → cờ BẬT 30.
- Image: tritonserver:26.09-py3, driver 535.161.08, SDK bắn tải 24.08 (cùng cái cân với mọi số trước).
