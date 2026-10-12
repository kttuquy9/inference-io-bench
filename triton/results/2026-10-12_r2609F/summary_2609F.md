# Bản sửa F trên Triton 26.09 — tốc độ (ảnh/s, 2 vòng xen kẽ × 10 lần)

| ô | gốc | tự build | F tắt | F bật | bật/gốc |
|---|---|---|---|---|---|
| identity_fp32_s12288_b1 | 3549±27 | 3541±52 | 3567±22 | 3538±68 | 1.00× |
| identity_fp32_s12288_b16 | 28126±219 | 28171±253 | 28183±249 | 31100±394 | 1.11× |
| identity_fp32_s150528_b1 | 2000±18 | 1991±29 | 2011±19 | 2211±23 | 1.11× |
| identity_fp32_s150528_b16 | 2288±369 | 2498±354 | 2198±367 | 3797±155 | 1.66× |
| identity_fp32_s1555200_b1 | 311±43 | 336±5 | 287±50 | 424±31 | 1.36× |
| identity_fp32_s1555200_b16 | 159±2 | 159±2 | 159±1 | 271±2 | 1.71× |
| identity_in_s12288_b1 | 3672±21 | 3655±41 | 3631±58 | 3710±49 | 1.01× |
| identity_in_s12288_b16 | 37625±357 | 37404±685 | 37355±474 | 48409±198 | 1.29× |
| identity_in_s150528_b1 | 2568±39 | 2558±43 | 2562±36 | 3203±27 | 1.25× |
| identity_in_s150528_b16 | 5370±1292 | 5925±959 | 5221±1244 | 13855±294 | 2.58× |
| identity_in_s1555200_b1 | 664±46 | 702±10 | 603±104 | 1237±7 | 1.86× |
| identity_in_s1555200_b16 | 262±4 | 258±3 | 260±5 | 830±10 | 3.17× |

n mỗi ô: {'def': 20, 'A': 20, 'Doff': 20, 'Don': 20}

## Dữ liệu / hành vi

- def_1: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- def_2: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- A_1: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- A_2: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- Doff_1: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- Doff_2: dtypes_ok=True echo=True sai=0 mutate_in=[[0.0]] held_bad=0 held_bad=0
- Don_1: dtypes_ok=True echo=True sai=0 mutate_in=[[1.0]] held_bad=0 held_bad=0
- Don_2: dtypes_ok=True echo=True sai=0 mutate_in=[[1.0]] held_bad=0 held_bad=0
