# Thước L0_python_backend_boundary_perf — Triton 26.09, T4, gác cổng BOUNDARY_MAX_FAULTS_PER_SAMPLE=5

Kết quả gác cổng (2 vòng xen kẽ): gốc rc=1/rc=1 (ĐỎ), vá cờ tắt rc=1/rc=1 (ĐỎ, = gốc), vá cờ bật rc=0/rc=0 (XANH).

## base_1
```

=================================
== Triton Inference Server SDK ==
=================================

NVIDIA Release 24.08 (build 107631423)

Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

F
======================================================================
FAIL: test_boundary_cost (__main__.BoundaryPerfTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/t/boundary_perf_test.py", line 283, in test_boundary_cost
    self.assertEqual(failures, [], "\n" + "\n".join(failures))
AssertionError: Lists differ: ['input_only_fp32 elems=1555200 batch=16 c[101 chars]5.0'] != []

First list contains 2 additional elements.
First extra element 0:
'input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0'

+ []
- ['input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0',
-  'identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0'] : 
input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0
identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0

----------------------------------------------------------------------

```
## base_2
```
e the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

F
======================================================================
FAIL: test_boundary_cost (__main__.BoundaryPerfTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/t/boundary_perf_test.py", line 283, in test_boundary_cost
    self.assertEqual(failures, [], "\n" + "\n".join(failures))
AssertionError: Lists differ: ['input_only_fp32 elems=1555200 batch=16 c[101 chars]5.0'] != []

First list contains 2 additional elements.
First extra element 0:
'input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0'

+ []
- ['input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0',
-  'identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0'] : 
input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0
identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0

model             elems    batch conc  samples/s     srv us/sample  faults/sample  cost vs b1  gap to floor
input_only_fp32   1        1     1       2355.5±58.2         98.1            0.0        1.00x       148.7x
input_only_fp32   1        16    1      38728.1±622.0         5.7            0.0        0.06x         8.7x
input_only_fp32   150528   1     1       1936.9±104.5       149.3            0.0        1.00x         6.3x
input_only_fp32   150528   16    1       6563.1±185.6        68.6            0.0        0.46x         2.9x
input_only_fp32   1555200  1     1        655.3±12.2        575.3            0.0        1.00x         1.2x
input_only_fp32   1555200  16    1        268.8±2.2        2599.6           17.8        4.52x         5.3x
identity_fp32     1        1     1       2393.0±30.7         94.1            0.0        1.00x       143.2x
identity_fp32     1        16    1      38701.7±487.4         5.8            0.0        0.06x         8.8x
identity_fp32     150528   1     1       1589.1±56.7        197.7            0.0        1.00x         7.2x
identity_fp32     150528   16    1       2894.2±99.5        135.3            0.0        0.68x         4.9x
identity_fp32     1555200  1     1        330.6±4.0        1170.0            0.0        1.00x         2.4x
identity_fp32     1555200  16    1        162.1±0.5        3803.6           49.7        3.25x         7.7x

----------------------------------------------------------------------

```
## off_1
```

=================================
== Triton Inference Server SDK ==
=================================

NVIDIA Release 24.08 (build 107631423)

Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

F
======================================================================
FAIL: test_boundary_cost (__main__.BoundaryPerfTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/t/boundary_perf_test.py", line 283, in test_boundary_cost
    self.assertEqual(failures, [], "\n" + "\n".join(failures))
AssertionError: Lists differ: ['input_only_fp32 elems=1555200 batch=16 c[101 chars]5.0'] != []

First list contains 2 additional elements.
First extra element 0:
'input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0'

+ []
- ['input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0',
-  'identity_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0'] : 
input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0
identity_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0

----------------------------------------------------------------------

```
## off_2
```
e the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .


model             elems    batch conc  samples/s     srv us/sample  faults/sample  cost vs b1  gap to floor
input_only_fp32   1        1     1       2432.0±22.7         86.9            0.0        1.00x       154.3x
input_only_fp32   1        16    1      38806.8±876.4         5.7            0.0        0.07x        10.2x
input_only_fp32   150528   1     1       1864.7±65.3        158.3            0.0        1.00x         5.9x
input_only_fp32   150528   16    1       7090.5±231.9        62.3            0.0        0.39x         2.3x
input_only_fp32   1555200  1     1        668.0±15.0        577.9            0.0        1.00x         1.2x
input_only_fp32   1555200  16    1        268.3±1.4        2631.8           49.7        4.55x         5.4x
identity_fp32     1        1     1       2504.9±24.0         89.5            0.0        1.00x       158.9x
identity_fp32     1        16    1      39170.8±866.3         5.8            0.0        0.07x        10.3x
identity_fp32     150528   1     1       1619.1±57.5        190.9            0.0        1.00x         6.9x
identity_fp32     150528   16    1       2914.8±83.9        129.7            0.0        0.68x         4.7x
identity_fp32     1555200  1     1        334.8±15.4       1164.8            0.0        1.00x         2.4x
identity_fp32     1555200  16    1        161.7±0.3        3802.6           49.7        3.26x         7.8x
F
======================================================================
FAIL: test_boundary_cost (__main__.BoundaryPerfTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/t/boundary_perf_test.py", line 283, in test_boundary_cost
    self.assertEqual(failures, [], "\n" + "\n".join(failures))
AssertionError: Lists differ: ['input_only_fp32 elems=1555200 batch=16 c[101 chars]5.0'] != []

First list contains 2 additional elements.
First extra element 0:
'input_only_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0'

+ []
- ['input_only_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0',
-  'identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0'] : 
input_only_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0
identity_fp32 elems=1555200 batch=16 conc=1: 49.7 faults/sample > 5.0

----------------------------------------------------------------------

```
## on_1
```

=================================
== Triton Inference Server SDK ==
=================================

NVIDIA Release 24.08 (build 107631423)

Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

.
----------------------------------------------------------------------

```
## on_2
```

=================================
== Triton Inference Server SDK ==
=================================

NVIDIA Release 24.08 (build 107631423)

Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

.
----------------------------------------------------------------------

```
