- `qa_base`: name=base image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=r26.09 | stub=image sha256= env=none | start=2026-10-07T03:07:44Z par=2 units=46 | end=2026-10-07T04:32:20Z
- `qa_on`: name=on image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=zc | stub=/home/ubuntu/patch/out/r26.09/stub_D sha256=74cea95fd47c76ed env=TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1 | start=2026-10-06T17:04:55Z par=2 units=46 | name=on image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=zc | stub=/home/ubuntu/patch/out/r26.09/stub_D sha256=74cea95fd47c76ed env=TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1 | start=2026-10-07T02:39:58Z par=2 units=46 | end=2026-10-07T03:07:44Z
- `qa_off`: name=off image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=zc | stub=/home/ubuntu/patch/out/r26.09/stub_D sha256=74cea95fd47c76ed env=none | start=2026-10-07T04:32:20Z par=2 units=46 | end=2026-10-07T05:50:02Z

| đơn vị | qa_base | qa_on | qa_off | kết luận |
|---|---|---|---|---|
| bp:io | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:decoupled | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:ensemble | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:response_sender | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:bls | rc=0 (0✓/0✗) | rc=1 (12✓/1✗) | rc=0 (0✓/0✗) | HỒI QUY |
| bp:core | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:async_execute | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:variants | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| L0_io | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| bp:python_based_backends | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| L0_large_payload | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_infer | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_string_io | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_shared_memory | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_nullchar_string | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_cuda_shared_memory | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_simple_ensemble | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_decoupled | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_sequence_batcher | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_sequence_corrid_batcher | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_batcher | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_backend_output_detail | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_trace | rc=127 (0✓/0✗) | rc=127 (0✓/0✗) | rc=127 (0✓/0✗) | CẢ HAI |
| L0_dlpack_multi_gpu | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_metrics | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_parameters | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| L0_warmup | rc=1 (1✓/0✗) | rc=1 (1✓/0✗) | rc=1 (1✓/0✗) | CẢ HAI |
| L0_model_update | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_memory | rc=2 (0✓/0✗) | rc=2 (0✓/0✗) | rc=2 (0✓/0✗) | CẢ HAI |
| L0_lifecycle | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_memory_growth | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_grpc | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_http | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_sagemaker | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| L0_vertex_ai | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| bp:lifecycle | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:argument_validation | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:logging | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:custom_metrics | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:parameters | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:model_control | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:examples | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:request_rescheduling | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| bp:model_readiness | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | rc=0 (0✓/0✗) | OK |
| L0_model_config | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |
| bp:env | rc=127 (0✓/133964✗) | rc=1 (0✓/0✗) | rc=1 (0✓/0✗) | CẢ HAI |

Tổng: OK 18, HỒI QUY 1, CẢ HAI 27
