# Triton python_backend: zero-copy CPU inputs

Evidence and tooling for one change to
[triton-inference-server/python_backend](https://github.com/triton-inference-server/python_backend):
CPU request inputs are copied into a new NumPy buffer in the stub on every request. With the opt-in
`TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1`, the NumPy array of a request input wraps the shared-memory
buffer instead, and holds its own reference to that block.

Everything below runs on one GPU machine with Docker and the NVIDIA container runtime. All numbers in
`results/` were taken on AWS g4dn.xlarge (Tesla T4, driver 535.161.08).

## Layout

| Path | What it is |
|---|---|
| `patch/fix_660_zero_copy_inputs_main.patch` | the change (code + README), applies to python_backend `main` |
| `patch/fix_660_inputs_only_r26.09.patch` | same code change on `r26.09` (what was measured) |
| `patch/build_stub.sh` | builds the stub (and `libtriton_python.so`) inside `tritonserver:<tag>-py3` |
| `ruler/server/qa/L0_python_backend_boundary_perf/` | QA test in the `server/qa` format: per-sample cost by batch size, new pages in the stub, gap to a plain memcpy |
| `ruler/run_ruler.sh` | runs that QA test against a stub, the way CI runs it (server + client on one machine) |
| `ruler/qa_prep.sh`, `ruler/qa_run.sh`, `ruler/qa_compare.py` | run every `server/qa` group that uses Python models on stock / flag off / flag on, and compare |
| `harness/` | throughput / data-integrity harness (perf_analyzer, system shared memory) |
| `results/` | raw JSON/CSV with the stub hash, image tag and environment of each run |

## Reproduce

```bash
# 1. Build the patched stub for tritonserver 26.09 (patch/out/r26.09/stub_F, plus the shm monitor module used by the qa leak checks)
cd patch && SRC_TAG=r26.09 IMG_TAG=26.09 VARIANTS="F" BUILD_MONITOR=1 bash build_stub.sh

# 2. QA test: stock stub vs patched stub with the flag on, page-fault gate of 5 per sample
cd ../ruler
BOUNDARY_MAX_FAULTS_PER_SAMPLE=5 bash run_ruler.sh base                      # fails (red)
BOUNDARY_MAX_FAULTS_PER_SAMPLE=5 bash run_ruler.sh fix ../patch/out/r26.09/stub_F \
    TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1                                  # passes (green)

# 3. Existing server/qa tests that use Python models, stock vs patched (several hours)
bash qa_prep.sh
bash qa_run.sh base
bash qa_run.sh on ../patch/out/r26.09/stub_F TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1
python3 qa_compare.py qa_base qa_on
```

`run_ruler.sh` writes `out_<name>/boundary_perf.json`, the client log and the server log.
`qa_run.sh` writes one log per test group plus `summary.tsv` and `_run.txt` (stub sha256, flag, qa commit).

## Notes

- `qa_prep.sh` generates the ONNX/libtorch test models with the generators in `server/qa/common` and
  builds a local git mirror so groups that rebuild python_backend from source build the patched code.
  `qa/L0_*` folders that need other internal test assets (prebuilt C++ test binaries, more model
  repositories) fail on the stock stub as well; `qa_compare.py` reports them separately.
- Heavy groups (`bp:core`, `L0_large_payload`) run alone: on a 16 GB machine they run out of memory when
  run next to another group.
