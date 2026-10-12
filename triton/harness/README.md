# Throughput / data-integrity harness

Measures what it costs to move tensors from the Triton core into a Python backend model, and checks
that the data the model sees is correct. Runs on a GPU machine with Docker and the NVIDIA container
runtime (all numbers in `../results/` come from AWS g4dn.xlarge, Tesla T4).

Inputs go through system shared memory (`perf_analyzer --shared-memory system`) so HTTP/gRPC transport
is not part of the measurement.

```bash
bash setup.sh                 # once: pull the Triton server and SDK images
bash run.sh floor             # memcpy floor of this machine (reuse vs fresh buffer)
bash run.sh sweep             # tensor size x batch x {input only, input+output}
bash run.sh integrity         # exact echo, held arrays, in-place writes, 9 dtypes
bash run.sh hold              # models that keep 10/100 inputs alive; shutdown while held
bash run.sh inst | bls | gpu | chaos | soak | costmap | leak
STUB=/path/to/triton_python_backend_stub bash run.sh sweep   # measure a self-built stub
SRV_ENV="TRITON_PYTHON_BACKEND_ZERO_COPY_INPUTS=1" ...         # extra env for the server
python3 measure.py <name>     # summary JSON in results/ (stub sha256, image, GPU, noise)
```

Each configuration runs `MEASURE_RUNS` times (default 10); `measure.py` reports mean, std and whether
the noise floor is met. Settings are in `config.env` (override any of them from the environment).
Comments in the scripts are in Vietnamese.
