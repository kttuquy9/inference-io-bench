# inference-io-bench

Measurements of how tensors move through ML inference servers: how many copies are made between the
server core and the model, what each copy costs per sample as batch size grows, how far that is from a
plain memcpy on the same machine, and whether the data the model sees is correct.

Each folder covers one server, with the harness, raw results and any proposed change.

| Folder | Server | Status |
|---|---|---|
| [`triton/`](triton/) | Triton Inference Server, Python backend | zero-copy CPU inputs (opt-in): measurements, QA test, patch |

All runs so far: AWS g4dn.xlarge (Tesla T4, driver 535), Docker with the NVIDIA container runtime.
