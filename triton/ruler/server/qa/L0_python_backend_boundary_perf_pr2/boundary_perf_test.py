#!/usr/bin/env python3

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Boundary cost of the Python backend: what it costs to move a tensor from
the server core into the Python model (stub), per sample, as batch size grows.

For each case (model, elements per sample, batch, concurrency) it measures:
  - samples/s seen by the client (inputs sent through system shared memory)
  - server time per sample (compute_infer from the statistics extension)
  - minor page faults of the stub process per sample (fresh memory touched)
  - per-sample cost relative to batch 1 of the same tensor size
  - gap to the physical floor: server time per sample divided by the time of
    one plain memcpy of the same bytes on this machine

Report mode (default): prints the table and writes JSON, never fails.
Gate mode: set any BOUNDARY_MAX_* variable and the test fails when a case
exceeds it.
"""

import json
import os
import statistics
import threading
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm

_url = os.environ.get("TRITONSERVER_IPADDR", "localhost") + ":8001"
MODELS = os.environ.get("BOUNDARY_MODELS", "input_only_fp32 identity_fp32").split()
ELEMS = [int(x) for x in os.environ.get("BOUNDARY_ELEMS", "1 150528 1555200").split()]
BATCHES = [int(x) for x in os.environ.get("BOUNDARY_BATCHES", "1 16").split()]
CONCURRENCY = [int(x) for x in os.environ.get("BOUNDARY_CONCURRENCY", "1").split()]
WARMUP = int(os.environ.get("BOUNDARY_WARMUP", "20"))
ROUNDS = int(os.environ.get("BOUNDARY_ROUNDS", "10"))
REQUESTS = int(os.environ.get("BOUNDARY_REQUESTS", "50"))
OUT = os.environ.get("BOUNDARY_RESULT", "boundary_perf.json")


def _gate(name):
    v = os.environ.get(name)
    return float(v) if v else None


MAX_FAULTS_PER_SAMPLE = _gate("BOUNDARY_MAX_FAULTS_PER_SAMPLE")
MAX_BATCH_COST_RATIO = _gate("BOUNDARY_MAX_BATCH_COST_RATIO")
MAX_FLOOR_GAP = _gate("BOUNDARY_MAX_FLOOR_GAP")


def stub_faults(model):
    """Sum of minor page faults of the stub process(es) serving `model`."""
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore")
            if "triton_python_backend_stub" not in cmd or f"/{model}/" not in cmd:
                continue
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            total += int(fields[7])  # minflt
        except (OSError, IndexError, ValueError):
            continue
    return total


def server_stats(client, model):
    s = client.get_inference_statistics(model, as_json=True)["model_stats"][0]
    infer = s["inference_stats"]
    return {
        "inference_count": int(s.get("inference_count", 0)),
        "compute_input_ns": int(infer["compute_input"].get("ns", 0)),
        "compute_infer_ns": int(infer["compute_infer"].get("ns", 0)),
    }


def copy_floor_us(nbytes, repeat=30):
    """Median time of one memcpy of `nbytes` into an existing buffer."""
    src = np.random.rand(nbytes // 4 or 1).astype(np.float32)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    xs = []
    for _ in range(repeat):
        t = time.perf_counter()
        np.copyto(dst, src)
        xs.append(time.perf_counter() - t)
    return statistics.median(xs) * 1e6


class _Sender:
    """One client thread with its own input/output system shared memory."""

    def __init__(self, idx, model, elems, batch):
        self.model = model
        self.client = grpcclient.InferenceServerClient(_url)
        out_elems = elems if model == "identity_fp32" else 1
        in_bytes = batch * elems * 4
        out_bytes = batch * out_elems * 4
        tag = f"boundary_{os.getpid()}_{idx}"
        self.in_name, self.out_name = tag + "_in", tag + "_out"
        self.in_h = shm.create_shared_memory_region(
            self.in_name, "/" + self.in_name, in_bytes
        )
        self.out_h = shm.create_shared_memory_region(
            self.out_name, "/" + self.out_name, out_bytes
        )
        shm.set_shared_memory_region(self.in_h, [np.ones((batch, elems), np.float32)])
        self.client.register_system_shared_memory(
            self.in_name, "/" + self.in_name, in_bytes
        )
        self.client.register_system_shared_memory(
            self.out_name, "/" + self.out_name, out_bytes
        )
        self.inp = grpcclient.InferInput("INPUT0", [batch, elems], "FP32")
        self.inp.set_shared_memory(self.in_name, in_bytes)
        self.out = grpcclient.InferRequestedOutput("OUTPUT0")
        self.out.set_shared_memory(self.out_name, out_bytes)
        self.error = None

    def run(self, n):
        try:
            for _ in range(n):
                self.client.infer(self.model, [self.inp], outputs=[self.out])
        except Exception as e:  # reported by the caller
            self.error = e

    def close(self):
        for name in (self.in_name, self.out_name):
            self.client.unregister_system_shared_memory(name)
        shm.destroy_shared_memory_region(self.in_h)
        shm.destroy_shared_memory_region(self.out_h)
        self.client.close()


def measure(model, elems, batch, conc):
    senders = [_Sender(i, model, elems, batch) for i in range(conc)]
    stats_client = grpcclient.InferenceServerClient(_url)
    try:
        for s in senders:
            s.run(max(1, WARMUP // conc))
        rounds = []
        for _ in range(ROUNDS):
            f0, s0 = stub_faults(model), server_stats(stats_client, model)
            threads = [
                threading.Thread(target=s.run, args=(REQUESTS,)) for s in senders
            ]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            f1, s1 = stub_faults(model), server_stats(stats_client, model)
            for s in senders:
                if s.error:
                    raise s.error
            samples = s1["inference_count"] - s0["inference_count"]
            rounds.append(
                {
                    "samples_per_s": samples / wall,
                    "server_us_per_sample": (
                        s1["compute_infer_ns"] - s0["compute_infer_ns"]
                    )
                    / 1e3
                    / samples,
                    "faults_per_sample": (f1 - f0) / samples,
                }
            )
    finally:
        for s in senders:
            s.close()
        stats_client.close()
    agg = {}
    for k in rounds[0]:
        xs = [r[k] for r in rounds]
        agg[k] = statistics.mean(xs)
        agg[k + "_std"] = statistics.pstdev(xs)
    return agg


class BoundaryPerfTest(unittest.TestCase):
    def test_boundary_cost(self):
        cases = []
        for model in MODELS:
            for conc in CONCURRENCY:
                for elems in ELEMS:
                    floor = copy_floor_us(elems * 4)
                    base = None
                    for batch in sorted(BATCHES):
                        r = measure(model, elems, batch, conc)
                        r.update(
                            model=model,
                            elems=elems,
                            batch=batch,
                            concurrency=conc,
                            copy_floor_us_per_sample=floor,
                        )
                        r["floor_gap"] = r["server_us_per_sample"] / floor
                        if batch == 1:
                            base = r["server_us_per_sample"]
                        r["batch_cost_ratio"] = (
                            (r["server_us_per_sample"] / base) if base else None
                        )
                        cases.append(r)

        print(
            "\nmodel             elems    batch conc  samples/s     srv us/sample  "
            "faults/sample  cost vs b1  gap to floor"
        )
        for c in cases:
            ratio = f"{c['batch_cost_ratio']:.2f}x" if c["batch_cost_ratio"] else "-"
            print(
                f"{c['model']:<17} {c['elems']:<8} {c['batch']:<5} {c['concurrency']:<4} "
                f"{c['samples_per_s']:>9.1f}±{c['samples_per_s_std']:<6.1f} "
                f"{c['server_us_per_sample']:>10.1f}     {c['faults_per_sample']:>10.1f}     "
                f"{ratio:>8}    {c['floor_gap']:>8.1f}x"
            )
        with open(OUT, "w") as f:
            json.dump(
                {
                    "warmup": WARMUP,
                    "rounds": ROUNDS,
                    "requests_per_round": REQUESTS,
                    "cases": cases,
                },
                f,
                indent=2,
            )

        failures = []
        for c in cases:
            name = f"{c['model']} elems={c['elems']} batch={c['batch']} conc={c['concurrency']}"
            if c["elems"] == 1:
                continue  # fixed per-request overhead, not a data movement case
            if (
                MAX_FAULTS_PER_SAMPLE is not None
                and c["faults_per_sample"] > MAX_FAULTS_PER_SAMPLE
            ):
                failures.append(
                    f"{name}: {c['faults_per_sample']:.1f} faults/sample > {MAX_FAULTS_PER_SAMPLE}"
                )
            if (
                MAX_BATCH_COST_RATIO is not None
                and c["batch_cost_ratio"]
                and c["batch_cost_ratio"] > MAX_BATCH_COST_RATIO
            ):
                failures.append(
                    f"{name}: per-sample cost {c['batch_cost_ratio']:.2f}x batch 1 > {MAX_BATCH_COST_RATIO}"
                )
            if MAX_FLOOR_GAP is not None and c["floor_gap"] > MAX_FLOOR_GAP:
                failures.append(
                    f"{name}: {c['floor_gap']:.1f}x copy floor > {MAX_FLOOR_GAP}"
                )
        self.assertEqual(failures, [], "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
