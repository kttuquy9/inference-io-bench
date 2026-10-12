== qa_prbase
name=prbase image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=r26.09
stub=image sha256= env=none
start=2026-10-12T12:32:43Z par=1 units=2
end=2026-10-12T12:37:25Z
unit	rc	seconds	passed	failed	note
L0_python_backend_boundary_perf	0	117	0	0	 
L0_python_backend_boundary_perf_pr2	1	162	0	0	 
== qa_prF
name=prF image=nvcr.io/nvidia/tritonserver:26.09-py3 qa=r26.09@8187625 python_backend_tag=zcF
stub=/home/ubuntu/patch/out/r26.09/stub_F sha256=d37cbfd0e251c0fd env=none
start=2026-10-12T12:37:25Z par=1 units=2
end=2026-10-12T12:41:41Z
unit	rc	seconds	passed	failed	note
L0_python_backend_boundary_perf	0	117	0	0	 
L0_python_backend_boundary_perf_pr2	0	134	0	0	 

AssertionError: Lists differ: ['input_only_fp32 elems=1555200 batch=16 conc=1: 17.8 faults/sample > 5.0'] != []
