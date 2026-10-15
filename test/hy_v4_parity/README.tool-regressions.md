# Test-tool regression boundaries

## Draft graph engine seed

`draft_graph_check.py` passes the declared `--seed` to both prompt generation
and `Engine`. The CPU regression exercises the actual child CLI parsing and
engine-call path with a stub engine, including zero and a non-default seed:

```bash
python -m pytest -q test/hy_v4_parity/test_draft_graph_check_contracts.py
```

The test does not export weights, launch a subprocess, initialize the runtime,
or establish graph/eager numerical parity.

## Marlin dense-prefill comparison

`tokenspeed-kernel/test/nvidia/ops/moe/test_marlin_mxfp8_apply.py` checks that
the dense-prefill result retains the input dtype, then compares its float view
against the FP32 reference accumulation. The reference and existing numerical
tolerances are unchanged. This separates the output-storage contract from the
numerical comparison; it does not change the kernel.

CPU wiring and syntax checks do not validate the Marlin output on a GPU. That
requires a separate run of the affected test on supported hardware, and a
single test result does not establish that the complete suite passed.
