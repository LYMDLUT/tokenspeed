# Bounded generation diagnostics

This directory contains CPU preparation tools for collecting matched prefill
and decode evidence. It does not install source hooks, contact a running service,
or provide a live execution command.

From the repository root, run the standard-library supervisor tests:

```bash
python3 test/diagnostics/hy4_r6/c4_followup_r6_driver.py --selftest
python3 test/diagnostics/hy4_r6/c4_followup_r6_driver.py --plan
```

The source-boundary frontend checks require the repository's CPU dependencies,
including `msgspec`, `fastapi`, and `httpx`:

```bash
TOKENSPEED_CPU_TEST_ROOT="$PWD" python3 -m unittest discover \
  -s test/diagnostics/hy4_r6 -p focused_cpu_tests.py -v
```

The focused harness compiles the actual selected production bodies and existing
regression tests while omitting GPU startup imports and unused engine bases.
It covers frontend state and ASGI contracts, not deployment-level completion or
scheduler drain. Use normal imports and the full relevant test modules in the
target dependency environment before deployment.

A future live integration must supply explicit limits, the approved input SHA,
authorization and execution-rule references, a monotonic clock, transport,
preflight evidence, and a capture reader. These inputs are operator-owned and
must not be embedded in source or committed as raw run artifacts.

The supervisor validates capture receipts, not tensor bytes or numerical
correctness. Source-bound instrumentation, independent tensor analysis, and a
conditional follow-up controller sharing the same budget remain to be built.

See [the validation plan](../../../docs/recipes/hy4-r6-validation.md) for
implementation gates, evidence requirements, and the bounded experiment.
