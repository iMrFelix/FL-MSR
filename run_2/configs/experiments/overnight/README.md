# Overnight Evaluation — Stage Flow and Supervisor Runbook

Implements the staged matrix of `writeup/01-candidate-selection.md` §6
(56 runs: Stage A 15, Stage B 25, Stage C 16) with the staged-ε design of
`writeup/00-design.md` §4.2–4.3.  Generated configs live in
`stage_a/`, `stage_b/`, `stage_c/` next to this file; run specs, status,
run outputs, and analysis artifacts live under `results/overnight/`
(gitignored runtime state).

Decisions taken during the night are logged with reasoning in
`writeup/02-overnight-log.md`.

## Moving parts

| Piece | What it does |
|---|---|
| `scripts/gen_overnight_specs.py` | Emits per-run YAML configs (here) + run-spec JSONs (queue). Stage A is fixed; B/C are parameterized by the ε selection and the winners. Deterministic; refuses to silently overwrite differing files (`--force`). |
| `scripts/overnight_runner.py` | Idempotent sequential queue consumer. Skips specs whose `output_dir/results/report.json` exists; hard per-run timeout with process-group kill + `docker rm -f node-0..3 monitor` + compose-network sweep; appends `{run_id, started, ended, status}` to `<queue>/status.jsonl`; re-scans the queue after every run (specs may be appended mid-stage); exits when nothing is pending. |
| `scripts/eps_knee_selection.py` | Gate ruling G6: affine per-class cost fit from receiver telemetry, β(ε)/predicted-t_ε curves via the deployed CoverageEFT code path, knee candidates + 1/16 anchor (cap 0.5, ≤3 values, fallback {0.0625, 0.2, 0.5}), Stage-A noise floor. |
| `scripts/posthoc_cost.py` | RQ4 $-scoring per κ price vector from logged bytes + cancelled-byte telemetry (no extra runs). |
| `scripts/offline_opt_gap.py` | Per-round optimality gap: exact tail-subset enumeration (L=14) + LP fluid bound vs the deployed scheduler's predicted/realized t_ε. |

Every run is launched as `python -m scripts.run --config <yaml>
--output-dir results/overnight/runs/<run_id>`; the report the runner keys
on is written by the monitor container to `<output_dir>/results/report.json`.

One experiment at a time (fixed container names/ports, shared 10.0.0.0/24
subnet); the runner enforces a clean docker slate before and after each run.

## Stage flow (what the supervisor runs, in order)

All commands from the repo root with the venv active:

```sh
source .venv/bin/activate
```

### 0. One-time prep

```sh
# Rebuild images once per code state (the launcher also rebuilds, cached):
docker build -t fl-node:latest    -f docker/Dockerfile.node    .
docker build -t fl-monitor:latest -f docker/Dockerfile.monitor .

# Full test suite must be green before Stage A:
python -m pytest tests/ -q --ignore=tests/test_cifar10_cnn.py
```

### 1. Stage A — ε = 0 screening (15 runs, ~80 min)

```sh
python -m scripts.gen_overnight_specs --stage A
python -m scripts.overnight_runner --queue-dir results/overnight/queue --stage A
```

Matrix: coverage-EFT × {delta_sq_norm, relative, snr_reweight, raw_norm} ×2
repeats, byte_balanced ×2, gap_based (native raw-norm) ×2, monolithic ×3.
3 rounds, ε=0, same seed (repeats measure the noise floor, not seed
variance). Watch `results/overnight/queue/status.jsonl`; per-run container
logs are in `<output_dir>/run.log`.

### 2. Offline ε-selection (scripted, ~minutes)

```sh
python -m scripts.eps_knee_selection \
    --runs results/overnight/runs \
    --groups coveft \
    --out  results/overnight/analysis/eps_selection.json \
    --plot results/overnight/analysis/eps_curves.png
```

Read `chosen_eps` and `noise_floor` from the JSON. Supervisor judgement
calls at this point:

- arms whose predicted t_ε improvement sits below `noise_floor` are cut
  before Stage B (log the cut in `writeup/02-overnight-log.md`);
- a flat β-curve means the fallback grid {0.0625, 0.2, 0.5} was selected —
  say so in the log;
- an interesting extra knee can be appended later as an extra spec without
  restarting anything.

### 3. Stage B — slippage screening at the chosen ε (25 runs, ~125 min)

```sh
python -m scripts.gen_overnight_specs --stage B \
    --eps <eps_low> <eps_mid> <eps_high> \
    --winner-metric <stage-A winner> \
    --runner-up-metric <stage-A runner-up> \
    --winner-assignment coverage_eft
python -m scripts.overnight_runner --queue-dir results/overnight/queue --stage B
```

Composition: slippage {drop, recycle_last_delta, renormalize} × 3ε × 2
repeats (18) + starvation controls at ε-high (additive-capped aging,
stochastic tail; 4) + runner-up metric spot-check (2) + network-blind
cyclic control (1).

**Cut lines if late** (gate §6): run_ids are ordered so first-cut arms sort
last — delete spec JSONs from the end of the queue: first
`b14_cyclic_*`, then `b13_spot_*`.

### 4. Stage C — 10-round accuracy validation (16 runs, ~225 min)

```sh
python -m scripts.gen_overnight_specs --stage C \
    --eps-low <eps_low> --eps-high <eps_high> \
    --winner-metric <B winner metric> \
    --winner-assignment <B winner assignment> \
    --winner-slippage <B winner slippage>
python -m scripts.overnight_runner --queue-dir results/overnight/queue --stage C
```

Composition: monolithic ×2 seeds, ε=0 bug-detector ×2 (accuracy divergence
here = implementation bug, not science), best triplet at ε-low ×3 and
ε-high ×3 seeds, drop-control at ε-high ×2, warm-up ε-schedule ×2, cyclic
×2 (first cut: `c07_cyclic_*`). Accuracy is the **veto KPI**.

### 5. Post-hoc analyses (no runs)

```sh
python -m scripts.posthoc_cost   --runs results/overnight/runs \
    --out results/overnight/analysis/costs.csv
python -m scripts.offline_opt_gap --runs results/overnight/runs \
    --eps <chosen eps values> \
    --out results/overnight/analysis/opt_gap.csv
```

## Operational notes

- **Restart-safety:** the runner may be killed at any time; re-invoking the
  same command resumes (completed runs are skipped via `report.json`).
  Failed/timed-out runs are retried only on a *new* runner invocation, so a
  crashing config cannot loop all night.
- **Timeouts:** specs default to 1500 s (stages A/B) / 3600 s (stage C) —
  the runner-level belt-and-braces over the in-engine watchdog (G4). On
  timeout the runner kills the process group, force-removes
  `node-0..3 monitor`, compose-downs the run's network, marks
  `status=timeout`, and continues.
- **Appending runs mid-stage:** drop a new spec JSON into
  `results/overnight/queue/` (use `gen_overnight_specs` functions or copy
  an existing spec); the runner picks it up on its next queue re-scan.
- **Hygiene** (design doc §4.3): quiet host, no CPU pinning (E4), one
  runner at a time, seeds recorded in every spec.
