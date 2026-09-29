# `fedluar_hh` — FedLUAR-native vs ImpRoute, head to head, matched budgets

Staged 2026-08-05. **81 runs, 23 arms, 20 rounds, CIFAR-10 / deep_cnn /
Dirichlet(0.1), 3 workers + 1 aggregator, global held-out eval.**

**Queued on moltres, behind `stab`.** Two corrections to the original plan,
both established by reading the boxes rather than the ledger:

1. **charizard is down.** `charizard-claude` fails `Permission denied
   (publickey)` while the same key works on moltres — the ephemeral live
   overlay wiped its `authorized_keys`. It needs re-provisioning
   (`scripts/moltres_bootstrap2.sh`) via the hardware key before it can host
   anything. Any h150 / renorm6 / commbound data that lived only on charizard
   should be presumed lost.
2. **`commbound` is running on moltres, not charizard**, started 11:44:33,
   and it is chained: one parent shell runs
   `bash run_commbound.sh; bash run_stab.sh`, with `stab` resuming at 15/36.
   So **commbound is not the tail of the chain — `stab` is.** Waiting on
   `COMMBOUND.DONE`, as originally specified, would have started this campaign
   on top of `stab` and made both crawl. The guard waits on
   `campaigns/stab/STAB.DONE` for the appended `=== STAB DONE` line
   (the runners *truncate* their `.DONE` at start, so file existence means
   "started", never "finished").

Everything runs from an isolated tree, `~/fl-framework-fedluar`, so the src
patch below can never reach the campaigns running out of `~/fl-framework`.

## Why

FedLUAR (NeurIPS 2025) reports accuracy at a fixed **round** budget against a
`Comm` ratio (byte-weighted per-layer aggregation count ÷ rounds, uplink only)
and never measures wall-clock anywhere in 32 pages. We reported against
wall-clock in a regime where the wire is ~2.7% of a round, which is why our
numbers looked bad. Re-analysing the existing runs on a bytes axis moved the
shedding arms by +9 to +24.5 pp — but it also showed that at Comm ≈ 0.5 our
best arm (−7.48 pp) sits on FedLUAR's own **blind Random control** (−7.33 pp),
not on their method (−1.12 pp).

Nothing on disk can say whether that gap is FedLUAR's importance metric
earning its keep or a property of our workload, because **FedLUAR-native has
never been run here**. This campaign runs it, alongside the two controls that
decide the question.

## The four selection rules

Every `fedluar*` arm is identical in δ (the fixed skipped count), in the
one-global-set broadcast, in the recycle fill (θ + Δ_prev, FedLUAR Eq. 3–5),
in ε (0 — no latency trigger; FedLUAR has none) and in aging (off — FedLUAR
§3.3 bounds staleness nowhere). They differ in **one** thing:

| arm | selection rule | staleness | in the paper? |
|---|---|---|---|
| `fedluar_d<δ>` | inverse-ratio sampling ‖Δ_ℓ‖/‖θ_ℓ‖ | unbounded | their method |
| `luarand_d<δ>` | uniform i.i.d. resampling | unbounded | their Table 4 "Random" |
| `luarcyc_d<δ>` | deterministic round-robin rotation | **bounded, free** | **never run** |

`luarcyc` is the point. FedLUAR's Table 4 beats i.i.d. uniform Random — but
i.i.d. Random shares their own unbounded-staleness pathology. Round-robin
rotation bounds staleness with no metric and no τ_max, and it is the control
that dominates *our* frontier. If `luarcyc` matches or beats `fedluar` at
matched bytes here, the paper's headline ablation does not survive the harder
control. That is a citable result either way.

Against those:

| arm | what it is |
|---|---|
| `improute_tau<τ>` | our shedding path (`skip_feedback=shed` + aging cap) |
| `cyclic_k<k>` | blind rotation with **freeze** fill (the cp2fix control) |
| `eps0` | per-layer transport, nothing shed — the **instrumented** Comm=1.0 denominator |
| `mono` | monolithic FedAvg — accuracy anchor, pairs with `w3/mono` |

`cyclic_k` vs `luarcyc_d(14−k)` comes free: same rotation, same bytes,
**freeze fill vs recycle fill** — our analogue of FedLUAR's Table 5, the one
place they hold bytes fixed and vary only the fill rule.

## Byte budgets (measured model: 749,401 B/worker, L = 14 tensors)

`luarand`, `luarcyc` and `cyclic` select **byte-blind**, so their uplink share
is exactly `1 − δ/14` resp. `k/14` — exact over a rotation for `luarcyc`/
`cyclic`, in expectation for `luarand`.

| arm | Comm_up | uplink MB/round | Comm_total | provenance |
|---|---|---|---|---|
| `mono`, `eps0` | 1.0000 | 2.2482 | 1.0000 | measured (eps0) / derived (mono) |
| `*_d4` | 0.7143 | 1.6059 | 0.8571 | analytic |
| `*_d6` | 0.5714 | 1.2846 | 0.7857 | analytic |
| `*_d7`, `cyclic_k7` | 0.5000 | 1.1241 | 0.7500 | analytic |
| `*_d9` | 0.3571 | 0.8028 | 0.6786 | analytic |
| `*_d11`, `cyclic_k3` | 0.2143 | 0.4818 | 0.6071 | analytic |
| `fedluar_d<δ>` | **EMERGENT** | — | — | **measure it** |
| `improute_tau<τ>` | **EMERGENT** | — | — | measure it (w3 prior: τ2 .722 τ3 .585 τ5 .344 τ8 .227) |

The δ grid {4, 6, 7, 9, 11} was chosen so the byte-blind arms land within
0.014 of Comm of our measured τ_max frontier, with an exact 0.500 anchor
opposite `cyclic_k7`.

**`fedluar`'s budget is not settable and not predictable.** At δ=7 it lies
anywhere in [0.003, 0.999] depending on whether the draws land on the five
147.5 kB conv kernels (98.4% of the payload) or the seven biases (0.28%).
That is exactly the paper's own situation — their `Comm` column is an
*outcome* they grid-search δ to land, not a setting. So the **design** matches
on count and the **analysis** matches on measured Comm, interpolating along
the grid. Downlink is 2.2482 MB/round in every arm; no knob touches it, so
Comm_total can never fall below 0.5.

Round 0 carries no advice (R₀ = ∅, as in the paper), so every `fedluar*` arm
ships the full model once. Read steady state from rounds ≥ 3, cycle-aligned
(`analysis_common.cycle_stats`), never a naive mean.

## Tiers (specs.txt is tier-ordered — a short window still lands the headline)

| tier | arms | n | runs |
|---|---|---|---|
| 0 | `mono`, `fedluar_d7`, `luarand_d7`, `luarcyc_d7` | 6 | 24 |
| 1 | `eps0`, `improute_tau3`, `improute_tau5`, `cyclic_k7` | 3 | 12 |
| 2 | `{fedluar,luarand,luarcyc}_d{4,6,9,11}` | 3 | 36 |
| 3 | `improute_tau2`, `improute_tau8`, `cyclic_k3` | 3 | 9 |

Tier 0 is the only tier that can support a headline: at n=3 the exact
sign-flip floor is 0.125 > α, so the ROBUST tier is unreachable **by
construction**.

## Optimizer bundle (identical to w3, so numbers pair)

`sgd`, lr 0.1, momentum 0.0, batch 64, 1 epoch/round, 20 rounds,
`global_eval: true`, no server momentum, `watchdog_factor` 3.0 (base).

Deliberate non-uniformity, stated rather than configured away:
`improute_tau*` keeps **ε = 0.3** and `coverage_eft`, exactly as in w3.
`shed` advice *is* "the layers your update was missing", so ε slippage is what
**bootstraps** the shedding — setting ε=0 there would not make the arm cleaner,
it would silently turn it into `eps0`. The consequence — our path couples two
mechanisms where FedLUAR's couples none — is a finding, not a defect.

## Scope conditions (must travel with every number)

- Sampling universe is per-**tensor** (14), not per-**layer** (7 kernel+bias
  groups), so half the slots carry ~0.3% of the bytes. Applied *identically*
  to all four selection rules, so the head-to-head still isolates the
  selection rule — but **do not compare these absolute Comm values to the
  paper's**.
- Byte axis is a coarse staircase (five interchangeable 147.5 kB kernels).
- Control-plane bytes (manifest, skip advice) are on the wire and in no field
  (BYTE-09): every byte figure is payload-only, which flatters the shedding arms.
- No wall-clock is reported. On this testbed the round-time noise floor
  (±10%, proved on bit-identical sentinel re-runs) exceeds the effects.

## Requires a src patch

`skip_feedback: fedluar_random` and `fedluar_cyclic` do not exist in the
committed tree. The reviewable diff is `scripts/fedluar_hh_src.patch`
(schema Literal + `FedAvg._compute_skip_advice` + engine validation + 8 tests,
including a wire-level smoke parameterized over all three modes). **Regenerating
these configs without the patch applied will fail `load_config`** — that is the
intended failure mode, not a bug.

## Commands

```bash
# regenerate + validate all 81 configs (needs the patch applied)
python scripts/gen_fedluar_hh_configs.py

# analyse once the campaign lands
python -m scripts.analyze_fedluar_hh campaigns/fedluar_hh
```
