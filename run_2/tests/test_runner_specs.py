"""Tests for the overnight runner, spec generation, and analysis scripts.

Covers the deliverables of writeup/01-candidate-selection.md §6:
- gen_overnight_specs: exact stage matrices (A=15, B=25, C=16), determinism
  (byte-identical regeneration), schema validation of emitted configs, and
  the cut-line ordering contract (first-cut arms sort last).
- overnight_runner: queue loading, skip/pending logic, status JSONL, and
  the ok/error/timeout execution paths with injected commands (no docker).
- overnight_common: telemetry/manifest parsing against the documented
  sidecar schema, the affine cost fit, and the cost-model primitives.
- eps_knee_selection: β(ε) curves + knee picking on synthetic manifests via
  an injected stand-in strategy (the real coverage_eft path is exercised in
  a skip-if-unavailable integration test).
- offline_opt_gap: exact subset enumeration on hand-checkable instances.
- posthoc_cost: κ vectors and $-arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts import overnight_common as oc
from scripts import overnight_runner as runner
from scripts.eps_knee_selection import (
    build_eps_grid,
    compute_curves,
    compute_noise_floor,
    find_knees,
)
from scripts.gen_overnight_specs import (
    Arm,
    build_config,
    stage_a_arms,
    stage_b_arms,
    stage_c_arms,
    write_stage,
)
from scripts.offline_opt_gap import enumerate_opt
from scripts.posthoc_cost import (
    cancelled_bytes_by_class,
    dollars,
    ratio_kappa,
    sent_bytes_by_class,
)
from src.importance.assignment import AssignmentResult, AssignmentStrategy


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class DensityPrefixStrategy(AssignmentStrategy):
    """Deterministic stand-in: density-ascending prefix tail, head on class 0.

    Used so the knee/curve tests validate *this* module's logic without
    depending on the (separately owned) coverage-EFT implementation.
    """

    def assign(self, *, scores, sizes, bandwidths, epsilon, must_receive,
               ages=None, trigger_scores=None):
        total = sum(scores.values())
        order = sorted(
            scores, key=lambda n: (scores[n] / max(sizes[n], 1), n)
        )
        tail: set[str] = set()
        acc = 0.0
        for name in order:
            if name in must_receive:
                continue
            if acc + scores[name] <= epsilon * total + 1e-12:
                tail.add(name)
                acc += scores[name]
            else:
                break
        head = set(scores) - tail
        return AssignmentResult(
            assignment={n: 0 for n in scores}, head=head, tail=tail
        )


def _manifest(run_id="run_x", round_num=1, source="node-1"):
    """3-layer synthetic manifest with known β(ε) step structure.

    Density-ascending order is big -> mid -> small; with total score 10 the
    big layer (80% of bytes) crosses into the tail at ε=0.1 and mid at 0.2.
    """
    return oc.ManifestRecord(
        run_id=run_id,
        round=round_num,
        source=source,
        scores={"big": 1.0, "mid": 1.0, "small": 8.0},
        sizes={"big": 800_000, "mid": 100_000, "small": 1_000},
        classes={"big": 0, "mid": 0, "small": 0},
    )


_ONE_MBPS = {0: oc.ClassCost(alpha_s=0.0, bytes_per_s=125_000.0, source="fit")}


def _write_spec(queue_dir: Path, run_id: str, output_dir: Path,
                stage="A", timeout_s=5.0) -> Path:
    spec = {
        "run_id": run_id,
        "stage": stage,
        "config_path": str(output_dir / "cfg.yaml"),
        "output_dir": str(output_dir),
        "timeout_s": timeout_s,
        "seed": 42,
    }
    path = queue_dir / f"{run_id}.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _mark_complete(output_dir: Path) -> None:
    report = output_dir / "results" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage matrices
# ---------------------------------------------------------------------------

class TestStageAMatrix:

    def test_fifteen_runs_exact_composition(self):
        arms = stage_a_arms()
        assert len(arms) == 15
        by_assignment: dict[str, int] = {}
        for arm in arms:
            key = arm.assignment if arm.mode == "per_layer" else "monolithic"
            by_assignment[key] = by_assignment.get(key, 0) + 1
        assert by_assignment == {
            "coverage_eft": 8,   # 4 metrics x 2 repeats
            "byte_balanced": 2,
            "gap_based": 2,
            "monolithic": 3,
        }

    def test_coverage_eft_covers_all_four_metrics(self):
        arms = [a for a in stage_a_arms() if a.assignment == "coverage_eft"
                and a.mode == "per_layer"]
        assert {a.metric for a in arms} == {
            "delta_sq_norm", "relative", "snr_reweight", "raw_norm"
        }

    def test_gap_based_is_native_raw_norm(self):
        arms = [a for a in stage_a_arms() if a.assignment == "gap_based"]
        assert all(a.metric == "raw_norm" for a in arms)

    def test_screening_parameters(self):
        for arm in stage_a_arms():
            assert arm.rounds == 3
            assert arm.epsilon == 0.0
            assert arm.seed == 42  # same-seed repeats: noise, not seed variance

    def test_repeats_share_group(self):
        arms = stage_a_arms()
        groups: dict[str, int] = {}
        for arm in arms:
            groups[arm.group] = groups.get(arm.group, 0) + 1
        assert sorted(groups.values()) == [2, 2, 2, 2, 2, 2, 3]
        for arm in arms:
            assert oc.repeat_group(arm.run_id) == arm.group


class TestStageBMatrix:

    EPS = [0.0625, 0.2, 0.45]

    def test_twentyfive_runs_exact_composition(self):
        arms = stage_b_arms(self.EPS)
        assert len(arms) == 25
        slip = [a for a in arms if a.group.split("_")[1] == "slip"]
        aging = [a for a in arms if a.aging_mode != "none"]
        spot = [a for a in arms if "_spot_" in a.run_id]
        cyclic = [a for a in arms if a.assignment == "cyclic"]
        assert len(slip) == 18      # 3 slippage x 3 eps x 2 repeats
        assert len(aging) == 4      # additive_capped x2 + stochastic_tail x2
        assert len(spot) == 2
        assert len(cyclic) == 1

    def test_slippage_arms_cover_all_eps_and_policies(self):
        arms = [a for a in stage_b_arms(self.EPS) if "_slip_" in a.run_id]
        combos = {(a.slippage, a.epsilon) for a in arms}
        assert combos == {
            (s, e)
            for s in ("drop", "recycle_last_delta", "renormalize")
            for e in self.EPS
        }

    def test_slippage_aliases_accepted(self):
        arms = stage_b_arms(self.EPS, slippage_policies=["recycle", "renorm"])
        slips = {a.slippage for a in arms if "_slip_" in a.run_id}
        assert slips == {"recycle_last_delta", "renormalize"}

    def test_starvation_controls_at_eps_high(self):
        arms = [a for a in stage_b_arms(self.EPS) if a.aging_mode != "none"]
        assert all(a.epsilon == 0.45 for a in arms)
        additive = [a for a in arms if a.aging_mode == "additive_capped"]
        assert all(a.aging_lambda > 0 and a.aging_tau_max > 0 for a in additive)

    def test_cut_lines_sort_last(self):
        # Gate §6: first cut cyclic, then the spot-check -> they must be the
        # final run_ids in lexicographic queue order.
        run_ids = sorted(a.run_id for a in stage_b_arms(self.EPS))
        assert "cyclic" in run_ids[-1]
        assert all("_spot_" in rid for rid in run_ids[-3:-1])

    def test_invalid_eps_rejected(self):
        with pytest.raises(ValueError):
            stage_b_arms([])
        with pytest.raises(ValueError):
            stage_b_arms([0.0, 0.2])


class TestStageCMatrix:

    def _arms(self):
        return stage_c_arms(eps_low=0.0625, eps_high=0.45)

    def test_sixteen_runs_exact_composition(self):
        arms = self._arms()
        assert len(arms) == 16
        tags = {}
        for arm in arms:
            tag = arm.group.split("_", 1)[1]
            tags[tag] = tags.get(tag, 0) + 1
        assert tags == {
            "mono": 2,
            "eps0bug": 2,
            "best_e0p0625": 3,
            "best_e0p45": 3,
            "dropctl_e0p45": 2,
            "warmup_e0p45": 2,
            "cyclic_e0p45": 2,
        }

    def test_accuracy_parameters(self):
        for arm in self._arms():
            assert arm.rounds == 10

    def test_bug_detector_is_eps_zero_per_layer(self):
        arms = [a for a in self._arms() if "eps0bug" in a.run_id]
        assert all(a.epsilon == 0.0 and a.mode == "per_layer" for a in arms)
        assert {a.seed for a in arms} == {42, 43}

    def test_best_triplet_uses_three_seeds(self):
        arms = [a for a in self._arms() if "best_e0p45" in a.run_id]
        assert {a.seed for a in arms} == {42, 43, 44}

    def test_warmup_arm_sets_schedule(self):
        arms = [a for a in self._arms() if "warmup" in a.run_id]
        assert all(a.epsilon_warmup_rounds == 3 and a.epsilon == 0.45
                   for a in arms)

    def test_cyclic_sorts_last(self):
        run_ids = sorted(a.run_id for a in self._arms())
        assert all("cyclic" in rid for rid in run_ids[-2:])


# ---------------------------------------------------------------------------
# Config emission + determinism
# ---------------------------------------------------------------------------

class TestConfigGeneration:

    def test_per_layer_config_knobs(self):
        arm = next(a for a in stage_a_arms()
                   if a.assignment == "coverage_eft" and a.metric == "relative")
        cfg = build_config(arm)
        training = cfg["training"]
        assert training["update_mode"] == "per_layer"
        assert training["importance_metric_v2"] == "relative"
        assert training["assignment_strategy"] == "coverage_eft"
        assert training["epsilon_deadline"] == 0.0
        assert training["total_rounds"] == 3
        assert training["watchdog_factor"] == 3.0
        assert cfg["monitoring"]["report_output"] == "./results/"
        # Template topology preserved: 3 classes at 6/3/1 Mbps.
        assert cfg["traffic_classes"]["num_classes"] == 3

    def test_monolithic_config_from_mono_template(self):
        arm = next(a for a in stage_a_arms() if a.mode == "monolithic")
        cfg = build_config(arm)
        assert cfg["training"]["update_mode"] == "monolithic"
        assert cfg["traffic_classes"]["num_classes"] == 1

    def test_seed_propagates_to_partition(self):
        arm = next(a for a in stage_c_arms(0.0625, 0.45) if a.seed == 44)
        cfg = build_config(arm)
        assert cfg["training"]["dataset"]["partition"]["seed"] == 44

    def test_stage_b_configs_validate(self):
        # build_config runs ExperimentConfig.model_validate internally; a
        # bad knob combination must raise here, not at container start.
        for arm in stage_b_arms([0.0625, 0.2, 0.45]):
            build_config(arm)

    def test_generation_is_deterministic(self, tmp_path):
        arms = stage_a_arms()
        outs = []
        for sub in ("one", "two"):
            queue = tmp_path / sub / "queue"
            configs = tmp_path / sub / "configs"
            queue.mkdir(parents=True)
            write_stage(arms, queue, configs, Path("results/overnight/runs"),
                        timeout_s=1500, repo_root=tmp_path / sub)
            contents = {}
            for path in sorted((tmp_path / sub).rglob("*")):
                if path.is_file():
                    contents[path.relative_to(tmp_path / sub)] = path.read_bytes()
            outs.append(contents)
        assert outs[0] == outs[1]
        assert len([p for p in outs[0] if p.suffix == ".json"]) == 15

    def test_regeneration_identical_is_noop_and_drift_rejected(self, tmp_path):
        arms = stage_a_arms()[:2]
        queue = tmp_path / "queue"
        queue.mkdir()
        kwargs = dict(
            queue_dir=queue, configs_dir=tmp_path / "configs",
            output_root=Path("runs"), timeout_s=1500, repo_root=tmp_path,
        )
        write_stage(arms, **kwargs)
        write_stage(arms, **kwargs)  # identical content -> no error
        with pytest.raises(FileExistsError):
            write_stage(arms, **{**kwargs, "timeout_s": 99})
        write_stage(arms, **{**kwargs, "timeout_s": 99}, force=True)

    def test_spec_contents(self, tmp_path):
        arms = stage_b_arms([0.2], repeats=1)
        queue = tmp_path / "queue"
        queue.mkdir()
        specs = write_stage(arms, queue, tmp_path / "configs",
                            Path("runs"), timeout_s=777, repo_root=tmp_path)
        for spec in specs:
            for key in ("run_id", "stage", "config_path", "output_dir",
                        "timeout_s", "seed", "group", "arm"):
                assert key in spec
            assert spec["timeout_s"] == 777
            assert spec["stage"] == "B"
            assert (tmp_path / spec["config_path"]).exists()
            on_disk = json.loads(
                (queue / f"{spec['run_id']}.json").read_text(encoding="utf-8")
            )
            assert on_disk == spec


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunnerQueue:

    def test_load_specs_sorted_and_tolerant(self, tmp_path):
        queue = tmp_path / "queue"
        queue.mkdir()
        _write_spec(queue, "b_run", tmp_path / "out_b")
        _write_spec(queue, "a_run", tmp_path / "out_a")
        (queue / "broken.json").write_text("{not json", encoding="utf-8")
        (queue / "incomplete.json").write_text('{"run_id": "x"}',
                                               encoding="utf-8")
        (queue / "status.jsonl").write_text("", encoding="utf-8")
        specs = runner.load_specs(queue, repo_root=tmp_path)
        assert [s.run_id for s in specs] == ["a_run", "b_run"]

    def test_pending_skips_completed_attempted_and_other_stages(self, tmp_path):
        queue = tmp_path / "queue"
        queue.mkdir()
        _write_spec(queue, "r1", tmp_path / "o1", stage="A")
        _write_spec(queue, "r2", tmp_path / "o2", stage="A")
        _write_spec(queue, "r3", tmp_path / "o3", stage="B")
        _mark_complete(tmp_path / "o1")
        specs = runner.load_specs(queue, repo_root=tmp_path)
        pending = runner.pending_specs(specs, stage="A", attempted=set())
        assert [s.run_id for s in pending] == ["r2"]
        pending = runner.pending_specs(specs, stage=None, attempted={"r2"})
        assert [s.run_id for s in pending] == ["r3"]

    def test_relative_paths_resolve_against_repo_root(self, tmp_path):
        queue = tmp_path / "queue"
        queue.mkdir()
        spec = {
            "run_id": "rel", "stage": "A", "config_path": "cfg/x.yaml",
            "output_dir": "runs/rel", "timeout_s": 5, "seed": 1,
        }
        (queue / "rel.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = runner.load_specs(queue, repo_root=tmp_path)[0]
        assert loaded.output_dir == tmp_path / "runs" / "rel"
        assert loaded.config_path == tmp_path / "cfg" / "x.yaml"


class TestRunnerExecution:

    def _spec(self, tmp_path, timeout_s=30.0):
        queue = tmp_path / "queue"
        queue.mkdir(exist_ok=True)
        spec_path = _write_spec(queue, "exec_run", tmp_path / "out",
                                timeout_s=timeout_s)
        return runner.load_specs(queue, repo_root=tmp_path)[0], queue

    def test_ok_requires_zero_exit_and_report(self, tmp_path):
        spec, _ = self._spec(tmp_path)
        report = spec.report_path
        cmd = [
            sys.executable, "-c",
            "import pathlib,sys; p = pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')",
            str(report),
        ]
        cleanups: list = []
        status, rc = runner.execute_spec(
            spec, repo_root=tmp_path, command=cmd,
            cleanup_fn=lambda n, c: cleanups.append((n, c)),
        )
        assert (status, rc) == ("ok", 0)
        assert len(cleanups) == 1  # pre-run clean slate
        assert (spec.output_dir / "run.log").exists()

    def test_zero_exit_without_report_is_error(self, tmp_path):
        spec, _ = self._spec(tmp_path)
        status, rc = runner.execute_spec(
            spec, repo_root=tmp_path,
            command=[sys.executable, "-c", "pass"],
            cleanup_fn=lambda n, c: None,
        )
        assert (status, rc) == ("error", 0)

    def test_nonzero_exit_is_error(self, tmp_path):
        spec, _ = self._spec(tmp_path)
        status, rc = runner.execute_spec(
            spec, repo_root=tmp_path,
            command=[sys.executable, "-c", "import sys; sys.exit(3)"],
            cleanup_fn=lambda n, c: None,
        )
        assert (status, rc) == ("error", 3)

    def test_timeout_kills_and_cleans_up(self, tmp_path):
        spec, _ = self._spec(tmp_path, timeout_s=0.4)
        cleanups: list = []
        status, rc = runner.execute_spec(
            spec, repo_root=tmp_path, grace_s=0.2,
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            cleanup_fn=lambda n, c: cleanups.append((n, c)),
        )
        assert (status, rc) == ("timeout", None)
        assert len(cleanups) == 2  # pre-run + post-kill

    def test_run_queue_drains_and_logs_status(self, tmp_path):
        queue = tmp_path / "queue"
        queue.mkdir()
        _write_spec(queue, "q1", tmp_path / "o1", timeout_s=30)
        _write_spec(queue, "q2", tmp_path / "o2", timeout_s=30)
        _mark_complete(tmp_path / "o2")  # pre-completed -> skipped

        def command_builder(spec):
            return [
                sys.executable, "-c",
                "import pathlib,sys; p = pathlib.Path(sys.argv[1]); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text('{}')",
                str(spec.report_path),
            ]

        counts = runner.run_queue(
            queue, repo_root=tmp_path,
            command_builder=command_builder,
            cleanup_fn=lambda n, c: None,
        )
        assert counts == {"ok": 1, "skipped": 1}
        lines = [
            json.loads(line)
            for line in (queue / "status.jsonl").read_text().splitlines()
        ]
        by_run = {rec["run_id"]: rec for rec in lines}
        assert by_run["q2"]["status"] == "skipped"
        assert by_run["q1"]["status"] == "ok"
        for key in ("run_id", "started", "ended", "status"):
            assert key in by_run["q1"]
        # Idempotency: a second invocation skips everything.
        counts = runner.run_queue(
            queue, repo_root=tmp_path,
            command_builder=command_builder, cleanup_fn=lambda n, c: None,
        )
        assert counts == {"skipped": 2}

    def test_failed_run_not_retried_within_invocation(self, tmp_path):
        queue = tmp_path / "queue"
        queue.mkdir()
        _write_spec(queue, "f1", tmp_path / "o1", timeout_s=30)
        calls = {"n": 0}

        def command_builder(spec):
            calls["n"] += 1
            return [sys.executable, "-c", "import sys; sys.exit(1)"]

        counts = runner.run_queue(
            queue, repo_root=tmp_path,
            command_builder=command_builder, cleanup_fn=lambda n, c: None,
        )
        assert counts == {"error": 1}
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Telemetry / manifest parsing
# ---------------------------------------------------------------------------

def _synthetic_report() -> dict:
    """Two-round report matching the collector layout + sidecar schema."""
    def lcm(importance, tc, n_bytes):
        return {"layer_name": None, "importance": importance,
                "traffic_class": tc, "bytes_sent": n_bytes}

    def worker_entry():
        metrics = []
        for name, imp, tc, b in (
            ("conv/kernel", 4.0, 0, 147_456),
            ("conv/bias", 1.0, 1, 326),
            ("head/kernel", 0.5, 2, 2_560),
        ):
            m = lcm(imp, tc, b)
            m["layer_name"] = name
            metrics.append(m)
        return {"val_accuracy": 0.5, "layer_comm_metrics": metrics}

    def agg_entry(t_eps):
        telemetry = {
            "node-1": {
                "manifest_arrival_rel": 0.10,
                "trigger_fire_rel": 0.10 + t_eps,
                "watchdog_fired": False,
                "t_eps_local": t_eps,
                "layer_arrivals_rel": {
                    "conv/kernel": 0.10 + t_eps,
                    "conv/bias": 0.15,
                    "head/kernel": 0.30,
                },
                "shed_layers": [],
                "kappa_realized": 0.0,
                "ordering_violation": False,
            }
        }
        return {
            "val_accuracy": 0.5,
            "uplink_telemetry_json": json.dumps(telemetry),
        }

    return {
        "experiment": {"algorithm": "fedavg"},
        "per_round": [
            {"round": r, "nodes": {"node-0": agg_entry(1.5 + 0.1 * r),
                                   "node-1": worker_entry()}}
            for r in (0, 1)
        ],
        "summary": {},
    }


class TestReportParsing:

    def test_worker_manifests_skip_aggregator(self):
        report = _synthetic_report()
        manifests = oc.worker_manifests("run_a", report, aggregator="node-0")
        assert len(manifests) == 2
        m = manifests[0]
        assert m.source == "node-1"
        assert m.sizes["conv/kernel"] == 147_456
        assert m.classes["head/kernel"] == 2
        assert m.total_score == pytest.approx(5.5)

    def test_uplink_telemetry_json_string_form(self):
        report = _synthetic_report()
        obs = oc.uplink_observations("run_a", report, aggregator="node-0")
        assert len(obs) == 2
        # The synthetic sidecar uses the pre-NT-03 key spelling on purpose:
        # every campaign already on disk does, so the parser must map it onto
        # the domain-tagged fields.
        assert obs[0].t_eps_local_receiver_s == pytest.approx(1.5)
        assert obs[0].manifest_arrival_rel_receiver_s == pytest.approx(0.10)
        assert not obs[0].watchdog_fired

    def test_cost_observations_join(self):
        report = _synthetic_report()
        manifests = oc.worker_manifests("run_a", report, "node-0")
        telemetry = oc.uplink_observations("run_a", report, "node-0")
        observations = oc.collect_cost_observations(manifests, telemetry)
        # Class 0: one layer arriving t_eps after the manifest.
        n, b, t = observations[0][0]
        assert (n, b) == (1, 147_456)
        assert t == pytest.approx(1.5)
        assert set(observations) == {0, 1, 2}

    def test_empty_sidecar_tolerated(self):
        entry = {"uplink_telemetry_json": ""}
        assert oc.parse_uplink_telemetry(entry) is None
        assert oc.parse_uplink_telemetry({}) is None
        # Already-decoded dict form (alternative collector storage).
        assert oc.parse_uplink_telemetry(
            {"uplink_telemetry": {"node-1": {}}}
        ) == {"node-1": {}}


class TestAffineFit:

    def test_recovers_known_parameters(self):
        alpha, rate = 0.05, 750_000.0
        obs = [
            (n, b, alpha * n + b / rate)
            for n, b in [(1, 100_000), (3, 400_000), (5, 750_000),
                         (2, 50_000), (4, 900_000)]
        ]
        costs = oc.fit_affine_class_costs({0: obs}, {0: 6.0})
        assert costs[0].source == "fit"
        assert costs[0].alpha_s == pytest.approx(alpha, rel=1e-6)
        assert costs[0].bytes_per_s == pytest.approx(rate, rel=1e-6)

    def test_single_observation_uses_rate_only_fit(self):
        costs = oc.fit_affine_class_costs(
            {0: [(1, 750_000, 1.0)]}, {0: 6.0}
        )
        assert costs[0].source == "fit"
        assert costs[0].alpha_s == 0.0
        assert costs[0].bytes_per_s == pytest.approx(750_000.0)

    def test_no_observations_fall_back_to_nominal(self):
        costs = oc.fit_affine_class_costs({}, {0: 6.0, 1: 3.0, 2: 1.0})
        assert all(c.source == "nominal" for c in costs.values())
        assert costs[2].bytes_per_s == pytest.approx(125_000.0)
        assert costs[2].mbps == pytest.approx(1.0)

    def test_eft_makespan_balances_by_finish_time(self):
        costs = {
            0: oc.ClassCost(0.0, 100.0, "fit"),
            1: oc.ClassCost(0.0, 50.0, "fit"),
        }
        makespan, per_class = oc.eft_makespan([1000, 100, 50], costs)
        assert makespan == pytest.approx(10.0)  # big job pins the fast class
        assert per_class[1] == (2, 150)

    def test_fluid_bound_respects_layer_atomicity(self):
        costs = {
            0: oc.ClassCost(0.0, 100.0, "fit"),
            1: oc.ClassCost(0.0, 50.0, "fit"),
        }
        # Divisible-load bound would be 1150/150 ~ 7.67; the atomic 1000-byte
        # layer cannot beat 10s on the fastest class.
        assert oc.fluid_lower_bound([1000, 100, 50], costs) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Knee selection
# ---------------------------------------------------------------------------

class TestKneeSelection:

    def test_eps_grid_inclusive(self):
        grid = build_eps_grid(0.0, 0.6, 0.01)
        assert len(grid) == 61
        assert grid[0] == 0.0 and grid[-1] == 0.6

    def test_curves_step_where_layers_cross(self):
        curves = compute_curves(
            [_manifest()], DensityPrefixStrategy(), _ONE_MBPS,
            build_eps_grid(0.0, 0.6, 0.01),
        )
        eps = curves["eps"]
        beta = dict(zip(eps, curves["beta_mean"]))
        assert beta[0.0] == 0.0
        assert beta[0.09] == 0.0
        assert beta[0.1] == pytest.approx(800_000 / 901_000)
        assert beta[0.2] == pytest.approx(900_000 / 901_000)
        # Predicted t_eps drops with the shed bytes (affine, 1 Mbps).
        t = dict(zip(eps, curves["t_pred_mean"]))
        assert t[0.0] == pytest.approx(901_000 / 125_000)
        assert t[0.1] == pytest.approx(101_000 / 125_000)

    def test_knees_plus_anchor(self):
        grid = build_eps_grid(0.0, 0.6, 0.01)
        curves = compute_curves(
            [_manifest()], DensityPrefixStrategy(), _ONE_MBPS, grid
        )
        result = find_knees(grid, curves["beta_mean"])
        assert result["fallback_used"] is False
        assert result["chosen_eps"] == [0.0625, 0.1, 0.2]
        # Largest marginal byte saving first.
        assert result["knee_candidates"][0]["eps"] == pytest.approx(0.1)

    def test_flat_curve_falls_back(self):
        grid = build_eps_grid(0.0, 0.6, 0.01)
        result = find_knees(grid, [0.0] * len(grid))
        assert result["fallback_used"] is True
        assert result["chosen_eps"] == [0.0625, 0.2, 0.5]

    def test_cap_applies(self):
        grid = [0.0, 0.55, 0.6]
        beta = [0.0, 0.9, 0.9]  # single knee above the cap
        result = find_knees(grid, beta, cap=0.5)
        assert result["chosen_eps"] == [0.0625, 0.5]

    def test_noise_floor_groups_repeats(self):
        per_run = {
            "a01_coveft_delta_r1": [1.0, 1.2],
            "a01_coveft_delta_r2": [1.4, 1.4],   # mean 1.4 vs 1.1
            "a07_mono_r1": [2.0],
            "a07_mono_r2": [2.0],                # zero variance group
            "a05_lonely_r1": [9.9],              # no repeat -> ignored
        }
        result = compute_noise_floor(per_run)
        groups = result["per_group"]
        assert set(groups) == {"a01_coveft_delta", "a07_mono"}
        # Sample stdev of the per-run means [1.1, 1.4] = 0.3/sqrt(2).
        assert groups["a01_coveft_delta"]["stdev_t_eps_s"] == pytest.approx(
            0.3 / (2 ** 0.5)
        )
        # Pooled floor = RMS over group stdevs (second group has zero var).
        assert result["noise_floor"] == pytest.approx(
            ((0.3 / (2 ** 0.5)) ** 2 / 2) ** 0.5
        )

    def test_no_telemetry_yields_none(self):
        assert compute_noise_floor({"a_r1": [], "a_r2": []})["noise_floor"] is None

    def test_real_coverage_eft_code_path_if_integrated(self):
        """Selector-consistency smoke test against the deployed strategy."""
        try:
            strategy = oc.load_assignment_strategy("coverage_eft")
        except ValueError:
            pytest.skip("coverage_eft strategy not integrated yet")
        curves = compute_curves(
            [_manifest()], strategy, _ONE_MBPS, [0.0, 0.2, 0.45]
        )
        beta = curves["beta_mean"]
        assert beta[0] == 0.0  # nothing sheddable at eps=0 (positive scores)
        assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(beta, beta[1:]))


# ---------------------------------------------------------------------------
# Offline OPT gap
# ---------------------------------------------------------------------------

class TestEnumerateOpt:

    COSTS = {
        0: oc.ClassCost(0.0, 100.0, "fit"),
        1: oc.ClassCost(0.0, 50.0, "fit"),
    }
    SCORES = {"a": 5.0, "b": 3.0, "c": 2.0}
    SIZES = {"a": 100, "b": 50, "c": 1000}

    def test_eps_zero_keeps_everything(self):
        opt = enumerate_opt(self.SCORES, self.SIZES, 0.0, self.COSTS)
        assert opt["beta_opt"] == 0.0
        assert opt["n_feasible_tails"] == 1  # only the empty tail
        assert opt["t_opt_eft"] == pytest.approx(10.0)
        assert opt["t_fluid_lb"] == pytest.approx(10.0)

    def test_sheds_heavy_low_score_layer(self):
        opt = enumerate_opt(self.SCORES, self.SIZES, 0.2, self.COSTS)
        # Budget 2.0 admits tail {c}: 1000 bytes shed, head {a, b}.
        assert opt["beta_opt"] == pytest.approx(1000 / 1150)
        assert opt["t_opt_eft"] == pytest.approx(1.0)

    def test_must_receive_blocks_shedding(self):
        opt = enumerate_opt(
            self.SCORES, self.SIZES, 0.2, self.COSTS,
            must_receive=frozenset({"c"}),
        )
        assert opt["beta_opt"] == 0.0
        assert opt["t_opt_eft"] == pytest.approx(10.0)

    def test_fluid_never_exceeds_eft(self):
        for eps in (0.0, 0.1, 0.2, 0.5):
            opt = enumerate_opt(self.SCORES, self.SIZES, eps, self.COSTS)
            assert opt["t_fluid_lb"] <= opt["t_opt_eft"] + 1e-12

    def test_layer_bound_enforced(self):
        scores = {f"l{i}": 1.0 for i in range(21)}
        sizes = {f"l{i}": 10 for i in range(21)}
        with pytest.raises(ValueError, match="enumeration bound"):
            enumerate_opt(scores, sizes, 0.1, self.COSTS)


# ---------------------------------------------------------------------------
# Post-hoc cost
# ---------------------------------------------------------------------------

class TestPosthocCost:

    def test_ratio_kappa_geometric(self):
        assert ratio_kappa(10.0, 3) == pytest.approx([10.0, 10.0 ** 0.5, 1.0])
        assert ratio_kappa(1.0, 3) == pytest.approx([1.0, 1.0, 1.0])
        assert ratio_kappa(5.0, 1) == pytest.approx([5.0])

    def test_dollars_per_gb(self):
        bytes_by_class = {0: 1_000_000_000, 1: 2_000_000_000, 2: 4_000_000_000}
        kappa = [10.0, 2.0, 1.0]
        assert dollars(bytes_by_class, kappa) == pytest.approx(10 + 4 + 4)

    def test_dollars_prices_unknown_class_at_cheapest(self):
        assert dollars({-1: 1_000_000_000, 5: 1_000_000_000}, [10.0, 1.0]) \
            == pytest.approx(2.0)

    def test_sent_bytes_scope(self):
        report = _synthetic_report()
        all_bytes = sent_bytes_by_class(report, "node-0", scope="all")
        uplink = sent_bytes_by_class(report, "node-0", scope="uplink")
        assert uplink == {0: 2 * 147_456, 1: 2 * 326, 2: 2 * 2_560}
        assert all_bytes == uplink  # synthetic aggregator logs no downlink

    def test_cancelled_bytes_shapes(self):
        report = {
            "per_round": [{
                "round": 0,
                "nodes": {
                    "node-1": {"cancelled_bytes_by_class": {"2": 500}},
                    "node-2": {"cancelled_telemetry_json":
                               json.dumps({"1": 100})},
                    "node-3": {"cancelled_bytes": 42},
                },
            }],
        }
        assert cancelled_bytes_by_class(report) == {2: 500, 1: 100, -1: 42}
        assert cancelled_bytes_by_class({"per_round": []}) == {}


# ---------------------------------------------------------------------------
# Misc common helpers
# ---------------------------------------------------------------------------

class TestCommonHelpers:

    def test_repeat_group_strips_rep_and_seed_suffixes(self):
        assert oc.repeat_group("a01_coveft_delta_sq_norm_r2") == \
            "a01_coveft_delta_sq_norm"
        assert oc.repeat_group("c03_best_e0p0625_s43") == "c03_best_e0p0625"
        assert oc.repeat_group("no_suffix") == "no_suffix"

    def test_find_run_dirs_expands_parents(self, tmp_path):
        for name in ("r1", "r2"):
            _mark_complete(tmp_path / name)
        (tmp_path / "not_a_run").mkdir()
        runs = oc.find_run_dirs([tmp_path])
        assert [r.name for r in runs] == ["r1", "r2"]
        assert oc.find_run_dirs([tmp_path / "r1"]) == [tmp_path / "r1"]

    def test_find_aggregator_from_configs(self, tmp_path):
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "node-0.yaml").write_text(
            yaml.dump({"node_id": "node-0", "role": "aggregator"}),
            encoding="utf-8",
        )
        (cfg_dir / "node-1.yaml").write_text(
            yaml.dump({"node_id": "node-1", "role": "worker"}),
            encoding="utf-8",
        )
        assert oc.find_aggregator(tmp_path) == "node-0"
        assert oc.find_aggregator(tmp_path / "missing") == "node-0"

    def test_nominal_uplink_bandwidths(self, tmp_path):
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "node-0.yaml").write_text(
            yaml.dump({"node_id": "node-0", "role": "aggregator"}),
            encoding="utf-8",
        )
        (cfg_dir / "node-1.yaml").write_text(
            yaml.dump({
                "node_id": "node-1",
                "role": "worker",
                "outgoing_edges": [{
                    "src": 1, "dst": 0,
                    "classes": {
                        0: {"bandwidth_mbps": 6},
                        1: {"bandwidth_mbps": 3},
                        2: {"bandwidth_mbps": None},
                    },
                }],
            }),
            encoding="utf-8",
        )
        bw = oc.nominal_uplink_bandwidths(tmp_path)
        assert bw[0] == 6.0 and bw[1] == 3.0
        assert bw[2] == float("inf")
