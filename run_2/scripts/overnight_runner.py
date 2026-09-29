"""Idempotent sequential queue runner for the overnight evaluation.

Consumes run-spec JSON files from a queue directory (sorted by filename) and
executes each as ``python -m scripts.run --config <...> --output-dir <...>``
with a HARD per-run timeout — the runner-level belt-and-braces watchdog
required by gate ruling G4 on top of the in-engine coverage watchdog.

Design (design doc §4.3, "dynamic queue"):

- A spec is *complete* iff ``<output_dir>/results/report.json`` exists; the
  runner skips complete specs, so it can be killed and restarted at any time
  and will resume exactly where it left off.  Failed/timed-out runs leave no
  report and are therefore retried on the next runner invocation (but not
  within the same invocation — a deterministic single pass per spec keeps a
  crashing config from looping all night).
- The queue is re-scanned after every run, so the supervisor may append new
  spec files mid-stage (e.g. an extra ε flagged by the β-curve) and they are
  picked up without restarting the runner.  The runner exits when a scan
  finds nothing left to do.
- One experiment at a time: container names (node-0..N-1, monitor) and the
  emulated subnet are fixed, so before every run — and after every kill —
  the runner force-removes those containers and any leftover ``fl-net``
  compose network (two stacks would collide on the 10.0.0.0/24 pool).
- Per-run status lines ``{run_id, started, ended, status, ...}`` are
  appended to ``<queue>/status.jsonl``; container/launcher output is
  redirected to ``<output_dir>/run.log``.

Spec schema (extra keys are preserved but ignored here)::

    {
      "run_id":      "a01_coveft_delta_sq_norm_r1",
      "stage":       "A",
      "config_path": "configs/experiments/overnight/stage_a/<run_id>.yaml",
      "output_dir":  "results/overnight/runs/<run_id>",
      "timeout_s":   1500,
      "seed":        42
    }

Relative paths are resolved against the repository root, which is also the
subprocess working directory (``python -m scripts.run`` needs it).

Usage::

    source .venv/bin/activate
    python -m scripts.overnight_runner --queue-dir results/overnight/queue
    python -m scripts.overnight_runner --queue-dir results/overnight/queue --stage B
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_SPEC_KEYS = (
    "run_id",
    "stage",
    "config_path",
    "output_dir",
    "timeout_s",
    "seed",
)

#: Statuses that count as a failure for the process exit code.
_FAILURE_STATUSES = frozenset({"timeout", "error"})


@dataclass(frozen=True)
class RunSpec:
    """One queued run, as loaded from a spec JSON file."""

    run_id: str
    stage: str
    config_path: Path
    output_dir: Path
    timeout_s: float
    seed: int
    spec_path: Path

    @property
    def report_path(self) -> Path:
        return self.output_dir / "results" / "report.json"

    @property
    def compose_path(self) -> Path:
        return self.output_dir / "docker-compose.yml"


def _resolve(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (repo_root / path)


def load_specs(queue_dir: Path, repo_root: Path = REPO_ROOT) -> list[RunSpec]:
    """Load all valid spec files from the queue dir, sorted by filename.

    Malformed specs are warned about and skipped rather than aborting the
    queue — an unattended runner must never die on one bad file.
    """
    specs: list[RunSpec] = []
    for spec_path in sorted(queue_dir.glob("*.json")):
        try:
            with spec_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[runner] WARNING: skipping unreadable spec {spec_path.name}: {exc}")
            continue
        missing = [k for k in _REQUIRED_SPEC_KEYS if k not in raw]
        if missing:
            print(
                f"[runner] WARNING: skipping {spec_path.name}: "
                f"missing keys {missing}"
            )
            continue
        specs.append(
            RunSpec(
                run_id=str(raw["run_id"]),
                stage=str(raw["stage"]),
                config_path=_resolve(str(raw["config_path"]), repo_root),
                output_dir=_resolve(str(raw["output_dir"]), repo_root),
                timeout_s=float(raw["timeout_s"]),
                seed=int(raw["seed"]),
                spec_path=spec_path,
            )
        )
    return specs


def pending_specs(
    specs: list[RunSpec],
    stage: str | None,
    attempted: set[str],
) -> list[RunSpec]:
    """Specs still to run: stage-matched, not complete, not attempted yet."""
    out = []
    for spec in specs:
        if stage is not None and spec.stage.upper() != stage.upper():
            continue
        if spec.run_id in attempted:
            continue
        if spec.report_path.exists():
            continue
        out.append(spec)
    return out


# ---------------------------------------------------------------------------
# Docker cleanup
# ---------------------------------------------------------------------------

def _run_quiet(cmd: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def docker_cleanup(max_nodes: int, compose_path: Path | None = None) -> None:
    """Force-remove the fixed-name containers and leftover fl-net networks.

    Called before every run (clean slate regardless of how the previous run
    died) and after every kill.  ``docker compose down`` only works while
    the compose file exists, so ``docker rm -f`` on the well-known names is
    the primary mechanism and compose-down the network sweeper.
    """
    names = [f"node-{i}" for i in range(max_nodes)] + ["monitor"]
    _run_quiet(["docker", "rm", "-f", *names])

    if compose_path is not None and compose_path.exists():
        _run_quiet(
            [
                "docker", "compose", "-f", str(compose_path),
                "down", "-v", "--remove-orphans", "-t", "5",
            ]
        )

    # Compose networks are named <project>_fl-net with project = the output
    # dir basename, so a timed-out run leaves a network the *next* run's
    # project name will not match — but its 10.0.0.0/24 subnet still
    # collides.  Sweep anything fl-net-ish; removal fails harmlessly if a
    # container still uses it.
    listing = _run_quiet(["docker", "network", "ls", "--format", "{{.Name}}"])
    if listing is not None and listing.stdout:
        for net in listing.stdout.split():
            if "fl-net" in net:
                _run_quiet(["docker", "network", "rm", net], timeout=30.0)


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def _kill_process_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the whole process group, escalate to SIGKILL after a grace."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=max(wait_s, 0.1))
            return
        except subprocess.TimeoutExpired:
            continue


def build_command(spec: RunSpec) -> list[str]:
    """The exact launcher invocation for one spec."""
    return [
        sys.executable,
        "-m",
        "scripts.run",
        "--config",
        str(spec.config_path),
        "--output-dir",
        str(spec.output_dir),
    ]


def execute_spec(
    spec: RunSpec,
    repo_root: Path = REPO_ROOT,
    grace_s: float = 10.0,
    max_nodes: int = 4,
    command: list[str] | None = None,
    cleanup_fn: Callable[[int, Path | None], None] = docker_cleanup,
) -> tuple[str, int | None]:
    """Run one spec to completion, timeout, or error.

    Returns ``(status, returncode)`` with status in {ok, timeout, error}.
    ``ok`` requires both a zero exit code AND the report file — the launcher
    can exit zero after a degraded run, and the report is the artifact every
    downstream script needs, so its existence is the success criterion.

    ``command`` and ``cleanup_fn`` are injectable for tests; the defaults
    run the real launcher and the real docker cleanup.
    """
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_fn(max_nodes, spec.compose_path)

    cmd = command if command is not None else build_command(spec)
    log_path = spec.output_dir / "run.log"
    with log_path.open("ab") as log_fh:
        log_fh.write(
            f"\n===== runner: {spec.run_id} cmd={' '.join(cmd)} =====\n".encode()
        )
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group -> killable as a tree
        )
        try:
            returncode = proc.wait(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc, grace_s)
            cleanup_fn(max_nodes, spec.compose_path)
            return "timeout", None

    if returncode == 0 and spec.report_path.exists():
        return "ok", returncode
    return "error", returncode


def append_status(queue_dir: Path, record: dict) -> None:
    """Append one JSON line to ``<queue>/status.jsonl`` (append-only log)."""
    status_path = queue_dir / "status.jsonl"
    with status_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_queue(
    queue_dir: Path,
    stage: str | None = None,
    repo_root: Path = REPO_ROOT,
    grace_s: float = 10.0,
    max_nodes: int = 4,
    dry_run: bool = False,
    command_builder: Callable[[RunSpec], list[str]] | None = None,
    cleanup_fn: Callable[[int, Path | None], None] = docker_cleanup,
) -> dict[str, int]:
    """Drain the queue; returns counts per status (incl. ``skipped``)."""
    counts: dict[str, int] = {}
    attempted: set[str] = set()
    skipped_logged: set[str] = set()

    all_specs = load_specs(queue_dir, repo_root)
    stage_specs = [
        s for s in all_specs
        if stage is None or s.stage.upper() == stage.upper()
    ]
    print(
        f"[runner] queue={queue_dir} stage={stage or 'ALL'}: "
        f"{len(stage_specs)} spec(s) on disk"
    )

    # Log completed-on-arrival specs once, so status.jsonl tells the whole
    # story of this invocation (restart visibility for the supervisor).
    for spec in stage_specs:
        if spec.report_path.exists() and spec.run_id not in skipped_logged:
            skipped_logged.add(spec.run_id)
            counts["skipped"] = counts.get("skipped", 0) + 1
            print(f"[runner] skip {spec.run_id} (report exists)")
            if not dry_run:
                append_status(
                    queue_dir,
                    {
                        "run_id": spec.run_id,
                        "stage": spec.stage,
                        "started": None,
                        "ended": _now_iso(),
                        "status": "skipped",
                    },
                )

    while True:
        specs = load_specs(queue_dir, repo_root)  # re-scan: dynamic queue
        pending = pending_specs(specs, stage, attempted | skipped_logged)
        if not pending:
            break

        spec = pending[0]
        attempted.add(spec.run_id)
        position = len(attempted) + len(skipped_logged)
        total = len(
            [s for s in specs if stage is None or s.stage.upper() == stage.upper()]
        )
        if dry_run:
            print(f"[runner] [{position}/{total}] would run {spec.run_id} "
                  f"(timeout {spec.timeout_s:.0f}s)")
            counts["dry_run"] = counts.get("dry_run", 0) + 1
            continue

        print(
            f"[runner] [{position}/{total}] {spec.run_id} starting "
            f"(timeout {spec.timeout_s:.0f}s, log {spec.output_dir / 'run.log'})",
            flush=True,
        )
        started = _now_iso()
        t0 = time.monotonic()
        status, returncode = execute_spec(
            spec,
            repo_root=repo_root,
            grace_s=grace_s,
            max_nodes=max_nodes,
            command=command_builder(spec) if command_builder else None,
            cleanup_fn=cleanup_fn,
        )
        duration = time.monotonic() - t0
        counts[status] = counts.get(status, 0) + 1
        append_status(
            queue_dir,
            {
                "run_id": spec.run_id,
                "stage": spec.stage,
                "started": started,
                "ended": _now_iso(),
                "status": status,
                "returncode": returncode,
                "duration_s": round(duration, 1),
                "output_dir": str(spec.output_dir),
            },
        )
        print(
            f"[runner] [{position}/{total}] {spec.run_id} -> {status} "
            f"({duration:.1f}s)",
            flush=True,
        )

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do"
    print(f"[runner] queue empty — {summary}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Idempotent sequential runner for overnight run-spec queues."
    )
    parser.add_argument(
        "--queue-dir",
        required=True,
        help="Directory containing run-spec *.json files (status.jsonl is "
        "appended here).",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Only execute specs whose 'stage' matches (e.g. A, B, C). "
        "Default: all stages, in filename order.",
    )
    parser.add_argument(
        "--grace-s",
        type=float,
        default=10.0,
        help="Seconds between SIGTERM and SIGKILL when a run times out.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=4,
        help="Highest node count across queued experiments; controls which "
        "node-i containers are force-removed on cleanup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run without executing anything.",
    )
    args = parser.parse_args(argv)

    queue_dir = _resolve(args.queue_dir, REPO_ROOT)
    if not queue_dir.is_dir():
        print(f"[runner] ERROR: queue dir not found: {queue_dir}")
        return 2

    counts = run_queue(
        queue_dir,
        stage=args.stage,
        grace_s=args.grace_s,
        max_nodes=args.max_nodes,
        dry_run=args.dry_run,
    )
    failures = sum(counts.get(s, 0) for s in _FAILURE_STATUSES)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
