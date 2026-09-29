"""Main launcher: parse config -> pre-partition dataset -> generate docker-compose -> launch.

This runs on the HOST machine (not inside containers).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.config.schema import ExperimentConfig, load_config
from src.config.generator import generate_docker_compose, generate_node_configs
from src.datasets.cifar10 import CIFAR10Dataset
from src.datasets.femnist import FEMNISTDataset
from src.datasets.mnist import MNISTDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [launcher] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset registry for host-side partitioning.
# NOTE: Must stay in sync with DATASET_REGISTRY in src/node.py.
DATASET_REGISTRY = {
    "mnist": MNISTDataset,
    "cifar10": CIFAR10Dataset,
    "femnist": FEMNISTDataset,
}


def _existing_partitions(data_dir: Path, num_nodes: int) -> list[Path] | None:
    """Complete, readable, zero-image-free partitions already on disk, or None.

    Reuse is not an optimisation here, it is the corruption fix.  Every
    confirmed instance of silent shard corruption came from CONCURRENT
    materialisation: two runs calling ``tf.keras.datasets.cifar10.load_data()``
    at once race on the keras cache, and a reader can observe pages of a
    re-extracting archive that are still zero-filled.  The result is several
    percent of ``x_train`` replaced by all-zero images while ``y_train`` stays
    byte-identical to the clean partition — so shard size, class balance and
    the label histogram all look perfect and the run trains on injected label
    noise.  It cost us 11 of 81 cells in ``fedluar_hh``.

    Materialising serially once and reusing removes the race rather than
    detecting it afterwards.  The zero-row check below means a corrupt shard is
    never silently adopted: it is regenerated instead.
    """
    paths = [data_dir / f"node-{i}.npz" for i in range(num_nodes)]
    if not all(p.exists() for p in paths):
        return None
    for p in paths:
        try:
            with np.load(p) as d:
                for split in ("x_train", "x_val"):
                    x = d[split]
                    if len(x) == 0:
                        return None
                    if bool((x.reshape(len(x), -1) == 0).all(1).any()):
                        logger.warning(
                            "Discarding CORRUPT cached partition %s "
                            "(all-zero images present) — regenerating", p
                        )
                        return None
        except Exception as e:  # truncated / half-written / unreadable
            logger.warning("Discarding unreadable cached partition %s (%s) — "
                           "regenerating", p, e)
            return None
    return paths


def prepare_data(config: ExperimentConfig, output_dir: Path) -> Path:
    """Pre-partition the dataset on the host."""
    dataset_name = config.training.dataset.name
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cached = _existing_partitions(data_dir, config.federation.num_nodes)
    if cached is not None:
        logger.info(
            "Reusing %d pre-materialised partitions in %s "
            "(verified free of zero-image corruption)", len(cached), data_dir
        )
        return data_dir

    # HARD FAIL instead of regenerating, when the caller promised the
    # partitions were already materialised.
    #
    # Without this, a rejected shard reintroduces the exact hazard the reuse
    # guard exists to prevent, through the back door: the fall-through below
    # calls prepare_partitions -> tf.keras.datasets.cifar10.load_data(), and
    # inside a CONCURRENT batch that is the keras-cache race that injects
    # all-zero images in the first place. A run that quietly re-materialises
    # mid-campaign is the worst case — it looks like a normal run, it is not
    # gated by anything, and it can corrupt itself and its neighbours.
    #
    # So a runner that pre-materialised serially sets
    # FL_REQUIRE_PREMATERIALIZED=1 and gets a loud failure of ONE cell instead
    # of silent contamination of the batch. Rescue is then explicit: re-run
    # scripts/prematerialize.py serially and restart.
    if os.environ.get("FL_REQUIRE_PREMATERIALIZED", "").strip() not in ("", "0"):
        raise RuntimeError(
            f"FL_REQUIRE_PREMATERIALIZED is set but {data_dir} has no usable "
            "pre-materialised partition (missing, unreadable, or carrying "
            "all-zero images). Refusing to re-materialise: inside a concurrent "
            "batch that would race on the keras cache and is how silent "
            "zero-image corruption happens. Re-run "
            "`python -m scripts.prematerialize <specs> <campaign>` serially, "
            "then restart this run."
        )

    partition_config = config.training.dataset.partition
    kwargs = {}
    if partition_config.alpha is not None:
        kwargs["alpha"] = partition_config.alpha
    if partition_config.classes_per_node is not None:
        kwargs["classes_per_node"] = partition_config.classes_per_node
    if partition_config.max_writers is not None:
        kwargs["max_writers"] = partition_config.max_writers
    if partition_config.workers_only:
        kwargs["workers_only"] = True

    logger.info(
        f"Partitioning {dataset_name} for {config.federation.num_nodes} nodes "
        f"(strategy={partition_config.strategy}, seed={partition_config.seed})"
    )

    paths = dataset_cls.prepare_partitions(
        total_nodes=config.federation.num_nodes,
        output_dir=str(data_dir),
        partition_strategy=partition_config.strategy,
        seed=partition_config.seed,
        **kwargs,
    )

    logger.info(f"Dataset partitioned: {len(paths)} files in {data_dir}")
    return data_dir


def generate_configs(config: ExperimentConfig, output_dir: Path) -> tuple[Path, Path]:
    """Generate docker-compose.yml and per-node configs."""
    # Generate per-node configs
    node_configs = generate_node_configs(config)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for service_name, node_config in node_configs.items():
        config_path = configs_dir / f"{service_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(node_config, f, default_flow_style=False)
        logger.info(f"  Written: {config_path}")

    # Generate monitor config
    monitor_config = {
        "monitor_port": 5100,
        "tensorboard": config.monitoring.tensorboard,
        "tensorboard_port": config.monitoring.tensorboard_port,
        "tensorboard_logdir": "/logs/tensorboard",
        "report_output": "/results/report.json",
        "num_nodes": config.federation.num_nodes,
        "total_rounds": config.training.total_rounds,
        "training": {
            "algorithm": config.training.algorithm,
            "dataset": {"name": config.training.dataset.name},
            "model": config.training.model,
            "total_rounds": config.training.total_rounds,
        },
    }
    monitor_config_path = configs_dir / "monitor.yaml"
    with open(monitor_config_path, "w") as f:
        yaml.dump(monitor_config, f, default_flow_style=False)

    # Generate docker-compose.yml
    data_dir = output_dir / "data"
    compose = generate_docker_compose(config, str(data_dir))
    compose_path = output_dir / "docker-compose.yml"
    with open(compose_path, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Written: {compose_path}")

    return compose_path, configs_dir


def build_images(project_dir: Path) -> None:
    """Build Docker images for node and monitor."""
    logger.info("Building Docker images...")

    subprocess.run(
        [
            "docker", "build",
            "-t", "fl-node:latest",
            "-f", str(project_dir / "docker" / "Dockerfile.node"),
            str(project_dir),
        ],
        check=True,
    )
    logger.info("  Built fl-node:latest")

    subprocess.run(
        [
            "docker", "build",
            "-t", "fl-monitor:latest",
            "-f", str(project_dir / "docker" / "Dockerfile.monitor"),
            str(project_dir),
        ],
        check=True,
    )
    logger.info("  Built fl-monitor:latest")


def _project_name(compose_path: Path) -> str:
    """Unique compose project name for this run directory.

    Compose's default project name is the compose file's directory
    basename — for campaign layouts (<campaign>/<arm>/seed<N>/) that is
    just "seed<N>", so two different arms at the same seed running
    concurrently collide on one project (shared network name; one run's
    cleanup kills the other's containers — observed 2026-08-05, CP2
    batch 1).  Use the last two path components instead
    (e.g. "sentinel_mono-seed41"), sanitized to compose's allowed
    charset.
    """
    parts = [p.name for p in [compose_path.parent.parent, compose_path.parent] if p.name]
    raw = "-".join(parts) or compose_path.parent.name or "fl-run"
    sanitized = re.sub(r"[^a-z0-9_-]", "-", raw.lower()).strip("-_") or "fl-run"
    return sanitized


def launch(compose_path: Path) -> None:
    """Launch the Docker Compose stack."""
    logger.info("Launching Docker Compose stack...")
    subprocess.run(
        [
            "docker", "compose",
            "-p", _project_name(compose_path),
            "-f", str(compose_path), "up",
        ],
        check=True,
    )


def cleanup(compose_path: Path) -> None:
    """Stop and remove containers."""
    logger.info("Cleaning up...")
    subprocess.run(
        [
            "docker", "compose",
            "-p", _project_name(compose_path),
            "-f", str(compose_path), "down", "-v",
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Emulator Launcher")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./run_output",
        help="Output directory for generated files (default: ./run_output)",
    )
    parser.add_argument(
        "--build", "-b",
        action="store_true",
        help="Force rebuild Docker images",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only generate configs without launching containers",
    )
    args = parser.parse_args()

    # Resolve paths
    project_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    logger.info(
        f"Config validated: {config.federation.num_nodes} nodes, "
        f"algorithm={config.training.algorithm}, "
        f"dataset={config.training.dataset.name}"
    )

    # 2. Pre-partition dataset
    data_dir = prepare_data(config, output_dir)

    # 3. Generate docker-compose + node configs
    compose_path, configs_dir = generate_configs(config, output_dir)

    if args.no_launch:
        logger.info(
            f"Generated files in {output_dir}. "
            f"Run again without --no-launch to launch containers."
        )
        return

    # 4. Build Docker images
    build_images(project_dir)

    # 5. Launch
    try:
        launch(compose_path)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cleanup(compose_path)

    # 6. Copy results
    results_dir = Path(config.monitoring.report_output)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results available in {output_dir}")


if __name__ == "__main__":
    main()
