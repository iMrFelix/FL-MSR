"""Monitor container entry point.

Runs the MetricsCollector and TensorBoard, then generates the final report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys

import yaml

from src.monitoring.collector import MetricsCollector
from src.monitoring.report_generator import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [monitor] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_monitor(config: dict) -> None:
    """Main monitor coroutine."""
    host = "0.0.0.0"
    port = config.get("monitor_port", 5100)
    logdir = config.get("tensorboard_logdir", "/logs/tensorboard")
    report_output = config.get("report_output", "/results/report.json")
    num_nodes = config.get("num_nodes", 4)
    total_rounds = config.get("total_rounds", 50)
    tb_port = config.get("tensorboard_port", 6006)
    training_config = config.get("training", {})

    # Start TensorBoard as a subprocess
    tb_process = None
    if config.get("tensorboard", True):
        logger.info(f"Starting TensorBoard on port {tb_port}, logdir={logdir}")
        tb_process = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", logdir,
                "--host", "0.0.0.0",
                "--port", str(tb_port),
                "--reload_interval", "5",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Start the metrics collector
    collector = MetricsCollector(
        host=host,
        port=port,
        logdir=logdir,
        num_nodes=num_nodes,
        total_rounds=total_rounds,
    )
    await collector.start()

    # Wait for all nodes to complete training
    timeout = total_rounds * 120.0  # Generous timeout
    completed = await collector.wait_for_completion(timeout=timeout)

    if completed:
        # Generate final report
        metrics = collector.get_all_metrics()
        generate_report(metrics, training_config, report_output)
    else:
        logger.warning("Monitor timed out before all nodes completed")
        # Still generate partial report
        metrics = collector.get_all_metrics()
        if metrics:
            generate_report(metrics, training_config, report_output)

    # Wait a bit for TensorBoard to be accessible after training
    await asyncio.sleep(5)

    # Cleanup
    await collector.stop()
    if tb_process:
        tb_process.terminate()
        tb_process.wait(timeout=5)

    logger.info("Monitor shutdown complete")


def main() -> None:
    """Entry point."""
    config_path = os.environ.get("MONITOR_CONFIG", "/config/monitor.yaml")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig: int) -> None:
        logger.info(f"Received signal {sig}")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal, sig)

    try:
        loop.run_until_complete(run_monitor(config))
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Monitor cancelled")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
