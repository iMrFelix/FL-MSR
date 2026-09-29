#!/usr/bin/env python3
"""GPU probe payload v2 — runs INSIDE the fl-node-gpu container.

    python /probe.py <mode>

Modes:
  gpus         list visible GPUs (EXPECT_GPUS env, default 1)
  free-once    one training run, determinism OFF  -> timing baseline
  det-once     one training run, op-determinism enabled BEFORE any op
               -> emits a sha256 of the final weights; the orchestrator
               runs this TWICE IN SEPARATE CONTAINERS and compares the
               digests: cross-container repeatability, which is what the
               framework's canary methodology actually needs (a
               within-process comparison would answer a weaker question)
  bench-resnet / bench-transformer   sec/epoch (warm-up epoch discarded)
  quick        tiny run for the concurrent all-GPUs check

DATA: loaded from the read-only mount PROBE_DATA (default
/probe_data.npz), exported ONCE, serially, by the orchestrator on the
host. keras.datasets is deliberately NOT used in-container: Keras 3
re-extracts the archive on every load_data() call, so concurrent
containers sharing a cache race exactly like the corruption bug this
project spent August eliminating (review finding M1, reproduced), and a
root container would chown a mounted host cache (M2). Every load also
counts all-zero rows as defence in depth.

XLA is disabled (jit_compile=False) in all modes so the determinism
comparison and the timings are not confounded by Keras' GPU XLA
auto-enabling (review finding M4). Numbers are for fit/graph mode
without XLA; the framework's eager loop will be somewhat slower —
treat bench numbers as optimistic bounds for sizing, and treat the
determinism verdict as an op-level screen: the framework-exact answer
comes from the framework's own canary on GPU in step 3.

Every mode ends with one line:  RESULT_JSON: {...}
Env: SUBSET (det/quick subset; default 20000/2000), EPOCHS (default 2/1).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf


def result(**kw):
    print("RESULT_JSON: " + json.dumps(kw))


def load_data(n=None):
    path = os.environ.get("PROBE_DATA", "/probe_data.npz")
    if os.path.exists(path):
        with np.load(path) as d:
            x, y = d["x"], d["y"]
        src = path
    else:  # local CPU testing fallback only — never the shipped path
        (x, y), _ = tf.keras.datasets.cifar10.load_data()
        x = x.astype("float32") / 255.0
        src = "keras-fallback"
    if n:
        x, y = x[:n], y[:n]
    zero_rows = int((x.reshape(len(x), -1) == 0).all(1).sum())
    return x, y, {"data_src": src, "n": len(x), "zero_rows": zero_rows}


def small_cnn():
    L = tf.keras.layers
    return tf.keras.Sequential([
        tf.keras.Input((32, 32, 3)),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.MaxPool2D(),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.MaxPool2D(),
        L.Flatten(),
        L.Dense(10),
    ])


def compile_(m):
    m.compile("adam",
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              jit_compile=False)          # no XLA: see module docstring
    return m


def train(x, y, epochs, seed=42):
    tf.keras.utils.set_random_seed(seed)
    m = compile_(small_cnn())
    t0 = time.time()
    h = m.fit(x, y, batch_size=128, epochs=epochs, shuffle=True, verbose=0)
    secs = time.time() - t0
    digest = hashlib.sha256(
        b"".join(np.ascontiguousarray(w).tobytes()
                 for w in m.get_weights())).hexdigest()
    return digest, [float(v) for v in h.history["loss"]], secs


def mode_once(deterministic):
    if deterministic:
        # Enabled before any op. Precision (review N7): a late
        # enable_op_determinism() call DOES take effect; it is the
        # TF_DETERMINISTIC_OPS env var that is silently ignored once ops
        # have run. Enabling first keeps this unambiguous either way —
        # and note determinism CHANGES the numerics (det and free runs
        # differ at identical seeds), so comparisons are only valid
        # within one determinism setting.
        tf.config.experimental.enable_op_determinism()
    subset = int(os.environ.get("SUBSET", "20000"))
    epochs = int(os.environ.get("EPOCHS", "2"))
    x, y, meta = load_data(subset)
    digest, losses, secs = train(x, y, epochs)
    result(mode="det-once" if deterministic else "free-once",
           deterministic_mode=deterministic,
           weights_sha256=digest, losses=losses, seconds=round(secs, 1),
           epochs=epochs, tf=tf.__version__,
           gpu=len(tf.config.list_physical_devices("GPU")), **meta)


def mode_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    expect = int(os.environ.get("EXPECT_GPUS", "1"))
    details = []
    for g in gpus:
        try:
            details.append(tf.config.experimental.get_device_details(g))
        except Exception as e:  # noqa: BLE001
            details.append({"error": str(e)})
    result(mode="gpus", visible=len(gpus), expected=expect,
           ok=len(gpus) == expect, details=details, tf=tf.__version__)


def build_transformer(d=256, blocks=6, heads=8, patch=4):
    L = tf.keras.layers
    inp = tf.keras.Input((32, 32, 3))
    t = L.Conv2D(d, patch, strides=patch)(inp)
    t = L.Reshape((-1, d))(t)
    for _ in range(blocks):
        a = L.LayerNormalization()(t)
        t = t + L.MultiHeadAttention(heads, d // heads)(a, a)
        m = L.LayerNormalization()(t)
        m = L.Dense(4 * d, activation="gelu")(m)
        t = t + L.Dense(d)(m)
    t = L.GlobalAveragePooling1D()(t)
    return tf.keras.Model(inp, L.Dense(10)(t))


def mode_bench(which):
    x, y, meta = load_data()
    m = (tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3),
                                        classes=10)
         if which == "resnet" else build_transformer())
    compile_(m)
    m.fit(x, y, batch_size=128, epochs=1, verbose=0)          # warm-up
    t0 = time.time()
    m.fit(x, y, batch_size=128, epochs=1, verbose=0)          # measured
    result(mode=f"bench-{which}", sec_per_epoch=round(time.time() - t0, 1),
           params=m.count_params(),
           note="fit/graph without XLA; framework eager loop will be slower",
           gpu=len(tf.config.list_physical_devices("GPU")), **meta)


def mode_quick():
    subset = int(os.environ.get("SUBSET", "2000"))
    x, y, meta = load_data(subset)
    _, losses, secs = train(x, y, epochs=1)
    result(mode="quick",
           ok=bool(np.isfinite(losses[-1])) and meta["zero_rows"] == 0
              and len(tf.config.list_physical_devices("GPU")) >= 1,
           seconds=round(secs, 1), final_loss=round(losses[-1], 4),
           gpu=len(tf.config.list_physical_devices("GPU")), **meta)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gpus"
    {"gpus": mode_gpus,
     "free-once": lambda: mode_once(False),
     "det-once": lambda: mode_once(True),
     "bench-resnet": lambda: mode_bench("resnet"),
     "bench-transformer": lambda: mode_bench("transformer"),
     "quick": mode_quick}[mode]()
