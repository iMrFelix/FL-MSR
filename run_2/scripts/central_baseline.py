#!/usr/bin/env python3
"""Central (non-federated) baseline of the EXACT deep_cnn on full CIFAR-10.

Collapse-forensics diagnostic 2 (writeup/16 §5): the "63% best-ever vs
75-82% literature ceiling" comparison was literature-anchored, not measured
for this exact net.  This script measures it: same DeepCNN module, same
[0,1] float32 normalization, same batch size 64, full 50k train / 10k test.

Two arms:
- fl_matched:  plain SGD lr=0.1, momentum 0, no clipping, constant LR —
               the exact optimizer the FL runs use.  If this plateaus near
               the FL plateau, the plateau is optimizer-induced, not
               capacity- or federation-induced.
- competent:   SGD momentum 0.9, cosine decay to 0, clipnorm 1.0 — the
               competent-recipe ceiling for this architecture.

Usage: python scripts/central_baseline.py --arm fl_matched --epochs 60
Writes per-epoch JSON to campaigns/central/<arm>.json (test acc measured
every epoch on the full 10k test set).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.deep_cnn import DeepCNN  # noqa: E402

ARMS = {
    "fl_matched": dict(lr=0.1, momentum=0.0, clipnorm=None, cosine=False),
    "competent": dict(lr=0.1, momentum=0.9, clipnorm=1.0, cosine=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    spec = ARMS[args.arm]

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64).ravel()
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int64).ravel()

    model = DeepCNN()
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[{args.arm}] DeepCNN {n_params} params, "
          f"{len(x_train)} train / {len(x_test)} test", flush=True)

    opt_kwargs = {}
    if spec["clipnorm"] is not None:
        opt_kwargs["clipnorm"] = spec["clipnorm"]
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=spec["lr"], momentum=spec["momentum"], **opt_kwargs,
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=args.seed, reshuffle_each_iteration=True)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(512)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(xb, yb):
        logits = model(xb, training=False)
        loss = loss_fn(yb, logits)
        correct = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(logits, axis=1), yb), tf.float32)
        )
        return loss * tf.cast(tf.shape(yb)[0], tf.float32), correct

    out_dir = ROOT / "campaigns/central"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.json"
    history: list[dict] = []
    base_lr = spec["lr"]

    for epoch in range(args.epochs):
        if spec["cosine"]:
            horizon = max(args.epochs - 1, 1)
            new_lr = 0.5 * base_lr * (
                1.0 + math.cos(math.pi * min(epoch, horizon) / horizon)
            )
            optimizer.learning_rate.assign(new_lr)
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_ds:
            train_loss += float(train_step(xb, yb))
            n_batches += 1
        loss_sum, correct_sum = 0.0, 0.0
        for xb, yb in test_ds:
            ls, cs = eval_step(xb, yb)
            loss_sum += float(ls)
            correct_sum += float(cs)
        rec = dict(
            epoch=epoch,
            lr=float(optimizer.learning_rate),
            train_loss=train_loss / max(n_batches, 1),
            test_loss=loss_sum / len(x_test),
            test_acc=correct_sum / len(x_test),
            seconds=time.time() - t0,
        )
        history.append(rec)
        out_path.write_text(json.dumps(
            dict(arm=args.arm, seed=args.seed, epochs=args.epochs,
                 batch_size=args.batch_size, spec=spec, history=history),
            indent=1,
        ))
        print(f"[{args.arm}] epoch {epoch:3d} lr={rec['lr']:.4f} "
              f"train_loss={rec['train_loss']:.4f} "
              f"test_loss={rec['test_loss']:.4f} "
              f"test_acc={rec['test_acc']:.4f} ({rec['seconds']:.1f}s)",
              flush=True)

    best = max(history, key=lambda r: r["test_acc"])
    print(f"[{args.arm}] DONE best test_acc={best['test_acc']:.4f} "
          f"@epoch {best['epoch']} final={history[-1]['test_acc']:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
