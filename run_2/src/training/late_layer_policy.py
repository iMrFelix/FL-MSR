"""Late-layer policies for ImpRoute.

A *late layer* is a `LayerUpdate` that arrives at the receiver after its
round has already been completed by the ε-deadline trigger
(see ``docs/extensions/02-epsilon-trigger.md``).

The three slippage modes admitted by the step-2 gate (ruling G3,
``writeup/01-candidate-selection.md`` §7) share one receiver-side
behaviour: a late layer never enters its already-sealed round.  What
differs is how the *aggregator* compensates for the resulting missing
contribution, implemented in
``src.algorithms.fedavg.FedAvg._aggregate_as_aggregator``:

- ``drop`` (:class:`DropPolicy`, control): missing contributions are
  stale-filled from the aggregator's current global.
- ``recycle_last_delta`` (:class:`RecycleLastDeltaPolicy`, alias
  ``recycle``): the layer's previous aggregated delta is re-applied on
  the missing sender's behalf (FedLUAR-adapted).
- ``renormalize`` (:class:`RenormalizePolicy`, alias ``renorm``): the
  layer is averaged over arrived contributors only, with weights
  renormalized to sum to 1.

The policy classes below therefore differ only in name and log line:
they exist so the configured mode can be constructed by name and so late
arrivals are attributed to the right experiment arm in the logs.
Policies that *retain or apply* late payloads (async-apply,
buffer-for-next, error-feedback, skip-feedback v2) were deferred by the
gate — see ``writeup/01-candidate-selection.md`` §4 and §7.

See ``docs/extensions/03-late-layer-policy.md`` for the conceptual
contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class LateLayerPolicy(ABC):
    """Handler for layers arriving after their round has completed."""

    @abstractmethod
    def on_late_arrival(
        self,
        source_node: str,
        round_num: int,
        layer_name: str,
        array: np.ndarray,
        num_samples: int,
    ) -> None:
        """React to a late layer.

        Implementations may store, apply, or discard the layer, but must
        not raise on a well-formed call.  Any operational failure should
        be logged and swallowed so a single bad layer cannot poison the
        rest of the round.
        """


class DropPolicy(LateLayerPolicy):
    """Discard the late layer.

    The aggregator implicitly uses whatever value it already has for this
    layer (the previous round's aggregate, or the initial weight).  This
    is the simplest correct behaviour and the default for the ε-deadline
    preliminary experiment.
    """

    def on_late_arrival(
        self,
        source_node: str,
        round_num: int,
        layer_name: str,
        array: np.ndarray,
        num_samples: int,
    ) -> None:
        logger.debug(
            "Dropping late layer %s from %s (round %d)",
            layer_name, source_node, round_num,
        )
        # Intentionally no-op.


class RecycleLastDeltaPolicy(LateLayerPolicy):
    """Discard the late layer; the aggregator recycles in its place.

    The compensation does not use this payload at all: the round's
    contribution set is sealed the moment the ε-trigger fires, and the
    recycled value (current global + the layer's previous aggregated
    delta) is derived purely from aggregator state at aggregation time.
    Retaining the late payload here would be the deferred async-apply /
    error-feedback territory, explicitly out of scope for this arm
    (gate ruling G3).
    """

    def on_late_arrival(
        self,
        source_node: str,
        round_num: int,
        layer_name: str,
        array: np.ndarray,
        num_samples: int,
    ) -> None:
        logger.debug(
            "Late layer %s from %s (round %d) discarded; aggregator "
            "recycles the layer's previous aggregated delta instead",
            layer_name, source_node, round_num,
        )
        # Intentionally no-op (see class docstring).


class RenormalizePolicy(LateLayerPolicy):
    """Discard the late layer; the aggregator renormalizes without it.

    The layer's aggregate is computed over the contributors that did
    arrive, with their weights renormalized to sum to 1 — no value from
    this payload is needed, so the receiver-side action is identical to
    :class:`DropPolicy` (gate ruling G3).
    """

    def on_late_arrival(
        self,
        source_node: str,
        round_num: int,
        layer_name: str,
        array: np.ndarray,
        num_samples: int,
    ) -> None:
        logger.debug(
            "Late layer %s from %s (round %d) discarded; aggregator "
            "renormalizes the layer over arrived contributors",
            layer_name, source_node, round_num,
        )
        # Intentionally no-op (see class docstring).


# Long names match the config-schema literals
# (src/config/schema.py::TrainingConfig.late_layer_policy); the short
# aliases match FedAvg's normalized mode names so both spellings resolve
# to the same behaviour everywhere.
_POLICY_REGISTRY: dict[str, type[LateLayerPolicy]] = {
    "drop": DropPolicy,
    "recycle_last_delta": RecycleLastDeltaPolicy,
    "recycle": RecycleLastDeltaPolicy,
    "renormalize": RenormalizePolicy,
    "renorm": RenormalizePolicy,
}


def make_policy(name: str, **kwargs) -> LateLayerPolicy:
    """Construct a `LateLayerPolicy` by name.

    Args:
        name: One of the registered policy names: "drop",
            "recycle_last_delta" (alias "recycle"), or "renormalize"
            (alias "renorm").
        **kwargs: Forwarded to the policy constructor.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    try:
        cls = _POLICY_REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(_POLICY_REGISTRY))
        raise ValueError(
            f"Unknown late-layer policy {name!r}. "
            f"Registered: {registered}"
        ) from None
    return cls(**kwargs)
