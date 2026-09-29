"""Map per-layer importance scores to traffic classes.

In per-layer update mode, each layer is sent as a separate protobuf message.
The importance metric assigns a scalar score to each layer (e.g. gradient L2
norm), and this module decides which traffic class each layer should use.

Traffic class 0 is the highest-priority class (lowest latency, highest
bandwidth, lowest drop rate in tc/netem rules).  Higher class indices get
progressively worse treatment.

The mapping works by finding the (num_classes − 1) largest *gaps* between
consecutive importance scores (when sorted descending) and using those gaps
as class boundaries.  This respects natural clusters in the score distribution
rather than forcing equal-size buckets.

Example with scores [1, 0.9, 0.3, 0.28, 0.27, 0.04, 0.03, 0.02, 0.01]
and 3 classes:
    - Gaps: 0.10, **0.60**, 0.02, 0.01, **0.23**, 0.01, 0.01, 0.01
    - Two largest gaps → boundaries after positions 1 and 4
    - Class 0: scores 1.0, 0.9   (highest importance)
    - Class 1: scores 0.3, 0.28, 0.27
    - Class 2: scores 0.04, 0.03, 0.02, 0.01

When all scores are equal (e.g. the uniform importance metric) every gap is
zero and there is no natural boundary; the function falls back to equal-size
buckets with alphabetical tie-breaking, preserving the original behaviour.
"""

from __future__ import annotations


def assign_traffic_classes(
    importance_scores: dict[str, float],
    num_classes: int,
) -> dict[str, int]:
    """Assign each layer to a traffic class based on its importance score.

    Layers are sorted by importance in descending order.  The
    ``num_classes − 1`` largest gaps between consecutive scores are used as
    class boundaries.  All layers above the first boundary go into class 0,
    those between the first and second boundary into class 1, and so on.

    **Fallback to equal-size buckets**: when all gaps are zero (all scores
    identical, as with the ``uniform`` importance metric), there are no
    meaningful boundaries and the function falls back to dividing layers into
    equal-size buckets with alphabetical tie-breaking.

    Tie-breaking among equal gaps: the boundary is placed as early as possible
    (i.e. at the higher-importance side), keeping the top class small.

    Args:
        importance_scores: Mapping from layer name to its importance score.
            Higher scores mean more important.
        num_classes: Total number of traffic classes available (from the
            experiment config's ``traffic_classes.num_classes``).

    Returns:
        Mapping from layer name to assigned traffic class index (0-based,
        where 0 is highest priority).
    """
    if num_classes <= 1:
        return {name: 0 for name in importance_scores}, []

    # Sort layers by importance descending; alphabetical tie-break for
    # determinism across nodes that compute the same scores independently.
    ranked = sorted(
        importance_scores.keys(),
        key=lambda name: (-importance_scores[name], name),
    )

    n_layers = len(ranked)

    # Edge case: fewer layers than classes → each layer in its own class.
    if n_layers <= num_classes:
        return {name: cls for cls, name in enumerate(ranked)}, []

    scores = [importance_scores[name] for name in ranked]

    # Gaps between consecutive sorted scores (all non-negative).
    gaps = [scores[i] - scores[i + 1] for i in range(n_layers - 1)]

    # If all gaps are zero, scores are uniform → fall back to equal-size buckets.
    if all(g == 0.0 for g in gaps):
        assignment: dict[str, int] = {}
        base_size = n_layers // num_classes
        remainder = n_layers % num_classes
        idx = 0
        for cls in range(num_classes):
            bucket_size = base_size + (1 if cls < remainder else 0)
            for _ in range(bucket_size):
                assignment[ranked[idx]] = cls
                idx += 1
        return assignment, []

    # Find the (num_classes − 1) largest gaps; use their positions as class
    # boundaries.  Tie-break by position (prefer earlier = higher-importance).
    boundary_set = set(
        sorted(range(len(gaps)), key=lambda i: (-gaps[i], i))[: num_classes - 1]
    )

    assignment = {}
    cls = 0
    for i, name in enumerate(ranked):
        assignment[name] = cls
        if i in boundary_set:
            cls += 1

    return assignment, gaps
