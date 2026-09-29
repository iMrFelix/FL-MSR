"""Tests for the overnight-evaluation proto additions.

Two wire-format extensions (see writeup/01-candidate-selection.md):

- ``ImportanceEntry.sched_score`` (gate ruling G2): ``raw_score`` is frozen
  as the ε-trigger accounting metric for all arms; scheduling variants carry
  their score separately so same-ε comparisons stay valid.
- ``MetricsReport.uplink_telemetry_json`` (pre-run fix 1): receiver-side
  t_ε telemetry sidecar, JSON-encoded.

Both must round-trip on the wire and default safely (proto3 zero values)
so that messages produced by older code remain parseable.
"""

import json

import pytest

from src.proto_gen import federation_pb2


# ---------------------------------------------------------------------------
# ImportanceEntry.sched_score
# ---------------------------------------------------------------------------

def test_sched_score_round_trips():
    entry = federation_pb2.ImportanceEntry(
        layer_name="conv_0/kernel",
        raw_score=0.5,
        must_receive=True,
        sched_score=0.125,
    )
    blob = entry.SerializeToString()
    parsed = federation_pb2.ImportanceEntry()
    parsed.ParseFromString(blob)
    assert parsed.layer_name == "conv_0/kernel"
    assert parsed.raw_score == pytest.approx(0.5)
    assert parsed.must_receive is True
    assert parsed.sched_score == pytest.approx(0.125)


def test_sched_score_defaults_to_zero():
    """proto3 default: entries built without sched_score read back as 0.0.

    This is the single-metric-arm convention (G2): sched_score == 0.0 means
    "scheduling score equals the trigger score"; old serialized manifests
    therefore stay valid.
    """
    entry = federation_pb2.ImportanceEntry(layer_name="a", raw_score=1.0)
    blob = entry.SerializeToString()
    parsed = federation_pb2.ImportanceEntry()
    parsed.ParseFromString(blob)
    assert parsed.sched_score == 0.0


def test_manifest_entries_carry_independent_scores():
    """raw_score (trigger) and sched_score (routing) diverge per entry."""
    manifest = federation_pb2.RoundManifest(round=4, source_node_id="node-2")
    for name, raw, sched in [("k0", 0.9, 0.1), ("k1", 0.1, 0.9)]:
        e = manifest.entries.add()
        e.layer_name = name
        e.raw_score = raw
        e.sched_score = sched

    blob = manifest.SerializeToString()
    parsed = federation_pb2.RoundManifest()
    parsed.ParseFromString(blob)

    by_name = {e.layer_name: e for e in parsed.entries}
    assert by_name["k0"].raw_score == pytest.approx(0.9)
    assert by_name["k0"].sched_score == pytest.approx(0.1)
    assert by_name["k1"].raw_score == pytest.approx(0.1)
    assert by_name["k1"].sched_score == pytest.approx(0.9)


def test_old_manifest_bytes_still_parse():
    """A manifest serialized without the new field parses cleanly.

    Simulates the rolling-upgrade case: receivers built from the new proto
    must accept bytes from senders built from the old one.
    """
    old_style = federation_pb2.RoundManifest(round=1, source_node_id="node-0")
    e = old_style.entries.add()
    e.layer_name = "dense/bias"
    e.raw_score = 0.3
    # sched_score intentionally untouched.
    blob = old_style.SerializeToString()

    parsed = federation_pb2.RoundManifest()
    parsed.ParseFromString(blob)
    assert parsed.entries[0].sched_score == 0.0
    assert parsed.entries[0].raw_score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# MetricsReport.uplink_telemetry_json
# ---------------------------------------------------------------------------

def _example_telemetry() -> dict:
    """A telemetry dict exercising every field of the documented schema."""
    return {
        "node-1": {
            "manifest_arrival_rel": 0.0123,
            "trigger_fire_rel": 0.871,
            "watchdog_fired": False,
            "t_eps_local": 0.8587,
            "layer_arrivals_rel": {"conv_0/kernel": 0.42, "conv_0/bias": 0.05},
            "shed_layers": ["dense_1/kernel"],
            "kappa_realized": 0.04,
            "ordering_violation": False,
        },
        "node-2": {
            "manifest_arrival_rel": 0.02,
            "trigger_fire_rel": None,
            "watchdog_fired": True,
            "t_eps_local": None,
            "layer_arrivals_rel": {},
            "shed_layers": [],
            "kappa_realized": None,
            "ordering_violation": True,
        },
    }


def test_uplink_telemetry_json_round_trips():
    payload = json.dumps(_example_telemetry())
    report = federation_pb2.MetricsReport(
        node_id="node-0",
        round=7,
        uplink_telemetry_json=payload,
    )
    blob = report.SerializeToString()
    parsed = federation_pb2.MetricsReport()
    parsed.ParseFromString(blob)
    assert parsed.uplink_telemetry_json == payload

    # The sidecar must survive as parseable JSON with types intact.
    decoded = json.loads(parsed.uplink_telemetry_json)
    assert decoded["node-1"]["t_eps_local"] == pytest.approx(0.8587)
    assert decoded["node-2"]["trigger_fire_rel"] is None
    assert decoded["node-2"]["watchdog_fired"] is True
    assert decoded["node-1"]["shed_layers"] == ["dense_1/kernel"]


def test_uplink_telemetry_defaults_to_empty_string():
    """Reports from nodes without uplink telemetry read back as ''.

    Empty string is the documented "nothing to report" value (workers under
    FedAvg, monolithic mode); the collector must be able to skip it.
    """
    report = federation_pb2.MetricsReport(node_id="node-3", round=0)
    blob = report.SerializeToString()
    parsed = federation_pb2.MetricsReport()
    parsed.ParseFromString(blob)
    assert parsed.uplink_telemetry_json == ""


def test_report_with_telemetry_inside_envelope():
    """Full Envelope round-trip, as it travels node -> monitor."""
    report = federation_pb2.MetricsReport(
        node_id="node-0",
        round=2,
        uplink_telemetry_json=json.dumps(_example_telemetry()),
    )
    envelope = federation_pb2.Envelope(
        source_node="node-0",
        dest_node="monitor",
        timestamp_ns=123456789,
        metrics_report=report,
    )
    blob = envelope.SerializeToString()
    parsed = federation_pb2.Envelope()
    parsed.ParseFromString(blob)
    assert parsed.WhichOneof("payload") == "metrics_report"
    decoded = json.loads(parsed.metrics_report.uplink_telemetry_json)
    assert set(decoded.keys()) == {"node-1", "node-2"}


# ---------------------------------------------------------------------------
# Audit T2 additions: inclusion ack (TRIG-1/ML-01), divergence marker (ML-04)
# ---------------------------------------------------------------------------

def test_inclusion_ack_round_trips_on_the_skip_advice_channel():
    advice = federation_pb2.SkipAdvice(
        round=4,
        layer_names=["conv_0/bias"],
        inclusion_ack=True,
        included_layers=["conv_1/kernel", "head/bias"],
    )
    parsed = federation_pb2.SkipAdvice()
    parsed.ParseFromString(advice.SerializeToString())
    assert parsed.round == 4
    assert list(parsed.layer_names) == ["conv_0/bias"]
    assert parsed.inclusion_ack is True
    assert list(parsed.included_layers) == ["conv_1/kernel", "head/bias"]


def test_empty_ack_is_distinguishable_from_no_ack():
    """"None of your layers made it" is the case the counter needs most.

    A repeated field alone cannot express it (proto3 has no presence for
    repeated fields), which is why `inclusion_ack` is carried separately.
    """
    none_included = federation_pb2.SkipAdvice(round=1, inclusion_ack=True)
    no_ack = federation_pb2.SkipAdvice(round=1)
    for message in (none_included, no_ack):
        parsed = federation_pb2.SkipAdvice()
        parsed.ParseFromString(message.SerializeToString())
        assert list(parsed.included_layers) == []
        assert parsed.inclusion_ack == message.inclusion_ack
    assert none_included.inclusion_ack != no_ack.inclusion_ack


def test_divergence_marker_round_trips():
    report = federation_pb2.MetricsReport(
        node_id="node-1",
        round=39,
        val_accuracy=0.1,
        val_loss=float("nan"),
        diverged=True,
        diverged_reason="non_finite_weights:conv_2/kernel",
    )
    parsed = federation_pb2.MetricsReport()
    parsed.ParseFromString(report.SerializeToString())
    assert parsed.diverged is True
    assert parsed.diverged_reason == "non_finite_weights:conv_2/kernel"


def test_divergence_marker_defaults_to_healthy():
    """Pre-fix reports parse as not-diverged, so consumers can gate on it."""
    parsed = federation_pb2.MetricsReport()
    parsed.ParseFromString(
        federation_pb2.MetricsReport(node_id="node-2", round=0)
        .SerializeToString()
    )
    assert parsed.diverged is False
    assert parsed.diverged_reason == ""
