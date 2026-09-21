"""Detection frame (P1.1): annotated + privacy-transformed frame at cue time."""

import numpy as np
from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.ladder.orchestrator import _annotate_detection_frame
from care_ladder.models import CueEvent


def _frame() -> np.ndarray:
    f = np.zeros((120, 160, 3), dtype=np.uint8)
    f[40:100, 50:120] = 128  # person-ish block
    return f


def test_annotate_draws_bbox_and_numbers():
    cue = CueEvent(
        kind="distress_heuristic",
        confidence=0.85,
        detail={"bbox": [50, 40, 70, 60], "torso_angle_deg": 70.4, "pattern": "sudden_vertical_to_horizontal"},
    )
    out = _annotate_detection_frame(_frame(), cue)
    assert out is not None and out.shape == (120, 160, 3)
    assert not np.array_equal(out, _frame())  # annotation changed pixels


def test_annotate_none_inputs():
    assert _annotate_detection_frame(None, None) is None
    assert _annotate_detection_frame(_frame(), None) is None


def test_upload_incident_serves_detection_frame():
    # the fixture path attaches no frames; use the e2e photo fixture which does
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "opencv_dnn_person"})
        if r.status_code != 200:
            import pytest

            pytest.skip("models not downloaded")
        iid = r.json()["incident_id"]
        f = client.get(f"/incidents/{iid}/detection_frame")
        assert f.status_code == 200
        assert f.headers["content-type"] == "image/png"
        assert f.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_detection_frame_404_when_absent():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "no_movement_silence"})
        iid = r.json()["incident_id"]
        assert client.get(f"/incidents/{iid}/detection_frame").status_code == 404
        assert client.get("/incidents/nope/detection_frame").status_code == 404
