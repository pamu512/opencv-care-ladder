"""Live camera session: lifecycle, cue dispatch, guardrails (stubbed capture)."""

import threading
import time

import numpy as np
import pytest

from care_ladder.models import CueEvent
from care_ladder.vision.camera import CameraSession, start_session, active_session, stop_session


class FakeCapture:
    """Duck-typed cv2.VideoCapture returning synthesized frames."""

    def __init__(self, frames, fps=30.0):
        self.frames = frames
        self.fps = fps
        self.i = 0
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps if prop == 5 else 0.0  # CAP_PROP_FPS (id 5 in cv2)

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        self.i += 1
        return True, f

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, cue_after=None):
        self.cue_after = cue_after  # frame index at which to fire
        self.observed = 0

    def observe(self, frame, t=0.0):
        self.observed += 1
        if self.cue_after is not None and self.observed - 1 >= self.cue_after:
            return CueEvent(kind="no_movement", confidence=0.9, detail={"live": True})
        return None


@pytest.fixture()
def fake_cv2(monkeypatch):
    made = {}

    class FakeCV2:
        VideoCapture = staticmethod(lambda src: made.setdefault("cap", FakeCapture(FRAMES)))
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 1

    import sys
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    return made


FRAMES = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(60)]


def test_session_fires_cue_and_dispatches(fake_cv2):
    fired = []
    det = FakeDetector(cue_after=5)

    async def on_cue(cue, t):
        fired.append((cue.kind, t))

    s = CameraSession(source=0, detector=det, on_incident=on_cue, max_seconds=60)
    # patch the capture the module-level import will create
    fake_cv2["cap"] = FakeCapture(FRAMES)
    s.start()
    s.thread.join(timeout=5)
    assert fired, "cue never dispatched"
    assert fired[0][0] == "no_movement"
    assert s.last_cue["detail"]["live"] is True


class BlockingCapture(FakeCapture):
    def read(self):
        time.sleep(0.05)
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

def test_session_stops_on_signal(fake_cv2):
    fake_cv2["cap"] = BlockingCapture([np.zeros((120, 160, 3), dtype=np.uint8)])
    det = FakeDetector(cue_after=None)
    s = CameraSession(source=0, detector=det, on_incident=None, max_seconds=600)
    s.start()
    time.sleep(0.3)
    assert s.running
    s.stop()
    s.thread.join(timeout=5)
    assert not s.running


def test_session_error_when_capture_unopenable(monkeypatch):
    import sys

    class Dead:
        @staticmethod
        def isOpened():
            return False

        @staticmethod
        def get(prop):
            return 0.0

        @staticmethod
        def release():
            pass

    monkeypatch.setitem(sys.modules, "cv2", type("M", (), {"VideoCapture": staticmethod(lambda src: Dead())}))
    s = CameraSession(source=99, detector=FakeDetector(), on_incident=None)
    s.start()
    s.thread.join(timeout=5)
    assert "could not open" in (s.error or "")


def test_global_registry_one_session(fake_cv2):
    stop_session()  # clean slate
    fake_cv2["cap"] = BlockingCapture([np.zeros((120, 160, 3), dtype=np.uint8)])
    det = FakeDetector(cue_after=None)
    s1 = start_session(source=0, detector=det, on_incident=None, max_seconds=600)
    assert active_session() is s1
    try:
        with pytest.raises(RuntimeError):
            start_session(source=0, detector=det, on_incident=None, max_seconds=600)
    finally:
        stop_session()
    assert active_session() is None
