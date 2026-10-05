"""Test logic bằng model giả: python -m tests.test_logic  (chạy từ thư mục gốc dự án)."""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from rules import classify_light, crossed_line, line_direction  # noqa: E402
from storage import Storage  # noqa: E402


# ---------------------------------------------------------------- model giả
class T:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, xyxy, cls, ids=None):
        self.xyxy, self.cls = T(xyxy), T(cls)
        self.conf = T([0.9] * len(cls))
        self.id = T(ids) if ids is not None else None
        self._n = len(cls)

    def __len__(self):
        return self._n


class Res:
    def __init__(self, boxes, names=None):
        self.boxes, self.names = boxes, names or {}


class FakeDetector:
    def __init__(self, script):
        self.script, self.i = script, 0

    def track(self, frame, **kw):
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [out]


class FakeHelmet:
    names = {0: "With Helmet", 1: "Without_Helmet"}

    def __init__(self, per_frame):
        self.per_frame, self.i = per_frame, 0

    def predict(self, frame, **kw):
        boxes = self.per_frame[min(self.i, len(self.per_frame) - 1)]
        self.i += 1
        if not boxes:
            return [Res(None, self.names)]
        return [Res(Boxes(boxes, [1] * len(boxes)), self.names)]


def make_system(detector, helmet, cfg, tmp):
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda *a, **k: detector
    sys.modules["ultralytics"] = fake
    import importlib
    import pipeline
    importlib.reload(pipeline)
    storage = Storage(str(Path(tmp) / "t.db"), str(Path(tmp) / "ev"))
    sysm = pipeline.ViolationSystem(cfg, storage, log=lambda m: None)
    sysm.helmet = helmet
    return sysm, storage


def frame_with_light(color_bgr):
    f = np.full((480, 640, 3), 40, np.uint8)
    f[20:40, 20:40] = color_bgr
    return f


# ---------------------------------------------------------------- các test
def test_geometry():
    a, b = (0, 320), (640, 320)
    assert crossed_line((300, 340), (300, 300), a, b)              # đi lên -> vi phạm
    assert not crossed_line((300, 300), (300, 340), a, b)          # chiều ngược -> không
    assert crossed_line((300, 300), (300, 340), a, b, flip=True)   # đảo chiều -> có
    assert not crossed_line((900, 340), (900, 300), a, b)          # cắt phần kéo dài -> không
    assert line_direction(a, b)[1] < 0                             # mũi tên hướng lên


def test_light_colors():
    roi = [10, 10, 50, 50]
    assert classify_light(frame_with_light((0, 0, 255)), roi) == "red"
    assert classify_light(frame_with_light((0, 255, 0)), roi) == "green"
    assert classify_light(frame_with_light((0, 255, 255)), roi) == "yellow"
    assert classify_light(np.full((480, 640, 3), 40, np.uint8), roi) == "unknown"


def test_red_light(tmp):
    cfg = Settings()
    cfg.stop_line, cfg.light_roi = [[0, 320], [640, 320]], [10, 10, 50, 50]
    cfg.helmet_model = "none"

    def car(y2):
        return Res(Boxes([[200, y2 - 80, 300, y2]], [2], [1]))

    script = [car(340), car(330), car(310), car(290)]
    d = FakeDetector(script)
    s, st = make_system(d, None, cfg, tmp)
    red = frame_with_light((0, 0, 255))
    recs = []
    for _ in script:
        _, r = s.process(red)
        recs += r
    assert len(recs) == 1 and recs[0]["vtype"] == "red_light", recs
    assert Path(recs[0]["scene_path"]).exists() and Path(recs[0]["vehicle_path"]).exists()
    assert len(st.recent()) == 1

    # đèn xanh -> không vi phạm
    d2 = FakeDetector(script)
    s2, _ = make_system(d2, None, cfg, tmp)
    green = frame_with_light((0, 255, 0))
    assert sum(len(s2.process(green)[1]) for _ in script) == 0


def test_no_helmet(tmp):
    cfg = Settings()
    cfg.helmet_every, cfg.helmet_confirm_frames = 1, 3
    moto = Res(Boxes([[300, 300, 400, 400]], [3], [7]))
    d = FakeDetector([moto])
    head = [[330, 230, 370, 280]]  # đầu người lái nằm phía trên hộp xe máy
    h = FakeHelmet([head])
    s, _ = make_system(d, h, cfg, tmp)
    f = np.full((480, 640, 3), 90, np.uint8)
    counts = [len(s.process(f)[1]) for _ in range(5)]
    assert counts == [0, 0, 1, 0, 0], counts  # xác nhận ở khung thứ 3, chỉ báo 1 lần

    # đầu nằm ngoài vùng xe máy -> không gán
    d = FakeDetector([moto])
    h = FakeHelmet([[[10, 10, 40, 40]]])
    s, _ = make_system(d, h, cfg, tmp)
    assert sum(len(s.process(f)[1]) for _ in range(5)) == 0


if __name__ == "__main__":
    test_geometry()
    test_light_colors()
    for fn in (test_red_light, test_no_helmet):
        with tempfile.TemporaryDirectory() as tmp:
            fn(tmp)
    print("Tất cả test đạt.")
