"""Hàm hình học và luật: vạch dừng, trạng thái đèn giao thông, hộp bao."""
from __future__ import annotations

import math
import re
from collections import Counter, deque

import cv2


def normalize_name(name) -> str:
    """'No_Helmet' / 'no-helmet' / 'No  helmet' -> 'no helmet'."""
    return re.sub(r"[\s_\-]+", " ", str(name).lower()).strip()


# ---------------------------------------------------------------- vạch dừng
def side_of_line(p, a, b) -> float:
    """Dấu cho biết điểm p nằm ở phía nào của đường thẳng a->b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_line(prev, cur, a, b, flip: bool = False) -> bool:
    """True nếu đoạn prev->cur cắt đoạn a-b theo hướng từ phía dương sang phía âm.

    flip=True đảo chiều. Hướng vi phạm được vẽ bằng mũi tên trên video.
    """
    s1, s2 = side_of_line(prev, a, b), side_of_line(cur, a, b)
    if flip:
        s1, s2 = -s1, -s2
    if not (s1 > 0 and s2 <= 0):
        return False
    # đảm bảo giao điểm nằm trong đoạn a-b (không phải trên phần kéo dài)
    d1, d2 = side_of_line(a, prev, cur), side_of_line(b, prev, cur)
    return d1 * d2 <= 0


def line_direction(a, b, flip: bool = False):
    """Vector đơn vị chỉ hướng xe bị coi là vượt vạch (để vẽ mũi tên)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dy / n, -dx / n
    return (-ux, -uy) if flip else (ux, uy)


# ---------------------------------------------------------------- hộp bao
def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return (
        int(max(0, min(w - 1, x1))),
        int(max(0, min(h - 1, y1))),
        int(max(0, min(w, x2))),
        int(max(0, min(h, y2))),
    )


def expand_box(box, w, h, left=0.15, right=0.15, up=1.0, down=0.0):
    """Mở rộng hộp (mặc định mở lên trên ~1 chiều cao để bao cả người lái)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clamp_box((x1 - left * bw, y1 - up * bh, x2 + right * bw, y2 + down * bh), w, h)


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


# ---------------------------------------------------------------- đèn giao thông
def classify_light(frame, roi, min_ratio: float = 0.01) -> str:
    """Phân loại màu đèn trong vùng roi bằng HSV: red / yellow / green / unknown."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return "unknown"
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    bright = (sat > 90) & (val > 150)
    counts = {
        "red": int((bright & ((hue <= 10) | (hue >= 165))).sum()),
        "yellow": int((bright & (hue > 15) & (hue <= 35)).sum()),
        "green": int((bright & (hue >= 40) & (hue <= 95)).sum()),
    }
    state, n = max(counts.items(), key=lambda kv: kv[1])
    need = max(6, min_ratio * hsv.shape[0] * hsv.shape[1])
    return state if n >= need else "unknown"


class LightMonitor:
    """Làm mượt trạng thái đèn bằng bỏ phiếu đa số qua vài khung hình gần nhất."""

    def __init__(self, history: int = 7):
        self.hist = deque(maxlen=history)

    def update(self, frame, roi) -> str:
        self.hist.append(classify_light(frame, roi))
        return Counter(self.hist).most_common(1)[0][0]
