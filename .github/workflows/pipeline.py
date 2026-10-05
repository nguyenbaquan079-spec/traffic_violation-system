"""Pipeline phát hiện vi phạm: tracking phương tiện + luật vượt đèn đỏ / không đội mũ."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from config import Settings
from plate import PlateReader
from rules import (LightMonitor, box_center, clamp_box, crossed_line, expand_box,
                   line_direction, normalize_name, point_in_box)
from storage import Storage

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
MOTORCYCLE = 3
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VTYPE_EN = {"red_light": "RED LIGHT", "no_helmet": "NO HELMET"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
LIGHT_COLORS = {
    "red": (0, 0, 255), "yellow": (0, 255, 255),
    "green": (0, 200, 0), "unknown": (200, 200, 200),
}


@dataclass
class Track:
    tid: int
    cls: int
    box: tuple
    conf: float

    @property
    def foot(self):
        """Điểm chạm đất (giữa cạnh dưới hộp) dùng để xét vượt vạch."""
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])


def draw_zones(img, cfg: Settings, light: str = "unknown"):
    """Vẽ vạch dừng (kèm mũi tên hướng vi phạm) và vùng đèn lên ảnh."""
    color = LIGHT_COLORS.get(light, LIGHT_COLORS["unknown"])
    if cfg.stop_line:
        a = (int(cfg.stop_line[0][0]), int(cfg.stop_line[0][1]))
        b = (int(cfg.stop_line[1][0]), int(cfg.stop_line[1][1]))
        cv2.line(img, a, b, color, 3)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        ux, uy = line_direction(a, b, cfg.flip_direction)
        tip = (int(mid[0] + ux * 70), int(mid[1] + uy * 70))
        cv2.arrowedLine(img, mid, tip, color, 3, tipLength=0.3)
    if cfg.light_roi:
        x1, y1, x2, y2 = [int(v) for v in cfg.light_roi]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"light: {light}", (x1, max(15, y1 - 6)), FONT, 0.6, color, 2)
    return img


class ViolationSystem:
    def __init__(self, cfg: Settings, storage: Storage, log=print):
        from ultralytics import YOLO

        self.cfg, self.storage, self.log = cfg, storage, log
        self.detector = YOLO(cfg.vehicle_model)

        self.helmet = None
        if Path(cfg.helmet_model).exists():
            self.helmet = YOLO(cfg.helmet_model)
            log(f"Model mũ bảo hiểm — các lớp: {self.helmet.names}")
        else:
            log(f"[!] Không thấy model mũ bảo hiểm '{cfg.helmet_model}': tạm tắt luật không đội mũ.")

        self.plate = PlateReader(cfg.plate_model, cfg.use_ocr, cfg.conf_plate, cfg.device, log)
        self.light = LightMonitor()
        self.last_light = "unknown"

        self.frame_idx = 0
        self.prev_pts = {}               # track_id -> điểm chạm đất ở khung trước
        self.last_seen = {}              # track_id -> khung cuối thấy
        self.nh_count = defaultdict(int) # track_id (xe máy) -> số lần thấy không đội mũ
        self.reported = set()            # {(loại_vi_phạm, track_id)} đã ghi nhận
        self.flagged = {}                # track_id -> loại vi phạm (để vẽ khung đỏ)

    # ------------------------------------------------------------ nội bộ
    def _kw(self):
        kw = {"imgsz": self.cfg.imgsz, "verbose": False}
        if self.cfg.device:
            kw["device"] = self.cfg.device
        return kw

    def _track(self, frame):
        r = self.detector.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES,
            conf=self.cfg.conf_vehicle, **self._kw(),
        )[0]
        b = r.boxes
        if b is None or len(b) == 0 or b.id is None:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int)
        return [
            Track(int(i), int(c), tuple(float(v) for v in box), float(p))
            for box, c, p, i in zip(xyxy, cls, conf, ids)
        ]

    def _no_helmet_boxes(self, frame):
        r = self.helmet.predict(frame, conf=self.cfg.conf_helmet, **self._kw())[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        targets = {normalize_name(n) for n in self.cfg.no_helmet_names}
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        return [tuple(float(v) for v in box)
                for box, c in zip(xyxy, cls) if normalize_name(r.names[int(c)]) in targets]

    def _check_helmets(self, frame, tracks):
        h, w = frame.shape[:2]
        motos = [t for t in tracks if t.cls == MOTORCYCLE]
        hit = {}
        for nb in self._no_helmet_boxes(frame):
            c = box_center(nb)
            best, best_d = None, None
            for m in motos:
                if point_in_box(c, expand_box(m.box, w, h)):
                    mc = box_center(m.box)
                    d = math.hypot(c[0] - mc[0], c[1] - mc[1])
                    if best is None or d < best_d:
                        best, best_d = m, d
            if best is not None:
                hit[best.tid] = best

        events = []
        for m in motos:
            if m.tid in hit:
                self.nh_count[m.tid] += 1
                if self.nh_count[m.tid] >= self.cfg.helmet_confirm_frames:
                    events.append(("no_helmet", m))
            else:  # giảm dần để lọc báo nhầm thoáng qua
                self.nh_count[m.tid] = max(0, self.nh_count[m.tid] - 1)
        return events

    def _record(self, vtype, t: Track, frame):
        h, w = frame.shape[:2]
        crop_box = expand_box(t.box, w, h) if vtype == "no_helmet" else clamp_box(t.box, w, h)
        x1, y1, x2, y2 = crop_box
        vehicle = frame[y1:y2, x1:x2].copy()
        plate_img, text = self.plate.read(vehicle)

        scene = frame.copy()
        draw_zones(scene, self.cfg, "red" if vtype == "red_light" else "unknown")
        bx = clamp_box(t.box, w, h)
        cv2.rectangle(scene, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 3)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(scene, f"{VTYPE_EN[vtype]}  {stamp}", (20, 40), FONT, 1.0, (0, 0, 255), 2)

        rec = self.storage.add(vtype, t.tid, text, scene, vehicle, plate_img)
        self.log(f"VI PHẠM {VTYPE_EN[vtype]} — xe #{t.tid} — biển số: {text or '(chưa đọc được)'}")
        return rec

    def _draw(self, frame, tracks, light):
        img = frame.copy()
        draw_zones(img, self.cfg, light)
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.box)
            flagged = t.tid in self.flagged
            color = (0, 0, 255) if flagged else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if flagged else 2)
            label = f"{CLASS_NAMES.get(t.cls, '?')} #{t.tid}"
            if flagged:
                label += f" {VTYPE_EN[self.flagged[t.tid]]}"
            cv2.putText(img, label, (x1, max(15, y1 - 6)), FONT, 0.55, color, 2)
        return img

    def _prune(self):
        if self.frame_idx % 300:
            return
        old = [k for k, v in self.last_seen.items() if self.frame_idx - v > 600]
        for k in old:
            for d in (self.last_seen, self.prev_pts, self.nh_count):
                d.pop(k, None)

    # ------------------------------------------------------------ API chính
    def process(self, frame):
        """Xử lý 1 khung hình. Trả về (ảnh_đã_vẽ, danh_sách_vi_phạm_mới)."""
        cfg = self.cfg
        self.frame_idx += 1
        tracks = self._track(frame)

        light = self.light.update(frame, cfg.light_roi) if cfg.light_roi else "unknown"
        self.last_light = light

        events = []
        for t in tracks:
            self.last_seen[t.tid] = self.frame_idx
            prev = self.prev_pts.get(t.tid)
            self.prev_pts[t.tid] = t.foot
            if (cfg.stop_line and prev is not None and light == "red"
                    and ("red_light", t.tid) not in self.reported
                    and crossed_line(prev, t.foot, cfg.stop_line[0], cfg.stop_line[1],
                                     cfg.flip_direction)):
                events.append(("red_light", t))

        if self.helmet is not None and self.frame_idx % max(1, cfg.helmet_every) == 0:
            events += self._check_helmets(frame, tracks)

        records = []
        for vtype, t in events:
            key = (vtype, t.tid)
            if key in self.reported:
                continue
            self.reported.add(key)
            self.flagged[t.tid] = vtype
            records.append(self._record(vtype, t, frame))

        self._prune()
        return self._draw(frame, tracks, light), records
