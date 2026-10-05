"""Phát hiện biển số (YOLO) và đọc ký tự (EasyOCR, tuỳ chọn)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rules import clamp_box

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."


class PlateReader:
    def __init__(self, model_path: str, use_ocr: bool = True, conf: float = 0.3,
                 device: str = "", log=print):
        self.model = None
        self.reader = None
        self.conf = conf
        self.device = device

        if model_path and Path(model_path).exists():
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        else:
            log(f"[!] Không thấy model biển số '{model_path}': chỉ lưu ảnh phương tiện, không cắt biển số.")

        if use_ocr and self.model is not None:
            try:
                import easyocr
                self.reader = easyocr.Reader(["en"], verbose=False)
            except Exception as e:  # chưa cài easyocr hoặc không tải được model OCR
                log(f"[!] Không bật được OCR ({e}). Vẫn lưu ảnh biển số.")

    def read(self, vehicle_img):
        """Trả về (ảnh_biển_số hoặc None, chuỗi_ký_tự)."""
        if self.model is None or vehicle_img is None or vehicle_img.size == 0:
            return None, ""
        kw = {"device": self.device} if self.device else {}
        res = self.model.predict(vehicle_img, conf=self.conf, verbose=False, **kw)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, ""
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        h, w = vehicle_img.shape[:2]
        x1, y1, x2, y2 = clamp_box(boxes[int(np.argmax(confs))], w, h)
        crop = vehicle_img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None, ""
        return crop, self._ocr(crop)

    def _ocr(self, crop) -> str:
        if self.reader is None:
            return ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            parts = self.reader.readtext(gray, detail=0, allowlist=PLATE_CHARS)
        except Exception:
            return ""
        return " ".join(parts).strip()
