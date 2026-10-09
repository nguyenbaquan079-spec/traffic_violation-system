"""Luồng nền đọc video và chạy pipeline, gửi kết quả về GUI qua signal."""
from __future__ import annotations

import cv2
from PySide6.QtCore import QThread, Signal


class VideoWorker(QThread):
    frame_ready = Signal(object, str)   # (ảnh BGR đã vẽ, trạng thái đèn)
    violation = Signal(dict)            # bản ghi vi phạm mới
    message = Signal(str)               # thông báo trạng thái / log

    def __init__(self, source, cfg, storage):
        super().__init__()
        self.source, self.cfg, self.storage = source, cfg, storage
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = None
        try:
            from pipeline import ViolationSystem  # import muộn để bắt lỗi thiếu thư viện

            self.message.emit("Đang tải model...")
            system = ViolationSystem(self.cfg, self.storage, log=self.message.emit)

            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.message.emit("Không mở được nguồn video.")
                return
            self.message.emit("Đang chạy...")

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, records = system.process(frame)
                self.frame_ready.emit(annotated, system.last_light)
                for rec in records:
                    self.violation.emit(rec)
            self.message.emit("Đã dừng.")
        except Exception as e:  # hiển thị lỗi lên GUI thay vì sập im lặng
            self.message.emit(f"Lỗi: {e}")
        finally:
            if cap is not None:
                cap.release()
