"""Giao diện hệ thống phát hiện vi phạm giao thông (PySide6)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from config import Settings
from pipeline import draw_zones
from storage import Storage
from worker import VideoWorker

VTYPE_VI = {"red_light": "Vượt đèn đỏ", "no_helmet": "Không đội mũ bảo hiểm"}
LIGHT_VI = {"red": "ĐỎ", "yellow": "VÀNG", "green": "XANH", "unknown": "chưa rõ"}


def bgr_to_qimage(bgr) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def load_pixmap(path: str, w: int, h: int) -> QPixmap:
    if not path or not Path(path).exists():
        return QPixmap()
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return QPixmap()
    return QPixmap.fromImage(bgr_to_qimage(img)).scaled(
        w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class VideoLabel(QLabel):
    """Hiển thị video giữ tỉ lệ; phát toạ độ click theo hệ toạ độ ảnh gốc."""
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__("Chưa có video.\nChọn nguồn video ở phía trên.")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#111; color:#aaa;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._iw = self._ih = 0

    def show_frame(self, bgr):
        self._ih, self._iw = bgr.shape[:2]
        pm = QPixmap.fromImage(bgr_to_qimage(bgr)).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pm)

    def mousePressEvent(self, e):
        pm = self.pixmap()
        if pm is None or pm.isNull() or not self._iw:
            return
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        x = (e.position().x() - ox) * self._iw / pm.width()
        y = (e.position().y() - oy) * self._ih / pm.height()
        if 0 <= x < self._iw and 0 <= y < self._ih:
            self.clicked.emit(int(x), int(y))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống phát hiện vi phạm giao thông")
        self.cfg = Settings.load()
        self.storage = Storage(self.cfg.db_path, self.cfg.evidence_dir)
        self.worker = None
        self.source = None
        self.base_frame = None
        self.pick_mode = None
        self.pick_pts = []
        self.records = []

        # ---- thanh điều khiển
        self.btn_file = QPushButton("Mở video…")
        self.btn_cam = QPushButton("Webcam")
        self.btn_rtsp = QPushButton("Camera IP…")
        self.btn_start = QPushButton("▶ Bắt đầu")
        self.btn_stop = QPushButton("■ Dừng")
        self.btn_line = QPushButton("Đặt vạch dừng")
        self.btn_roi = QPushButton("Chọn vùng đèn")
        self.chk_flip = QCheckBox("Đảo chiều vạch")
        self.chk_flip.setChecked(self.cfg.flip_direction)
        self.lbl_light = QLabel("Đèn: chưa rõ")
        self.btn_stop.setEnabled(False)

        bar = QHBoxLayout()
        for w in (self.btn_file, self.btn_cam, self.btn_rtsp, self.btn_start, self.btn_stop,
                  self.btn_line, self.btn_roi, self.chk_flip):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.lbl_light)

        self.video = VideoLabel()
        left = QVBoxLayout()
        left.addLayout(bar)
        left.addWidget(self.video, 1)

        # ---- bảng vi phạm + ảnh bằng chứng
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Thời gian", "Vi phạm", "Xe #", "Biển số"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        self.img_scene = self._preview("Ảnh toàn cảnh", 420, 240)
        self.img_vehicle = self._preview("Phương tiện", 205, 160)
        self.img_plate = self._preview("Biển số", 205, 160)
        small = QHBoxLayout()
        small.addWidget(self.img_vehicle)
        small.addWidget(self.img_plate)

        right = QVBoxLayout()
        right.addWidget(QLabel("Danh sách vi phạm"))
        right.addWidget(self.table, 1)
        right.addWidget(self.img_scene)
        right.addLayout(small)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(450)
        root.addWidget(right_w)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # ---- kết nối
        self.btn_file.clicked.connect(self.open_file)
        self.btn_cam.clicked.connect(lambda: self.set_source(0))
        self.btn_rtsp.clicked.connect(self.open_rtsp)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_line.clicked.connect(lambda: self.begin_pick("line"))
        self.btn_roi.clicked.connect(lambda: self.begin_pick("roi"))
        self.chk_flip.toggled.connect(self.on_flip)
        self.video.clicked.connect(self.on_click)
        self.table.currentCellChanged.connect(self.on_select)

        for rec in reversed(self.storage.recent(200)):
            self.add_record(rec)
        self.statusBar().showMessage("Sẵn sàng.")

    @staticmethod
    def _preview(text, w, h):
        lb = QLabel(text)
        lb.setFixedSize(w, h)
        lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet("background:#222; color:#888;")
        return lb

    # ------------------------------------------------------------ nguồn video
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video (*.mp4 *.avi *.mkv *.mov);;Tất cả (*.*)")
        if path:
            self.set_source(path)

    def open_rtsp(self):
        url, ok = QInputDialog.getText(self, "Camera IP", "Địa chỉ RTSP/HTTP:")
        if ok and url.strip():
            self.set_source(url.strip())

    def set_source(self, src):
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            QMessageBox.warning(self, "Lỗi", "Không đọc được khung hình từ nguồn này.")
            return
        self.source, self.base_frame = src, frame
        self.refresh_setup_view()
        self.statusBar().showMessage(
            "Đã chọn nguồn. Đặt vạch dừng + vùng đèn (nếu cần) rồi bấm Bắt đầu.")

    # ------------------------------------------------------------ cấu hình vùng
    def refresh_setup_view(self):
        if self.base_frame is None or self.worker is not None:
            return
        img = self.base_frame.copy()
        draw_zones(img, self.cfg, "unknown")
        for p in self.pick_pts:
            cv2.circle(img, p, 6, (255, 0, 255), -1)
        self.video.show_frame(img)

    def begin_pick(self, mode):
        if self.base_frame is None and self.worker is None:
            QMessageBox.information(self, "Chưa có video", "Hãy chọn nguồn video trước.")
            return
        self.pick_mode, self.pick_pts = mode, []
        what = "vạch dừng (2 đầu của vạch)" if mode == "line" else "vùng đèn (2 góc đối diện)"
        self.statusBar().showMessage(f"Click 2 điểm trên video để đặt {what}.")

    def on_click(self, x, y):
        if self.pick_mode is None:
            return
        self.pick_pts.append((x, y))
        if len(self.pick_pts) < 2:
            self.refresh_setup_view()
            self.statusBar().showMessage("Đã nhận điểm 1/2, click điểm còn lại.")
            return
        (x1, y1), (x2, y2) = self.pick_pts
        if self.pick_mode == "line":
            self.cfg.stop_line = [[x1, y1], [x2, y2]]
        else:
            self.cfg.light_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        self.cfg.save()
        self.pick_mode, self.pick_pts = None, []
        self.statusBar().showMessage("Đã lưu cấu hình. Mũi tên trên vạch là hướng xe bị coi là vượt.")
        self.refresh_setup_view()

    def on_flip(self, checked):
        self.cfg.flip_direction = checked
        self.cfg.save()
        self.refresh_setup_view()

    # ------------------------------------------------------------ chạy / dừng
    def start(self):
        if self.worker is not None:
            return
        if self.source is None:
            QMessageBox.information(self, "Chưa có video", "Hãy chọn nguồn video trước.")
            return
        self.worker = VideoWorker(self.source, self.cfg, self.storage)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.violation.connect(self.on_violation)
        self.worker.message.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop(self):
        if self.worker is not None:
            self.worker.stop()

    def on_finished(self):
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def on_frame(self, frame, light):
        self.video.show_frame(frame)
        self.lbl_light.setText(f"Đèn: {LIGHT_VI.get(light, light)}")

    # ------------------------------------------------------------ danh sách vi phạm
    def on_violation(self, rec):
        self.add_record(rec, at_top=True)
        self.table.selectRow(0)

    def add_record(self, rec, at_top=True):
        row = 0 if at_top else self.table.rowCount()
        self.table.insertRow(row)
        self.records.insert(row, rec)
        vals = [rec["id"], rec["created_at"], VTYPE_VI.get(rec["vtype"], rec["vtype"]),
                rec["track_id"], rec["plate_text"] or "—"]
        for col, v in enumerate(vals):
            self.table.setItem(row, col, QTableWidgetItem(str(v)))

    def on_select(self, row, *_):
        if not (0 <= row < len(self.records)):
            return
        rec = self.records[row]
        for lb, key in ((self.img_scene, "scene_path"), (self.img_vehicle, "vehicle_path"),
                        (self.img_plate, "plate_path")):
            pm = load_pixmap(rec[key], lb.width(), lb.height())
            if pm.isNull():
                lb.setPixmap(QPixmap())
                lb.setText("(không có ảnh)")
            else:
                lb.setPixmap(pm)

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 820)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
