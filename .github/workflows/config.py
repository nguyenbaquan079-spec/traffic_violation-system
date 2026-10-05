"""Cấu hình hệ thống, lưu/đọc từ config.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

CONFIG_PATH = Path("config.json")


@dataclass
class Settings:
    # --- Model ---
    vehicle_model: str = "yolo11n.pt"        # model COCO có sẵn, tự tải về lần đầu
    helmet_model: str = "models/helmet.pt"   # model tự tải (Roboflow/Hugging Face) hoặc tự huấn luyện
    plate_model: str = "models/plate.pt"     # model phát hiện biển số
    device: str = ""                         # "" = tự chọn, "cpu", "0" (GPU đầu tiên)
    imgsz: int = 640

    # --- Ngưỡng tin cậy ---
    conf_vehicle: float = 0.4
    conf_helmet: float = 0.4
    conf_plate: float = 0.3

    # --- Mũ bảo hiểm ---
    # Tên lớp (trong model helmet) được coi là "không đội mũ". Không phân biệt hoa/thường, "_" và "-".
    no_helmet_names: List[str] = field(
        default_factory=lambda: ["no helmet", "without helmet", "nohelmet", "withouthelmet"]
    )
    helmet_confirm_frames: int = 3   # số lần phát hiện liên tiếp để xác nhận (giảm báo nhầm)
    helmet_every: int = 2            # chạy model mũ bảo hiểm mỗi N khung hình (tiết kiệm CPU)

    # --- Vượt đèn đỏ ---
    stop_line: Optional[List[List[int]]] = None   # [[x1, y1], [x2, y2]] theo toạ độ ảnh
    light_roi: Optional[List[int]] = None         # [x1, y1, x2, y2] vùng chứa đèn tín hiệu
    flip_direction: bool = False                  # đảo chiều xe vượt vạch

    # --- Biển số & lưu trữ ---
    use_ocr: bool = True
    evidence_dir: str = "evidence"
    db_path: str = "violations.db"

    def save(self, path: Path = CONFIG_PATH) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        s = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        return s
