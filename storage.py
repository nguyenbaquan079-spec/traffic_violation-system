"""Lưu ảnh bằng chứng ra đĩa và nhật ký vi phạm vào SQLite."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2


def save_image(path: Path, img) -> bool:
    """Ghi JPEG bằng imencode để an toàn với đường dẫn có ký tự Unicode (Windows)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(path))
    return bool(ok)


class Storage:
    def __init__(self, db_path: str, evidence_dir: str):
        self.db_path = str(db_path)
        self.root = Path(evidence_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS violations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT, vtype TEXT, track_id INTEGER, plate_text TEXT,
                    scene_path TEXT, vehicle_path TEXT, plate_path TEXT)"""
            )
            c.commit()

    def add(self, vtype, track_id, plate_text, scene, vehicle, plate) -> dict:
        now = datetime.now()
        day = self.root / now.strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = f"{now.strftime('%H%M%S_%f')}_{vtype}_{track_id}"

        paths = {}
        for kind, img in (("scene", scene), ("vehicle", vehicle), ("plate", plate)):
            if img is not None and getattr(img, "size", 0):
                p = day / f"{stem}_{kind}.jpg"
                save_image(p, img)
                paths[kind] = str(p)
            else:
                paths[kind] = ""

        created = now.strftime("%Y-%m-%d %H:%M:%S")
        with closing(sqlite3.connect(self.db_path)) as c:
            cur = c.execute(
                "INSERT INTO violations(created_at, vtype, track_id, plate_text,"
                " scene_path, vehicle_path, plate_path) VALUES (?,?,?,?,?,?,?)",
                (created, vtype, int(track_id), plate_text,
                 paths["scene"], paths["vehicle"], paths["plate"]),
            )
            c.commit()
            rid = cur.lastrowid
        return {
            "id": rid, "created_at": created, "vtype": vtype, "track_id": int(track_id),
            "plate_text": plate_text, "scene_path": paths["scene"],
            "vehicle_path": paths["vehicle"], "plate_path": paths["plate"],
        }

    def recent(self, limit: int = 200) -> list:
        with closing(sqlite3.connect(self.db_path)) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
