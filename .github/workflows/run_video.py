"""Chạy hệ thống trên 1 video không cần giao diện: xuất video đã chú thích + bản tóm tắt.

    python run_video.py video.mp4                          # -> output_video.mp4
    python run_video.py video.mp4 --out kq.mp4 --max-frames 600

Dùng stop_line / light_roi đã đặt trong config.json (đặt bằng giao diện: main_gui.py).
Bản tóm tắt cho biết số khung hình theo từng màu đèn, giúp kiểm tra vùng đèn có đúng không.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import cv2

from config import Settings
from pipeline import VTYPE_EN, ViolationSystem
from storage import Storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="đường dẫn video")
    ap.add_argument("--out", default="output_video.mp4", help="video kết quả")
    ap.add_argument("--max-frames", type=int, default=0, help="chỉ xử lý N khung đầu (0 = hết)")
    args = ap.parse_args()

    cfg = Settings.load()
    storage = Storage(cfg.db_path, cfg.evidence_dir)
    system = ViolationSystem(cfg, storage)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not cfg.stop_line:
        print("[!] Chưa đặt vạch dừng: sẽ không phát hiện vượt đèn đỏ (đặt bằng main_gui.py).")
    if not cfg.light_roi:
        print("[!] Chưa đặt vùng đèn: sẽ không biết đèn đỏ hay xanh.")

    light_frames, found = Counter(), Counter()
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, records = system.process(frame)
        writer.write(annotated)
        light_frames[system.last_light] += 1
        for r in records:
            found[r["vtype"]] += 1
            t = f"{n / fps:6.1f}s"
            print(f"  [{t}] khung {n}: {VTYPE_EN[r['vtype']]} — xe #{r['track_id']} — "
                  f"biển số: {r['plate_text'] or '(chưa đọc được)'}")
        n += 1
        if n % 100 == 0:
            print(f"... {n} khung, {n / (time.time() - t0):.1f} FPS")
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    writer.release()
    dt = max(time.time() - t0, 1e-6)
    print("\n===== TÓM TẮT =====")
    print(f"Đã xử lý {n} khung trong {dt:.0f}s ({n / dt:.1f} FPS) -> {args.out}")
    print("Số khung theo màu đèn:", dict(light_frames))
    print("Vi phạm:", dict(found) or "không có")
    print(f"Ảnh bằng chứng trong '{cfg.evidence_dir}/', nhật ký trong '{cfg.db_path}'.")


if __name__ == "__main__":
    main()
