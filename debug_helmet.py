"""Chẩn đoán vì sao hệ thống không báo 'không đội mũ bảo hiểm'.

    python debug_helmet.py video.mp4
    python debug_helmet.py video.mp4 --conf 0.1 --imgsz 1280 --max-frames 600 --out debug_helmet.mp4

Script chạy từng bước của luật mũ bảo hiểm (xe máy -> model mũ -> gán người cho xe -> xác nhận nhiều khung),
in ra bước nào bị tắc và xuất video để bạn nhìn tận mắt:
    khung xanh dương = xe máy (kèm số ID), khung cyan mỏng = vùng tìm đầu người lái,
    đỏ = 'không đội mũ' đủ ngưỡng, cam = 'không đội mũ' nhưng dưới ngưỡng conf_helmet, xanh lá = lớp khác.
Chạy từ đúng thư mục dự án (nơi có config.json).
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from config import Settings
from pipeline import MOTORCYCLE, VEHICLE_CLASSES
from rules import box_center, expand_box, normalize_name, point_in_box

FONT = cv2.FONT_HERSHEY_SIMPLEX


def associate(center, motos, w, h):
    """Giống pipeline: gán hộp 'không đội mũ' cho xe máy gần nhất có vùng mở rộng chứa nó."""
    best, best_d = None, None
    for tid, box in motos:
        if point_in_box(center, expand_box(box, w, h)):
            mc = box_center(box)
            d = math.hypot(center[0] - mc[0], center[1] - mc[1])
            if best is None or d < best_d:
                best, best_d = tid, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--conf", type=float, default=0.1, help="ngưỡng thấp để xem mọi phát hiện của model mũ")
    ap.add_argument("--imgsz", type=int, default=0, help="kích thước ảnh cho model mũ (0 = theo config)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="debug_helmet.mp4")
    args = ap.parse_args()

    cfg = Settings.load()
    hp = Path(cfg.helmet_model)
    print(f"Thư mục đang chạy: {Path.cwd()}")
    print(f"helmet_model = {cfg.helmet_model} -> {hp.resolve()}  [{'CÓ' if hp.exists() else 'KHÔNG TỒN TẠI'}]")
    if not hp.exists():
        print("=> NGUYÊN NHÂN: không tìm thấy file model mũ bảo hiểm nên luật này bị TẮT.\n"
              "   Chạy lại từ đúng thư mục dự án, hoặc sửa 'helmet_model' trong config.json.")
        return

    from ultralytics import YOLO

    helmet, detector = YOLO(str(hp)), YOLO(cfg.vehicle_model)
    names = helmet.names
    targets = {normalize_name(n) for n in cfg.no_helmet_names}
    print(f"Các lớp của model mũ: {dict(names)}")
    print(f"no_helmet_names trong config: {cfg.no_helmet_names}")
    if not [n for n in names.values() if normalize_name(n) in targets]:
        print("=> NGUYÊN NHÂN CÓ THỂ: không lớp nào của model khớp no_helmet_names. Sửa config.json.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    dev = {"device": cfg.device} if cfg.device else {}
    imgsz = args.imgsz or cfg.imgsz

    n = frames_moto = frames_det = 0
    cls_count, cls_max = Counter(), defaultdict(float)
    tgt_all = tgt_strong = tgt_assoc = 0
    cur, best_streak, hits = Counter(), Counter(), Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        r = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES,
                           conf=cfg.conf_vehicle, imgsz=cfg.imgsz, verbose=False, **dev)[0]
        motos, b = [], r.boxes
        if b is not None and len(b) and b.id is not None:
            for box, c, i in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                 b.id.cpu().numpy().astype(int)):
                if c == MOTORCYCLE:
                    motos.append((int(i), tuple(float(v) for v in box)))
        frames_moto += bool(motos)

        hr = helmet.predict(frame, conf=args.conf, imgsz=imgsz, verbose=False, **dev)[0]
        dets, hb = [], hr.boxes
        if hb is not None and len(hb):
            for box, c, p in zip(hb.xyxy.cpu().numpy(), hb.cls.cpu().numpy().astype(int),
                                 hb.conf.cpu().numpy()):
                dets.append((str(names[int(c)]), float(p), tuple(float(v) for v in box)))
        frames_det += bool(dets)

        img = frame.copy()
        for tid, box in motos:
            x1, y1, x2, y2 = (int(v) for v in box)
            ex1, ey1, ex2, ey2 = expand_box(box, w, h)
            cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"moto #{tid}", (x1, max(15, y1 - 6)), FONT, 0.55, (255, 0, 0), 2)

        hit_ids = set()
        for name, p, box in dets:
            cls_count[name] += 1
            cls_max[name] = max(cls_max[name], p)
            is_t = normalize_name(name) in targets
            strong = is_t and p >= cfg.conf_helmet
            color = (0, 0, 255) if strong else ((0, 165, 255) if is_t else (0, 200, 0))
            if is_t:
                tgt_all += 1
                tgt_strong += strong
                tid = associate(box_center(box), motos, w, h) if strong else None
                if tid is not None:
                    tgt_assoc += 1
                    hit_ids.add(tid)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {p:.2f}", (x1, max(15, y2 + 16)), FONT, 0.5, color, 2)

        for tid, _ in motos:
            if tid in hit_ids:
                cur[tid] += 1
                hits[tid] += 1
                best_streak[tid] = max(best_streak[tid], cur[tid])
            else:
                cur[tid] = 0
        writer.write(img)
        if n % 100 == 0:
            print(f"... {n} khung")
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    writer.release()
    need = cfg.helmet_confirm_frames * max(1, cfg.helmet_every)
    longest = max(best_streak.values(), default=0)

    print("\n===== CHẨN ĐOÁN MŨ BẢO HIỂM =====")
    print(f"Khung xử lý: {n} | có xe máy: {frames_moto} | model mũ có phát hiện gì đó: {frames_det}")
    print("Lớp model mũ đã phát hiện (số lần, conf cao nhất):",
          {k: (cls_count[k], round(cls_max[k], 2)) for k in cls_count} or "không có")
    print(f"Lớp 'không đội mũ': tổng {tgt_all} | đạt conf_helmet={cfg.conf_helmet}: {tgt_strong} | "
          f"gán được cho xe máy: {tgt_assoc}")
    print(f"Chuỗi khung liên tiếp dài nhất trên 1 xe: {longest} (cần khoảng ≥ {need} = "
          f"helmet_confirm_frames {cfg.helmet_confirm_frames} × helmet_every {cfg.helmet_every})")

    print("\n===== GỢI Ý =====")
    if frames_moto == 0:
        print("- Không thấy xe máy nào (COCO 'motorcycle'): luật mũ cần xe máy được theo dõi. "
              "Thử giảm conf_vehicle hoặc xem video debug xem xe máy bị nhận thành loại khác.")
    if frames_det == 0:
        print("- Model mũ không phát hiện gì ngay cả ở ngưỡng thấp: người lái quá nhỏ trong khung hình. "
              "Thử --imgsz 1280, dùng video gần hơn/độ phân giải cao hơn.")
    elif tgt_all == 0:
        print("- Model có phát hiện nhưng không lớp nào khớp no_helmet_names: sửa config.json cho đúng tên lớp.")
    elif tgt_strong == 0:
        print(f"- 'Không đội mũ' chỉ xuất hiện dưới ngưỡng. Giảm conf_helmet (conf cao nhất đã thấy: "
              f"{max(cls_max[k] for k in cls_count if normalize_name(k) in targets):.2f}).")
    elif tgt_assoc == 0:
        print("- Có 'không đội mũ' đủ ngưỡng nhưng không gán được cho xe máy nào: xe máy chưa được theo dõi "
              "hoặc đầu người nằm ngoài vùng cyan. Xem video debug để thấy lệch ở đâu.")
    elif longest < need:
        print("- Phát hiện chập chờn, không đủ liên tiếp để xác nhận. Thử đặt helmet_confirm_frames=2 "
              "và helmet_every=1 trong config.json.")
    else:
        print("- Các điều kiện đều đủ để báo vi phạm. Nếu hệ thống chính vẫn không báo, kiểm tra ID xe có đổi "
              "liên tục không (xem video debug) và chạy lại run_video.py.")
    print(f"\nVideo kiểm tra: {args.out}")


if __name__ == "__main__":
    main()
