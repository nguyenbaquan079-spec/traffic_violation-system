"""Tải 2 model mẫu từ Hugging Face vào models/ và tự cấu hình config.json.

    python download_models.py                  # tải + cấu hình
    python download_models.py --test anh.jpg   # chạy thử 2 model trên 1 ảnh -> test_output.jpg

Lưu ý: file .pt là pickle của PyTorch. Chỉ tải từ nguồn bạn tin cậy và nên chạy trong môi trường ảo riêng.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from config import Settings
from rules import normalize_name

SOURCES = {
    # Model nhận diện người đi xe máy có/không đội mũ (YOLOv8)
    "helmet": {"repo": "FatimaNoorAI/helmet-detection", "file": "safetyHelmet.pt",
               "dest": "models/helmet.pt"},
    # Model phát hiện biển số, 1 lớp (YOLOv8n, MIT)
    "plate": {"repo": "Koushim/yolov8-license-plate-detection", "file": "best.pt",
              "dest": "models/plate.pt"},
}

NEGATIVE_WORDS = {"no", "non", "without", "nohelmet", "nonhelmet", "withouthelmet"}


def pick_no_helmet_names(names) -> list:
    """Chọn các tên lớp mang nghĩa 'không đội mũ' (No Helmet, without_helmet, non-Helmet-...)."""
    out = []
    for n in names:
        s = normalize_name(n)
        tokens = set(re.findall(r"[a-z]+", s))
        if "helmet" in s and tokens & NEGATIVE_WORDS:
            out.append(str(n))
    return out


def fetch(kind: str) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    src = SOURCES[kind]
    files = list_repo_files(src["repo"])
    fname = src["file"] if src["file"] in files else next((f for f in files if f.endswith(".pt")), None)
    if fname is None:
        raise RuntimeError(f"Repo {src['repo']} không có file .pt (các file: {files[:10]})")
    local = hf_hub_download(src["repo"], fname)
    dest = Path(src["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, dest)
    print(f"[ok] {kind}: {src['repo']}/{fname} -> {dest}")
    return dest


def run_test(image_path: str, helmet, plate, cfg: Settings):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {image_path}")
    out = img.copy()
    if helmet is not None:
        out = helmet.predict(img, conf=cfg.conf_helmet, verbose=False)[0].plot()
    if plate is not None:
        r = plate.predict(img, conf=cfg.conf_plate, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
    ok, buf = cv2.imencode(".jpg", out)
    if ok:
        buf.tofile("test_output.jpg")
        print("Đã lưu test_output.jpg (khung vàng = biển số).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", metavar="ẢNH", help="chạy thử 2 model trên 1 ảnh")
    args = ap.parse_args()

    cfg = Settings.load()
    dests = {}
    for kind in ("helmet", "plate"):
        try:
            dests[kind] = fetch(kind)
        except Exception as e:
            print(f"[!] Không tải được model {kind}: {e}")

    from ultralytics import YOLO

    helmet = plate = None
    if "helmet" in dests:
        helmet = YOLO(str(dests["helmet"]))
        names = list(helmet.names.values())
        print(f"Các lớp của model mũ bảo hiểm: {names}")
        picked = pick_no_helmet_names(names)
        if picked:
            cfg.no_helmet_names = picked
            print(f"[ok] Đặt no_helmet_names = {picked}")
        else:
            print("[!] Không tự nhận ra lớp 'không đội mũ'. Hãy sửa no_helmet_names trong config.json "
                  "cho khớp một trong các lớp ở trên.")
        cfg.helmet_model = str(dests["helmet"])
    if "plate" in dests:
        plate = YOLO(str(dests["plate"]))
        print(f"Các lớp của model biển số: {list(plate.names.values())}")
        cfg.plate_model = str(dests["plate"])

    cfg.save()
    print("Đã lưu config.json.")
    if args.test:
        run_test(args.test, helmet, plate, cfg)


if __name__ == "__main__":
    main()
