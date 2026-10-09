# Hệ thống phát hiện vi phạm giao thông (YOLO + OpenCV + PySide6)

> 🇬🇧 Bản tiếng Anh (README chính của repo): [../README.md](../README.md)

Phát hiện **vượt đèn đỏ** và **không đội mũ bảo hiểm**, tự chụp ảnh toàn cảnh, ảnh phương tiện và ảnh/biển số,
lưu vào thư mục `evidence/` và nhật ký SQLite `violations.db`.

**Vị trí mã nguồn:** toàn bộ mã Python nằm ở **thư mục gốc của repo**. Mọi lệnh dưới đây chạy từ
thư mục gốc (nơi có `config.json`).

## Cài đặt
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Nếu trên Linux gặp lỗi `Could not load the Qt platform plugin "xcb"` (xung đột OpenCV với PySide6):
```bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless
```

## Chuẩn bị 2 model (đặt trong thư mục `models/`)
Cách nhanh nhất: script tự tải 2 model mẫu từ Hugging Face và tự điền `config.json`:
```bash
python download_models.py                 # tải models/helmet.pt, models/plate.pt + cấu hình
python download_models.py --test anh.jpg  # chạy thử trên 1 ảnh -> test_output.jpg
```
| File | Nguồn mẫu | Ghi chú |
|---|---|---|
| `models/helmet.pt` | `FatimaNoorAI/helmet-detection` (`safetyHelmet.pt`, YOLOv8) | Người đi xe máy có/không đội mũ. Model card sơ sài, không có số liệu đánh giá/giấy phép: chỉ để thử |
| `models/plate.pt` | `Koushim/yolov8-license-plate-detection` (`best.pt`, YOLOv8n, MIT) | 1 lớp `license_plate`. Chưa rõ dữ liệu huấn luyện |

Mỗi model helmet đặt tên lớp khác nhau; script tự nhận lớp "không đội mũ" và ghi vào `no_helmet_names`.
Nếu không nhận ra, xem tên lớp rồi sửa tay trong `config.json`:
```bash
python -c "from ultralytics import YOLO; print(YOLO('models/helmet.pt').names)"
```
Bạn có thể thay bằng model khác (chỉ cần là YOLO Ultralytics `.pt`). Muốn chính xác với video của bạn thì nên
fine-tune trên ảnh giao thông thực tế. Chỉ tải file `.pt` từ nguồn đáng tin.

## Chạy
```bash
python main_gui.py
```
1. Chọn nguồn: **Mở video**, **Webcam** hoặc **Camera IP** (RTSP).
2. **Chọn vùng đèn**: click 2 góc đối diện quanh đèn tín hiệu (vùng càng sát bóng đèn càng tốt).
3. **Đặt vạch dừng**: click 2 đầu vạch trên mặt đường. **Mũi tên** trên vạch là hướng xe bị coi là vượt;
   nếu ngược thì tick **Đảo chiều vạch**.
4. Bấm **Bắt đầu**. Chọn một dòng trong bảng để xem ảnh toàn cảnh, ảnh xe, ảnh biển số.

## Chạy thử trên video không cần giao diện
```bash
python run_video.py video.mp4 --out ketqua.mp4
```
Xuất video đã vẽ khung/vạch/đèn và in tóm tắt (số khung theo màu đèn, danh sách vi phạm, FPS).
Dùng `stop_line` và `light_roi` đã đặt bằng `main_gui.py` (lưu trong `config.json`).

## Cách hoạt động
- **Xe:** YOLO (COCO) + ByteTrack gán ID cho car/motorcycle/bus/truck.
- **Vượt đèn đỏ:** màu đèn xác định bằng HSV trong vùng đèn (bỏ phiếu đa số qua 7 khung). Khi đèn đỏ và điểm chạm
  đất của xe cắt vạch dừng theo đúng hướng thì ghi nhận (mỗi xe một lần).
- **Không mũ:** model helmet phát hiện "không đội mũ"; gán cho xe máy có hộp (mở rộng lên trên) chứa đầu người đó,
  cần `helmet_confirm_frames` lần liên tiếp mới xác nhận để giảm báo nhầm.
- **Biển số:** model plate chạy trên ảnh cắt của xe vi phạm, EasyOCR đọc ký tự (nếu cài).

## Kiểm thử logic (không cần model thật)
```bash
python -m tests.test_logic
```

## Hạn chế cần biết
- Đây là bản nền tảng/demo: chưa kiểm tra với model và video thật. Hãy thử trên video của chính bạn rồi chỉnh
  ngưỡng `conf_*`, `helmet_confirm_frames` trong `config.json`.
- Biển số chỉ đọc tốt khi đủ lớn và rõ: đặt camera nhìn từ phía có biển số, độ phân giải cao, vạch dừng không quá xa.
- ID tracking có thể đổi khi xe bị che, nên một xe hiếm khi bị ghi nhận 2 lần.
- Không có GPU thì chạy chậm: giảm `imgsz`, tăng `helmet_every`, hoặc dùng video đã ghi.
- Ảnh biển số là dữ liệu cá nhân: cần tuân thủ quy định pháp luật về quyền riêng tư; hệ thống này chỉ hỗ trợ,
  không thay thế thẩm quyền xử phạt.
