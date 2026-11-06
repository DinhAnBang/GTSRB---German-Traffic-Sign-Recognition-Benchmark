# GTSRB Traffic Sign Recognition

Ứng dụng Streamlit dùng [models/GTSRB.onnx](models/GTSRB.onnx) để nhận diện 43 loại biển báo giao thông.

## 1. Tạo virtual environment và cài dependency

Mở PowerShell tại thư mục project:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn việc activate script, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

## 2. Chạy giao diện

```powershell
streamlit run app.py
```

Sau đó mở URL Streamlit hiển thị trong terminal, thường là `http://localhost:8501`.

Model mặc định được đọc tại:

```text
models/GTSRB.onnx
```

Không đặt virtual environment hoặc model bên trong nhau: `.venv/` chỉ chứa Python packages, còn model nằm trong `models/` để dễ thay thế và quản lý.

Trong giao diện, thanh **Confidence tối thiểu để nhận diện** là bộ lọc thực sự:

- Chỉ các lớp có confidence lớn hơn hoặc bằng ngưỡng mới được hiển thị.
- Nếu không có lớp nào đạt ngưỡng, hãy kéo thanh này xuống thấp hơn hoặc dùng ảnh rõ hơn.
- Thanh **Số vùng cắt hiển thị** giới hạn số vùng đạt ngưỡng được hiển thị sau khi lọc.

Khi bấm **Nhận diện**, ảnh đầu vào được chia thành 4 vùng `Q1` đến `Q4`. Model phân loại từng vùng riêng; ảnh gốc có 4 khung cắt và 4 ảnh vùng cắt được hiển thị trong mục **Xem ảnh sau tiền xử lý**.

Để kiểm tra đúng/sai cho ảnh mới, mở **Kiểm tra đúng / sai** và chọn nhãn thật của từng vùng. Nếu chưa biết nhãn thật thì chỉ có thể tham khảo prediction và confidence, không thể kết luận chính xác tuyệt đối.

Các khung này là vùng cắt cố định, không phải bounding box tự động. Vì `GTSRB.onnx` là model classification, muốn tìm chính xác vị trí từng biển báo trong ảnh đường phố cần một model detection riêng.

## 3. Validation

Trang **Validation** nhận dataset dạng ImageFolder:

```text
dataset/
├── 0/
├── 1/
├── 2/
└── ...
    └── 42/
```

Mỗi thư mục chứa ảnh của lớp tương ứng. Nhập đường dẫn dataset trong giao diện rồi bấm **Chạy validation**.

## 4. Training

`GTSRB.onnx` là model dùng cho suy luận, không phải checkpoint để train tiếp. Muốn retrain cần dataset và checkpoint `.pt` tương ứng. Dependency training là tùy chọn:

```powershell
python -m pip install -r requirements-train.txt
```

