# usseg_startracker
Software module for CubeSat/Drone star tracker by USSEG

---

## 1. Python AttitudeDeterminator Package
Thư mục **`src/AttitudeDeterminator/`** chứa gói xử lý thái độ (Attitude Determination) được refactor hoàn toàn sang Python từ kịch bản C++ gốc kết hợp các thuật toán tiên tiến.

### Các thuật toán được hỗ trợ:
*   **TRIAD Estimator (`TRIADEstimator`):** Thuật toán xác định thái độ tĩnh từ 2 quan sát vector (Active Rotation, $Body \rightarrow Inertial$).
*   **QUEST Estimator (`QUESTEstimator`):** Giải toán tối ưu Wahba bằng phương pháp đa thức đặc trưng Shuster (Passive Rotation, $Inertial \rightarrow Body$).
*   **Davenport Q Estimator (`DavenportQEstimator`):** Giải tối ưu bằng cách tìm trị riêng lớn nhất của ma trận $K$ 4x4 (Passive Rotation, $Inertial \rightarrow Body$), khắc phục nhược điểm kỳ dị của QUEST tại góc xoay $180^\circ$.
*   **MEKF Filter (`MEKFEstimator`):** Bộ lọc Kalman mở rộng nhân tính (6 trạng thái: 3 góc sai số thái độ, 3 sai số gyro bias). Hỗ trợ dự báo tần số cao bằng Gyro và cập nhật đo lường tuần tự bằng thuật toán **Murrell's Scheme** giúp tiết kiệm tài nguyên nhúng.

### Yêu cầu thư viện:
*   **`numpy`** (chỉ yêu cầu duy nhất thư viện NumPy để tính toán ma trận hiệu năng cao).

### Cách chạy kiểm thử Python:
Bạn có thể chạy kiểm thử toán học toàn bộ các bộ giải tĩnh và bộ lọc động MEKF thông qua dữ liệu chòm sao Crux thực tế bằng lệnh:
```bash
# Cài đặt thư viện nếu chưa có
pip install numpy

# Thực thi kịch bản test
python test/test_attitude_determinator.py
```

---

## 2. Dự án C++ gốc
Dành cho việc biên dịch và chạy các kiểm thử gốc bằng ngôn ngữ C++:

### Hướng dẫn biên dịch:
- **Bước 1:** Tạo cơ sở dữ liệu đồng bộ (Yêu cầu file `default_database.npz` trong thư mục `data/`):
```bash
python generate_kvec_db.py
```
- **Bước 2:** Build code C++ bằng CMake:
```bash
cd build
cmake ..
cmake --build .
```
- **Bước 3:** Chạy thử nghiệm LIS (chạy từ thư mục root của project):
```bash
cd ..
.\build\Debug\test_lis.exe
```
- **Bước 4:** Chạy thử nghiệm Tracking Mode:
```bash
.\build\Debug\test_tracking.exe
```