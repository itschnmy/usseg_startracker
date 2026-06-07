# usseg_startracker
Software module for CubeSat/Drone star tracker by USSEG

## Chạy thử nghiệm
- Bước 1: Tạo cơ sở dữ liệu đồng bộ (Yêu cầu file default_database.npz trong thư mục `data/`)
```bash
python generate_kvec_db.py
```
- Bước 2: Build code C++
```bash
cd build
cmake ..
cmake --build .
```
- Bước 3: Chạy thử nghiệm LIS (chạy từ thư mục root của project)
```bash
cd ..
.\build\Debug\test_lis.exe
```
- Bước 4: Chạy thử nghiệm Tracking Mode
```bash
.\build\Debug\test_tracking.exe
```