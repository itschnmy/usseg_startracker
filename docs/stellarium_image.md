# Ubuntu Linux / dev/version2

For native Ubuntu X11 capture, use the [Ubuntu setup guide](ubuntu.md).
The Windows-specific instructions and historical validation below remain useful
for Windows; Linux uses xdotool and MSS instead of Win32/PyGetWindow.
The Remote Control API, scene preparation and labeling workflow are shared.

# Stellarium image — thu ảnh, tạo nhãn và kiểm tra USSEG

## 1. Nguồn ảnh và phạm vi

`dataset/stellarium_test/stellarium_view.png` được người dùng thu bằng script
`data/extract_star.py`. Script đã chuyển đến
[`tools/stellarium/extract_star.py`](../tools/stellarium/extract_star.py): đây là
công cụ thu dữ liệu, không phải catalog asset nên không còn đặt trong `data/`.
Đường dẫn cũ đã bỏ; cập nhật shortcut/lệnh đang dùng.

Có hai phép thử khác nhau:

| Track | Đầu vào USSEG | Reference dùng được |
|---|---|---|
| Screenshot trực tiếp | `stellarium_view.png` nguyên bản | Nhãn pixel và RA/Dec do công cụ xuất cho chính screenshot |
| Camera chụp màn hình lab | Frame gốc từ camera | Phải chuyển hệ pixel và hiệu chuẩn camera–màn hình; không dùng thẳng pixel nhãn screenshot |

Công cụ thu ảnh dùng **FOV/tâm nhìn đã biết từ API**, Hipparcos và RANSAC để
tạo nhãn. USSEG chạy **blind**: chỉ dùng ảnh, database, FOV/cấu hình hình học;
không nhận tâm RA/Dec, HIP, vị trí sao trong nhãn. Chỉ sau inference mới mở nhãn
để chấm. Nhãn/reference này không phải ground truth vật lý độc lập.

## 2. Chuẩn bị máy thu ảnh

- Stellarium desktop chạy trên **Windows**, không phải Stellarium Web.
- Windows Python **3.10**, chạy trong desktop session có màn hình mở khóa.
- Chạy script thu ảnh trên cùng máy Windows/cùng Stellarium instance mà API
  điều khiển. Không chạy script Win32 capture trong WSL hoặc máy headless.
- Cần Internet để Astroquery truy vấn VizieR Hipparcos và tên sao. Chạy USSEG
  với database có sẵn thì không cần Internet.
- Không cần GPU riêng cho USSEG; Stellarium cần render được cửa sổ bình thường.

Ví dụ PowerShell từ thư mục repo trên Windows:

```powershell
py -3.10 -m venv .venv-capture
.\.venv-capture\Scripts\python.exe -m pip install -r tools/stellarium/requirements.txt
.\.venv-capture\Scripts\python.exe tools/stellarium/extract_star.py --help
```

Dùng trực tiếp `python.exe` tránh phụ thuộc chính sách activate PowerShell.
`requirements.txt` riêng bổ sung `mss`, `pygetwindow`, `requests`, `astropy`,
`astroquery`; không đưa các phụ thuộc Windows này vào pipeline mặc định.

## 3. Bật API trên Stellarium

1. Mở Stellarium, nhấn **F2 → Plugins → Remote Control**.
2. Bật **Load at startup**, khởi động lại Stellarium nếu plugin chưa được nạp.
3. Quay lại plugin, chọn **Configure**, bật **Server enabled**; đặt port **8090**.
   Có thể bật khởi động server tự động nếu dùng lab thường xuyên. Tên mục có
   thể khác nhẹ giữa phiên bản/ngôn ngữ giao diện.
4. Trên cùng máy, mở `http://127.0.0.1:8090/`. Đây là giao diện web điều khiển.
5. Kiểm tra `http://127.0.0.1:8090/api/main/status` trả JSON, không phải HTML lỗi.

PowerShell, khi chưa bật xác thực:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/api/main/status
Invoke-RestMethod http://127.0.0.1:8090/api/main/view
```

Theo [tài liệu plugin chính thức](https://stellarium.org/doc/1.x/remoteControlDoc.html),
cổng mặc định là 8090. [API chính thức](https://stellarium.org/doc/1.x/remoteControlApi.html)
định nghĩa prefix `/api/`, GET trạng thái/hướng nhìn và POST thay đổi trạng thái.
Đây là **Remote Control**, không phải plugin Telescope Control.

### An toàn truy cập

Ưu tiên thu và điều khiển ngay trên máy Windows bằng loopback. Không mở router
port-forward 8090 ra Internet, không tắt Windows Firewall, không cho phép mạng
Public. URL loopback của client **không tự đảm bảo** server chỉ lắng nghe loopback:
kiểm tra cấu hình truy cập của plugin và firewall nếu máy nối LAN.

Nếu bật mật khẩu trong plugin, công cụ hỗ trợ biến môi trường:

```powershell
$secret = Read-Host "Stellarium API password" -AsSecureString
$env:STELLARIUM_API_PASSWORD = [System.Net.NetworkCredential]::new("", $secret).Password
```

API dùng HTTP Basic với username rỗng theo tài liệu chính thức. Không lưu mật
khẩu trong JSON, Git hay command line; HTTP Basic không mã hóa đường truyền,
nên mật khẩu không thay thế cho việc giới hạn mạng. Sau phiên thu:

```powershell
Remove-Item Env:STELLARIUM_API_PASSWORD -ErrorAction SilentlyContinue
```

## 4. Chuẩn bị cảnh và chụp screenshot

Chọn vùng trời có nhiều sao sáng, tránh hành tinh, Mặt Trăng, đường lưới,
crosshair, pointer/selection marker và overlay của ứng dụng khác. Không nên
dùng zoom quá sâu khiến chỉ còn vài sao trong database. Không resize, pan,
zoom hoặc thay đổi thời gian trong lúc thu. Ghi lại phiên bản Stellarium,
độ phân giải client area, OS display scaling và các setting render sao.

Mẫu hiện có dùng perspective, ảnh **1920×1080**, VFOV **16.9761°** (xấp xỉ 17°),
tâm RA **291.95429°**, Dec **41.12827°**. Đây là ví dụ để đối chiếu, không phải
vùng trời bắt buộc. Có thể dùng API đặt FOV trước khi chạy:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8090/api/main/fov -Body @{fov=17}
```

Chọn thư mục **mới**, không ghi đè bộ fixture đang có:

```powershell
.\.venv-capture\Scripts\python.exe tools/stellarium/extract_star.py `
  --api-url http://127.0.0.1:8090/api `
  --output-dir captures/stellarium/session-001
```

Nếu plugin dùng mật khẩu, ví dụ `Invoke-RestMethod` phía trên cần xác thực phù
hợp; công cụ Python tự dùng biến môi trường đã đặt. Không sửa source để đổi port:
chỉ thay `--api-url`, ví dụ `http://127.0.0.1:8091/api`.

Công cụ sẽ:

1. Kiểm tra API; lưu các property/thời gian cần khôi phục.
2. Dừng thời gian, đặt `ProjectionPerspective`, ẩn GUI, tên sao, chòm sao,
   lưới, khí quyển, phong cảnh và một số overlay.
3. Đưa **cửa sổ Stellarium.exe** ra trước, chụp client area bằng Win32/MSS,
   không gồm title bar/viền Windows. Nếu không xác nhận được foreground thì báo lỗi.
4. Đọc tâm J2000/FOV, rồi cố khôi phục GUI/projection/timerate kể cả khi capture lỗi.
   Thời gian mô phỏng đã trôi trước capture không được tua ngược; không phải
   mọi setting hay lỗi tắt cưỡng bức tiến trình đều có thể tự khôi phục.
5. Truy vấn Hipparcos (`Vmag < 6`), tìm blob trên screenshot với threshold
   140/255, khớp góc quay bằng RANSAC rồi gán một-một trong bán kính 3px.
6. Xuất ảnh nguyên bản, nhãn, metadata và ảnh debug có chú thích.

Nó **không** ẩn chắc chắn mọi plugin/overlay bên thứ ba hoặc mọi vật thể phi sao.
Hãy kiểm tra ảnh nguyên bản bằng mắt. Nếu property bắt buộc không tồn tại trên
phiên bản Stellarium đang dùng, công cụ dừng thay vì xuất nhãn giả.

```text
captures/stellarium/session-001/
├── stellarium_view.png          # Input thật cho USSEG
├── stars_labels.json            # Chỉ chấm sau inference
├── stellarium_debug_hip.png     # Debug gán nhãn, KHÔNG đưa vào solver
└── stellarium_debug_names.png   # Debug gán nhãn, KHÔNG đưa vào solver
```

Thư mục output được bảo vệ khỏi ghi đè các tên file này. Không pan/zoom giữa
ảnh và metadata. Nhãn có thể thiếu sao mờ, đôi sát nhau hoặc nguồn sáng chưa
phân giải; ảnh debug đúng về hình thức không chứng minh toàn bộ nhãn đúng.

## 5. FOV và convention của nhãn

- `fov_deg` của collector là **FOV dọc**, được dùng với chiều cao ảnh để dựng
  phép chiếu. Metadata mới bổ sung `fov_axis=vertical`.
- `PipelineConfig.fov_deg` là **FOV ngang**:
  `HFOV = degrees(2*atan(tan(radians(VFOV)/2)*width/height))`.
- Mẫu 1920×1080, VFOV 16.9761° → HFOV **29.717817°**. Không đưa 17° thẳng
  vào tham số HFOV của solver.
- `x,y`: tọa độ zero-based, gốc góc trên trái, X sang phải, Y xuống dưới.
- `ra_deg,dec_deg`: tọa độ catalog J2000; `hip_id` là ID Hipparcos.
- `alignment_matrix`: affine similarity 2×3 từ tangent-plane catalog đến pixel.
  `roll_angle_deg` là góc của phép căn chỉnh này, không được trừ trực tiếp với
  roll Euler do solver xuất vì khác convention.
- `center_ra_deg,center_dec_deg`: hướng nhìn API, không dùng làm hint cho USSEG.
- Metadata mới có nguồn screenshot, timestamp UTC và thời gian Stellarium.
  Bộ fixture cũ có thể chưa chứa các trường bổ sung này.

## 6. Chạy USSEG trên ảnh vừa chụp

Chạy pipeline trên Windows hoặc chuyển **nguyên bộ thư mục** sang Linux/WSL.
Không resize/flip/crop screenshot và vẫn dùng nhãn pixel cũ.
Từ root repo, môi trường đánh giá Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash examples/materialize_assets.sh
```

Trên Windows có thể cài `-e .` bằng interpreter capture ở trên và dùng
`--database C:/duong-dan/default_database.npz`; shell khôi phục asset chạy bằng
Git Bash/WSL hoặc chuyển database đã kiểm tra checksum sang Windows.

Nếu VFOV/độ phân giải khác fixture, tính HFOV mới bằng công thức ở mục 5.
Không sửa preset fixture để hợp với ảnh mới. Ví dụ ảnh mới cũng HFOV ~29.717817°:

```bash
.venv/bin/python -m models.pipeline.cli \
  --config configs/examples/stellarium.json \
  --image captures/stellarium/session-001/stellarium_view.png \
  --fov 29.717817 --output out/stellarium/session-001-result.json
```

Để chấm cùng nhãn sau inference (đổi `--vertical-fov` theo phiên thu):

```bash
.venv/bin/python examples/check_stellarium.py \
  --dataset captures/stellarium/session-001 \
  --database data/default_database.npz --vertical-fov 16.9761 \
  --config configs/examples/stellarium.json \
  --output-dir out/stellarium/session-001-check
```

`--config` phải có HFOV phù hợp **phiên mới**; copy preset sang file riêng rồi
sửa `fov_deg` nếu cần. Database JSON tương đối với vị trí file config. Runner
chấm sẽ chạy baseline và `configured_blind` trước khi đọc nhãn, sau đó chạy
oracle-centroid riêng để chẩn đoán. Không nhầm oracle với end-to-end.

Đọc `status`, `num_matches`, số cặp star-ID đúng theo nhãn, sai số quaternion /
boresight và residual reprojection. Exit code 0 chỉ có nghĩa xử lý hoàn tất,
không đảm bảo `status=ok`. `no_solve` phải được giữ trong báo cáo.
`selected-centroids.png` và phần detector diagnosis là baseline cũ;
`configured-selected-centroids.png` là các điểm từ preset mới. Metric của
preset nằm ở run `configured_blind` trong `results.json`.
Không dùng đúng một ảnh đã tuning để công bố accuracy hoặc FPS của hệ thống.

## 7. Code thay đổi gì để fixture Stellarium chạy được?

| Tham số / logic | Default lịch sử | Preset Stellarium |
|---|---|---|
| FOV | Caller cung cấp HFOV | Đổi VFOV 16.9761° → HFOV 29.717817° |
| Top-Hat | kernel 7×7 | Tắt: tránh triệt lõi sao render lớn |
| Area | 1–50px | 3–500px |
| Centroid, flux và peak | Toàn bounding rectangle | Chỉ pixel thuộc đúng component |
| Vùng biên | Không loại | Loại component chạm band 12px |
| Xếp hạng | Peak rồi flux | Tổng flux rồi peak |
| Max stars / pattern stars | 20 / 12 | 20 / 12 (giữ nguyên) |
| Timeout mềm | 600ms | 5000ms |
| Tetra / DB / default SVD | Baseline | Không đổi thuật toán cho preset này |

Lý do: Top-Hat 7×7 trên ảnh render này làm nhiều lõi sao sáng xuống dưới
ngưỡng; top-20 cũ không khớp sao trong danh sách nhãn. Config mới là opt-in,
không thay default benchmark cũ. Code chính:
[`star_detector.py`](../models/detector/star_detector.py),
[`pipeline.py`](../models/pipeline/pipeline.py),
[`stellarium.json`](../configs/examples/stellarium.json).

Trên fixture hiện tại: **16 matches, 16/16 cặp đúng**, full attitude error
**0.002738°**, boresight error **0.002289°** so với reference nhãn; 19/20
selected detections gần nhãn trong 3px. Đây là kết quả development trên ảnh
đã tuning. Reference fit có P50/P95 0.140/0.386px; không phải attitude ground
truth độc lập. Với cảnh/độ phân giải/render setting khác, kết quả có thể khác.

## 8. Camera chụp màn hình trong lab

Collector không tự điều khiển camera. Để giữ một cảnh sạch trong lúc người
dùng chụp bằng phần mềm camera của mình:

```powershell
.\.venv-capture\Scripts\python.exe tools/stellarium/extract_star.py `
  --output-dir captures/stellarium/lab-session-001 --hold-for-camera
```

Sau khi cảnh đã dừng và GUI đã ẩn, chụp camera; không pan/zoom/resize hay
che màn hình. Sau khi chụp xong, trở về terminal và nhấn Enter. Công cụ focus
lại Stellarium, thu screenshot tương ứng rồi khôi phục GUI. Nhãn xuất ra vẫn
**chỉ thuộc screenshot**, không tự thuộc frame camera.

Checklist lab:

1. Màn hình và camera cố định; ghi kích thước **vùng render** theo cm, khoảng
   cách đến camera, góc nghiêng; không lấy bề ngang vỏ màn hình làm vùng ảnh.
2. Tắt auto exposure/gain/white balance/focus nếu camera hỗ trợ; ghi thông số,
   sensor resolution, binning/crop, lens/focal length. Tránh saturation, moiré,
   phản xạ, rung và flicker do refresh màn hình/exposure không phù hợp.
3. Hiệu chuẩn camera intrinsics/distortion và quan hệ mặt phẳng màn hình–camera.
   Với camera vuông góc, màn hình nằm giữa và bề ngang vùng hiển thị W ở khoảng
   cách d: `2*atan(W/(2*d))` là góc màn hình chiếm trong camera, **không tự động
   là FOV thiên cầu Stellarium hay full-sensor HFOV**. Setup xiên cần phép chiếu
   đầy đủ, không chỉ một hệ số scale.
4. Giữ frame gốc, screenshot, nhãn và metadata cùng session ID; ghi timestamp.
   Thu dark frame nếu muốn nghiên cứu trừ nền; thuật toán hiện tại không tự
   thực hiện dark-frame/flat-field calibration.
5. Chạy raw camera frame bằng cấu hình HFOV/calibration phù hợp. Đừng dùng
   mặc định HFOV 29.717817° của screenshot cho frame camera. Nếu rectification,
   lưu transform/output dimensions và tách raw-track với corrected-track.
6. Chỉ chấm centroid/star-ID sau khi đã chuyển nhãn screenshot sang pixel camera
   bằng phép biến đổi được kiểm tra. Hướng nhìn camera vật lý và hướng thiên cầu
   ảo là hai đại lượng khác nhau trong setup màn hình; không lấy quaternion
   screenshot làm ground truth camera/body khi thiếu mô hình và mounting transform.

Làm việc theo session: vài cảnh dùng phát triển config, các cảnh khác giữ lại
để test; không chọn config tốt nhất bằng nhãn test. Chạy thêm nhiều vị trí trời,
FOV, brightness/exposure và lặp lại để đo robustness, không chỉ một ảnh đẹp.

## 9. Troubleshooting

| Triệu chứng | Kiểm tra |
|---|---|
| Connection refused / timeout | Plugin đã load sau restart? Server enabled? Port/URL có `/api`? Đúng instance? |
| HTTP 401 | Password plugin và `STELLARIUM_API_PASSWORD`; Basic username rỗng |
| Không tìm thấy window / foreground | Windows Python, desktop mở khóa, đúng `stellarium.exe`; không WSL/RDP đã disconnect |
| Unsupported property | Phiên bản/plugin khác schema; kiểm tra `/api/stelproperty/list`; không bỏ kiểm tra để ép chạy |
| Không có blob / alignment fail | Vùng quá ít sao sáng, saturation/render khác; threshold collector 140 khác detector USSEG |
| VizieR timeout | Internet/DNS; không thay nhãn bằng guessed HIP/RA/Dec |
| File đã có | Tạo session output mới; công cụ không ghi đè fixture |
| Detector tốt nhưng no_solve | Đúng HFOV ngang, DB generation range, projection/size, centroid ranking? Xem oracle riêng |
| Screenshot pass, camera fail | Camera/screen calibration, distortion, FOV hiệu dụng, moiré và PSF khác screenshot |

## 10. Xác minh và giới hạn

Ngày 2026-10-25 đã chạy collector thật trên **Windows Stellarium 26.3** thông
qua WSL interoperability và Windows Python 3.10.19. API loopback Windows hoạt
động; localhost riêng của WSL không kết nối được. Không mở thêm firewall/port
forward. Môi trường capture riêng không sửa các Conda env có sẵn.

Session `captures/stellarium/live-20261025-001`: ảnh 1920×1080, 40 nhãn,
VFOV 16.9767°, HFOV 29.718835°. Giữ nguyên preset preprocess, chỉ cập nhật
HFOV của session: blind pipeline trả **19 matches**, trong đó **18/18 cặp có
nhãn đối chiếu đúng**, một cặp chưa có label gần tương ứng để chấm. Full
attitude error so với reference nhãn **0.010914°**, boresight **0.010855°**.
GUI/projection/timerate đã được kiểm tra khôi phục sau capture.

Lần chạy thật cũng phát hiện console Windows cp1252 không in được tiếng Việt;
collector nay tự dùng stdout/stderr UTF-8. Báo cáo local tại
`out/stellarium/live-20261025-001/live-run.md`; ảnh/session sinh ra được Git ignore.

Đây là screenshot trực tiếp, **chưa phải phép thử camera vật lý chụp màn hình**.
Không suy rộng hai ảnh screenshot thành độ chính xác toàn hệ thống. Người vận
hành vẫn cần kiểm tra lại ảnh/nhãn và calibration khi đổi máy, render setting
hoặc camera lab.

Nguồn chính thức: [Remote Control plugin](https://stellarium.org/doc/1.x/remoteControlDoc.html),
[HTTP API](https://stellarium.org/doc/1.x/remoteControlApi.html),
[hướng dẫn bật plugin từ maintainer](https://github.com/Stellarium/stellarium/discussions/3760).
