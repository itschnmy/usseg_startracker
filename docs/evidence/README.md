# Evidence snapshots

Các artifact này được copy từ output thí nghiệm hiện có vào ngày2026-11-04 để
báo cáo có thể đọc từ checkout khác. Không chạy lại benchmark để tạo các số liệu
này trong lượt lập kế hoạch. Source commit/config/image hashes được giữ trong JSON.

| Snapshot | Nguồn tại workspace lúc copy |
|---|---|
| `image03.json` | `out/image-examples/image03.json` |
| `stellarium.json` | `out/image-examples/stellarium.json` |
| `image03-ablation.json` | `out/image03-experimental/results.json` |
| `image200701-raw.json` | `out/image200701-raw/results.json` |
| `image200701-calibrated.json` | `out/image200701-calibrated/results.json` |
| `synthetic-version01.csv` | `../lost-evals/out/version01/synthetic.csv` |
| `dust-version01.md` | `../lost-evals/out/version01/dust-final/dust-usseg-vs-lost-summary.md` |

Các path nguồn trên tính từ repo root. Chúng là provenance, không phải links
bắt buộc tồn tại trên máy nhận release. Không đóng các dataset benchmark lớn
vào container runtime chỉ vì chúng được nhắc tới ở đây.

`dust-version01.md` là nguyên bản báo cáo cũ. Cột Star-ID precision/recall đo
catalog-set overlap, không chứng minh correspondence từng centroid đúng. Các
CSV strata được nhắc trong đó vẫn nằm ở evaluator gốc. Đọc diễn giải hiện tại
trong [current-results.md](../current-results.md) trước khi dùng số liệu.

Ảnh overlay200701 trong `docs/assets/` copy từ output cùng JSON tương ứng.
Gain8 chỉ phục vụ hiển thị. Các snapshot bất biến không thay cho regression run
trên source/image/container release mới.
