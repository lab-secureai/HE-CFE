# Dữ liệu bổ sung cho revision

## Adult Income (UCI Census Income) — dùng cho E4 (plaintext, không HE)

Nguồn local: sao chép nguyên trạng (chỉ đọc) từ
`../../../A-star/data/adult/` ngày 2026-10-27. Nguồn gốc UCI:
https://archive.ics.uci.edu/static/public/2/adult.zip

| File | Dòng | SHA-256 |
|---|---:|---|
| `adult/adult.data` | 32,561 | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` |
| `adult/adult.test` | 16,281 (+1 dòng header `|1x3 Cross validator`) | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` |
| `adult/adult.names` | — | `c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a` |
| `adult/adult.zip` | — | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` (khớp `A-star/data/DATA_SOURCES.md`) |

Kiểm tra đã chạy: gộp data + test = 48,842 dòng; nhãn `<=50K` 37,155,
`>50K` 11,687 (23.9% positive) sau khi bỏ dấu chấm cuối nhãn ở file test;
3,620 dòng có giá trị thiếu (`?`, ở workclass/occupation/native_country).

Lưu ý:
- Đọc bằng `skipinitialspace=True, na_values='?'`; file test bỏ dòng đầu và
  nhãn có dấu `.` cuối.
- KHÔNG dùng `A-star/data/adult/adult_processed.npz`: đó là bản tiền xử lý
  108 chiều one-hot cho dự án MIA A-star, khác contract preprocessor của bài
  này (QuantileTransformer normal + one-hot drop-first, fit chỉ trên train).
- Trên Kaggle đã có dataset công khai `lamhuy8904/astar-mia-adult`, nhưng nó
  chứa bản `.npz` đã xử lý; cho E4 nên upload dataset mới chứa file thô (hoặc
  kiểm tra dataset đó có kèm file thô hay không trước khi dùng).
- Cột `fnlwgt` là trọng số lấy mẫu điều tra, đề xuất loại khỏi predictor;
  `education` trùng thông tin với `education_num`, đề xuất chỉ giữ
  `education_num` (ordinal, chỉ được tăng).
