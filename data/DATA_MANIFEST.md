# Local data manifest

Các file dưới đây là bản sao local dùng cho phase Plaintext. Notebook phải kiểm
SHA-256 trước khi chạy và không tự động thay bằng file khác.

| Dataset | Local file | Rows including header | SHA-256 |
| --- | --- | ---: | --- |
| Leipzig ECG | `data/ecg/ecg_beats_features_selected_20.csv` | 113,847 | `ffa8482f426ca0fdb6055f745610e88befe7e0cb36889a0bd2678dfe7bca5a9b` |
| Heart+ | `data/heartplus/merged_data.csv` | 764,928 | `8c3611f95469b3a06fe33628c2d82283dc3571b57bd0ff9b8c5c63ec44dac27e` |
| MIMIC-IV | `data/mimic/mimic_icu_mortality_cfe_12671.csv` | 12,672 | `d79aa6ecdc9dd04ad540923b30323c4d693a5e84fcad28444c366d2fd703eadb` |

ECG dùng feature table 20 biến từ source PhysioNet; Heart+ dùng merged BRFSS
table; MIMIC dùng extract 12,671 dòng. Raw row counts phải được báo riêng
với usable rows sau cleaning/label contract. Không coi bản copy này là toàn bộ
PhysioNet/MIMIC source dataset.
