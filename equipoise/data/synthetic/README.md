# Synthetic Protocol T data (fake)

These files are **entirely simulated** by `scripts/make_synthetic_data.py`. They contain no
DRCR data. They have exactly the schema of the real tables so tests and CI can run without
the DRCR download.

| File | Contents |
|---|---|
| `protocolT_analysis.csv` | one row per fake patient, same 47 columns and dtypes as `data/protocolT_analysis.csv` |
| `protocolT_visits.csv` | one row per fake patient × visit, same columns as `data/protocolT_visits.csv` |
| `protocolT_truth.csv` | noise-free week-52 letters under each arm (`mu_*`) and planted effects vs bevacizumab (`tau_*`) |

Planted effect (see `config/default.yaml`, `synthetic:`): aflibercept adds more letters when
`X_va_below_69 = 1`; ranibizumab adds a constant. **Never** report numbers from these files
as trial results.

Regenerate with:

```bash
python scripts/make_synthetic_data.py
```
