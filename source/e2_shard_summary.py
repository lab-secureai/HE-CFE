"""One line per downloaded E2 shard: acceptance, hours, yields, CKKS effects."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results" / "e2_kaggle"
for path in sorted(ROOT.glob("*/*/e2_*/e2_shard_acceptance.json")):
    a = json.loads(path.read_text())
    d = pd.read_csv(path.parent / "e2_per_factual.csv")
    e, p = d[d["mode"] == "encrypted_fixed_r"], d[d["mode"] == "plaintext_fixed_r"]
    print(f"{path.parent.name}: accepted {a['accepted']}, {a['wall_seconds'] / 3600:.2f} h, "
          f"n {len(e)}, yield enc {e.valid_cfe_yield_at_k.mean():.3f} vs pt "
          f"{p.valid_cfe_yield_at_k.mean():.3f}, fail {a['checks']['returned_failing_plaintext_margin_total']}, "
          f"twin {e.identical_to_plaintext_twin.mean():.2f}, maxerr {e.max_abs_logit_error_vs_torch.max():.4f}")
