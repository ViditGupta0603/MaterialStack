"""Step 2 — train the band-gap model.

Reads data/band_gaps.csv (measured gaps) and data/dft_gaps.csv (DFT hint), trains one LightGBM model on
all semiconductors and insulators, and saves it to models/gap_model.joblib. The exact table it trained on
(formula, measured gap and every feature) is written to data/features.csv for inspection only: nothing
reads it back, the features are always recomputed from the formulas (~5 s). How good it is, is measured
separately by validate.py (cross-validation), so this script only trains.

Run:  python train.py
"""
import time

from materialstack import model
from materialstack.config import FEATURES

t0 = time.time()
X, y, table = model.training_set()
print(f"training on {len(y)} materials, {X.shape[1]} features "
      f"({X.dft_gap_hybrid.notna().mean():.0%} have a hybrid-DFT hint, {X.dft_gap_gga.notna().mean():.0%} a GGA hint)")
X.assign(formula=table.formula, gap_ev=y)[["formula", "gap_ev", *X.columns]].round(4).to_csv(FEATURES, index=False)
m = model.fit(X, y)
model.save(m, n_train=len(y))
top = sorted(zip(m.feature_importances_, m.feature_name_), reverse=True)[:8]
print("most used features: " + ", ".join(name for _, name in top))
print(f"saved {model.MODEL} and {FEATURES.name} in {time.time() - t0:.0f} s")
