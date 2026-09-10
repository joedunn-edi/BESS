"""
evaluate_rtm_forecaster.py — Tier 2, part 2 for ERCOT: train/evaluate a
LightGBM forecaster on real RTM+DAM data against naive persistence, same
5-fold expanding-window time-series CV GB's forecaster stage used
(ADR-015), now with build_features_with_dam()'s DAM-price features.
"""

import pandas as pd

from bess.features import build_features_with_dam
from bess.forecaster import evaluate_forecaster
from bess.sources_ercot import CHICAGO

rtm_df = pd.read_parquet("data/ercot_rtm_west.parquet")
dam_df = pd.read_parquet("data/ercot_dam_west.parquet")

features = build_features_with_dam(rtm_df, dam_df, tz=CHICAGO)
print(f"{len(features)} feature rows after warm-up + DAM-merge drop")

# 15min, 1h, 3h, 6h, 12h, 24h at RTM's 15-min granularity — same time
# spans GB's forecaster stage used (1, 6, 12, 24, 48 at 30-min periods)
horizons = [1, 4, 12, 24, 48, 96]
results = evaluate_forecaster(features, horizons=horizons, n_splits=5)
print(results.to_string(index=False))
