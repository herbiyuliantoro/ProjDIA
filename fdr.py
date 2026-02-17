from __future__ import annotations

import numpy as np
import pandas as pd


def add_q_values(
    df: pd.DataFrame,
    is_decoy_col: str = "isDecoy",
    score_col: str = "finalScore",
    decoy_scale: float = 1.0,
) -> pd.DataFrame:
    """Compute running FDR and q-values with optional decoy scaling.

    If the library contains a different number of decoys vs targets, set:
      decoy_scale = n_targets / n_decoys
    For 1:1 target:decoy, decoy_scale = 1.
    """
    if len(df) == 0:
        out = df.copy()
        out["fdr"] = []
        out["q"] = []
        return out

    d = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    is_decoy = d[is_decoy_col].astype(int).to_numpy()

    dec = np.cumsum(is_decoy).astype(float) * float(decoy_scale)
    tot = np.arange(1, len(d) + 1, dtype=float)
    tar = tot - dec

    fdr = dec / np.maximum(tar, 1.0)
    q = np.minimum.accumulate(fdr[::-1])[::-1]

    d["fdr"] = fdr
    d["q"] = q
    return d


def filter_q(df: pd.DataFrame, q_cut: float = 0.01) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df[df["q"] <= q_cut].reset_index(drop=True)
