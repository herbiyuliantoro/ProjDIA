from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


@dataclass
class RescoreSettings:
    enabled: bool = True
    iterations: int = 4

    # Positives: use targets with q <= positive_q when available (semi-supervised)
    positive_q: float = 0.01
    # If q-positive set is too small, seed positives from the top-scoring targets
    seed_top_targets: int = 3000
    # Minimum number of positives required; otherwise fall back to seeding
    min_positives: int = 200
    # De-duplicate positives by peptide (reduces correlated-label overfit)
    dedup_positives_by_peptide: bool = True
    peptide_col: str = "peptide"

    # Regularization strength (smaller -> stronger regularization)
    C: float = 0.5

    # Auto-disable guards (avoid overfit on tiny datasets)
    min_psms: int = 2000
    min_decoys: int = 200
    min_targets: int = 200

    # Cross-validated rescoring by scan (reduces in-sample overfit)
    enable_cv: bool = True
    cv_folds: int = 3
    random_state: int = 0


FEATURE_COLS = [
    "shared",
    "cosineScore",
    "maccScore",
    "ppmAbsMean",
    "ppmPenalty",
    "explained",
    "explainedAll",
    "ionCount",
]


def _make_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    # Stabilize scale / tails
    if "shared" in X:
        X["shared"] = np.log1p(X["shared"].astype(float))
    if "ionCount" in X:
        X["ionCount"] = np.log1p(np.maximum(X["ionCount"].astype(float), 0.0))
    if "ppmAbsMean" in X:
        X["ppmAbsMean"] = np.log1p(np.maximum(X["ppmAbsMean"].astype(float), 0.0))
    # Fill NaNs safely
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=float)


def _add_q_values_from_score(df: pd.DataFrame, score_col: str, is_decoy_col: str, decoy_scale: float = 1.0) -> pd.DataFrame:
    """Add temporary FDR/q-values computed from score, preserving the original row order.

    NOTE: A previous implementation returned a score-sorted DataFrame (reset_index),
    which misaligned q-values when the caller expected the original order.
    """
    out = df.copy()
    s = pd.to_numeric(out[score_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float, copy=False)
    d = out[is_decoy_col].astype(int).to_numpy(copy=False)

    order = np.argsort(-s)  # descending by score
    d_sorted = d[order]

    dec = np.cumsum(d_sorted) * float(decoy_scale)
    tot = np.arange(1, len(out) + 1, dtype=float)
    tar = tot - dec
    fdr_sorted = dec / np.maximum(tar, 1.0)
    q_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]

    # unsort back to original positions
    fdr = np.empty_like(fdr_sorted)
    q = np.empty_like(q_sorted)
    fdr[order] = fdr_sorted
    q[order] = q_sorted

    out["fdr_tmp"] = fdr
    out["q_tmp"] = q
    return out


def _select_positives(
    score: np.ndarray,
    is_decoy: np.ndarray,
    q: np.ndarray,
    settings: RescoreSettings,
    peptides: np.ndarray | None = None,
) -> np.ndarray:
    """Select positive indices from targets, optionally de-duplicated by peptide."""
    pos = np.where((is_decoy == 0) & (q <= float(settings.positive_q)))[0]

    # If too few, seed from top-scoring targets
    if pos.size < int(settings.min_positives):
        rank = np.argsort(-score)
        tar_rank = [i for i in rank if is_decoy[i] == 0]
        pos = np.asarray(tar_rank[: int(settings.seed_top_targets)], dtype=int)

    # Optional de-duplication by peptide
    if settings.dedup_positives_by_peptide and peptides is not None and pos.size > 0:
        # Sort positives by score descending, then take first occurrence per peptide
        pos_sorted = pos[np.argsort(-score[pos])]
        seen = set()
        out = []
        for i in pos_sorted:
            p = peptides[i]
            if p in seen:
                continue
            seen.add(p)
            out.append(i)
        pos = np.asarray(out, dtype=int)

    return pos


def _fit_and_score_iterative(
    X_train: np.ndarray,
    X_test: np.ndarray,
    base_score_train: np.ndarray,
    is_decoy_train: np.ndarray,
    settings: RescoreSettings,
    decoy_scale: float,
    peptides_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Iterative semi-supervised training on train set; returns (score_train, score_test, info)."""
    score_train = base_score_train.copy()
    info: dict = {}

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=8000,
                    class_weight="balanced",
                    C=float(settings.C),
                    random_state=int(settings.random_state),
                ),
            ),
        ]
    )

    for it in range(int(settings.iterations)):
        tmp = _add_q_values_from_score(
            pd.DataFrame({"s": score_train, "d": is_decoy_train}),
            score_col="s",
            is_decoy_col="d",
            decoy_scale=decoy_scale,
        )
        q = tmp["q_tmp"].to_numpy()

        pos = _select_positives(score_train, is_decoy_train, q, settings, peptides=peptides_train)
        neg = np.where(is_decoy_train == 1)[0]

        y = np.full(X_train.shape[0], -1, dtype=int)
        y[pos] = 1
        y[neg] = 0
        labeled = y != -1

        # Safety: if we can't form both classes, stop early
        if np.unique(y[labeled]).size < 2:
            info[f"iter_{it+1}_stopped"] = "single_class"
            break

        pipe.fit(X_train[labeled], y[labeled])

        # logit score (ranking-friendly)
        proba_train = pipe.predict_proba(X_train)[:, 1]
        proba_test = pipe.predict_proba(X_test)[:, 1]

        proba_train = np.clip(proba_train, 1e-12, 1 - 1e-12)
        proba_test = np.clip(proba_test, 1e-12, 1 - 1e-12)

        score_train = np.log(proba_train / (1 - proba_train))
        score_test = np.log(proba_test / (1 - proba_test))

        info[f"iter_{it+1}_positives"] = int(pos.size)
        info[f"iter_{it+1}_negatives"] = int(neg.size)

    # If loop never ran, define score_test from base (no change)
    if "iter_1_positives" not in info:
        score_test = base_score_train[:0]  # will be overwritten by caller if needed

    return score_train, score_test, info


def percolator_lite_rescore(
    df: pd.DataFrame,
    base_score_col: str = "finalScore",
    is_decoy_col: str = "isDecoy",
    scan_col: str = "scan",
    settings: Optional[RescoreSettings] = None,
    decoy_scale: float = 1.0,
) -> Tuple[pd.Series, dict]:
    """Semi-supervised logistic regression rescoring (Percolator-lite).

    Improvements over the initial version:
      - Consistency guards: auto-disable on tiny datasets
      - Optional peptide de-duplication for positive set
      - Optional scan-grouped cross-validation to reduce in-sample overfit
    """
    settings = settings or RescoreSettings()

    if not settings.enabled:
        return df[base_score_col].astype(float), {"enabled": False}

    if not _HAVE_SKLEARN:
        return df[base_score_col].astype(float), {"enabled": False, "reason": "scikit-learn not installed"}

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(feature_cols) < 4:
        return df[base_score_col].astype(float), {"enabled": False, "reason": "insufficient features"}

    n = int(len(df))
    is_decoy = df[is_decoy_col].astype(int).to_numpy()
    n_dec = int(np.sum(is_decoy == 1))
    n_tar = int(np.sum(is_decoy == 0))

    # Guards to avoid overfitting / unstable models
    if n < int(settings.min_psms) or n_dec < int(settings.min_decoys) or n_tar < int(settings.min_targets):
        return df[base_score_col].astype(float), {
            "enabled": False,
            "reason": "insufficient_data",
            "n_psms": n,
            "n_decoys": n_dec,
            "n_targets": n_tar,
        }

    X = _make_feature_matrix(df, feature_cols)
    base = df[base_score_col].astype(float).to_numpy()

    peptide_col = getattr(settings, "peptide_col", "peptide")
    peptides_all = None
    if getattr(settings, "dedup_positives_by_peptide", True) and (peptide_col in df.columns):
        peptides_all = df[peptide_col].astype(str).to_numpy()

    info: dict = {
        "enabled": True,
        "iterations": int(settings.iterations),
        "feature_cols": feature_cols,
        "enable_cv": bool(settings.enable_cv),
        "cv_folds": int(settings.cv_folds),
    }

    # --- Cross-validated scoring by scan (recommended) ---
    if bool(settings.enable_cv) and int(settings.cv_folds) >= 2 and (scan_col in df.columns):
        scans = pd.to_numeric(df[scan_col], errors="coerce").fillna(-1).astype(int).to_numpy()
        # If scans are missing/invalid, fall back to non-CV
        if np.all(scans < 0):
            info["enable_cv"] = False
        else:
            folds = int(settings.cv_folds)
            fold_id = np.mod(np.abs(scans), folds)
            out_score = np.empty(n, dtype=float)

            for f in range(folds):
                test_mask = fold_id == f
                train_mask = ~test_mask

                # If a fold is empty or train set too small, fall back to base score for that fold
                if int(np.sum(test_mask)) == 0 or int(np.sum(train_mask)) < int(settings.min_psms) // 2:
                    out_score[test_mask] = base[test_mask]
                    info[f"fold_{f}_skipped"] = True
                    continue

                X_train = X[train_mask]
                X_test = X[test_mask]
                base_train = base[train_mask]
                is_decoy_train = is_decoy[train_mask]

                peptides_train = peptides_all[train_mask] if peptides_all is not None else None

                _, score_test, fold_info = _fit_and_score_iterative(
                    X_train=X_train,
                    X_test=X_test,
                    base_score_train=base_train,
                    is_decoy_train=is_decoy_train,
                    settings=settings,
                    decoy_scale=decoy_scale,
                    peptides_train=peptides_train,
                )

                # If iterative training failed early, default to base for this fold
                if score_test.shape[0] != int(np.sum(test_mask)):
                    out_score[test_mask] = base[test_mask]
                    info[f"fold_{f}_failed"] = True
                else:
                    out_score[test_mask] = score_test

                # keep a compact per-fold summary
                info[f"fold_{f}_n_test"] = int(np.sum(test_mask))
                for k, v in fold_info.items():
                    if k.endswith("_positives") or k.endswith("_negatives") or k.endswith("_stopped"):
                        info[f"fold_{f}_{k}"] = v

            return pd.Series(out_score, index=df.index, name="mlScore"), info

    # --- Non-CV (in-sample) rescoring fallback ---
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=8000,
                    class_weight="balanced",
                    C=float(settings.C),
                    random_state=int(settings.random_state),
                ),
            ),
        ]
    )

    score = base.copy()

    for it in range(int(settings.iterations)):
        tmp = _add_q_values_from_score(
            pd.DataFrame({"s": score, "d": is_decoy}),
            score_col="s",
            is_decoy_col="d",
            decoy_scale=decoy_scale,
        )
        q = tmp["q_tmp"].to_numpy()

        pos = _select_positives(score, is_decoy, q, settings, peptides=peptides_all)
        neg = np.where(is_decoy == 1)[0]

        y = np.full(n, -1, dtype=int)
        y[pos] = 1
        y[neg] = 0
        labeled = y != -1

        if np.unique(y[labeled]).size < 2:
            info[f"iter_{it+1}_stopped"] = "single_class"
            break

        pipe.fit(X[labeled], y[labeled])

        proba = pipe.predict_proba(X)[:, 1]
        proba = np.clip(proba, 1e-12, 1 - 1e-12)
        score = np.log(proba / (1 - proba))

        info[f"iter_{it+1}_positives"] = int(pos.size)
        info[f"iter_{it+1}_negatives"] = int(neg.size)

    return pd.Series(score, index=df.index, name="mlScore"), info
