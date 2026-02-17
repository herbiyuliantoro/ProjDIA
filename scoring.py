from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Returns 0.0 if inputs are empty or invalid."""
    if a.size == 0 or b.size == 0:
        return 0.0
    try:
        return float(1.0 - cosine_distance(a, b))
    except Exception:
        return 0.0


def macc_score(shared: int, cosine_sim: float) -> float:
    """MaCC score  (shared^(1/5) * cosine)."""
    return (shared ** (1 / 5)) * float(cosine_sim) if shared > 0 else 0.0


def ppm_penalty(ppm_abs_mean_centered: float, ppm_sigma: float) -> float:
    """Gaussian-like penalty based on centered mean absolute ppm error."""
    if ppm_sigma <= 0:
        return 1.0
    x = float(ppm_abs_mean_centered) / float(ppm_sigma)
    return float(np.exp(-0.5 * x * x))


def _tie_breaker_factor(x: float, lo: float = 0.85, hi: float = 1.0) -> float:
    """Map x in [0,1] to a gentle multiplicative factor in [lo, hi]."""
    x = float(np.clip(x, 0.0, 1.0))
    return float(lo + (hi - lo) * x)


def _reconstruct_library_mz(query_mz: float, ppm_diff: float) -> float:
    """Given query m/z and ppm difference (ref - query), reconstruct ref m/z."""
    denom = 1.0 - (ppm_diff * 1e-6)
    if denom == 0.0:
        return float("nan")
    return float(query_mz) / denom


def _project_to_library_bins(
    group: pd.DataFrame,
    lib_mz: np.ndarray,
    lib_intensity: np.ndarray,
    ppm_tolerance: float,
    ppm_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project query peaks into the library fragment bins.

    For each library fragment m/z bin, keep the *maximum* query intensity among matches to that bin.
    Returns:
      query_proj_intensity (len == n_lib_peaks)
      ppm_selected (len == n_lib_peaks, nan for unmatched)
    """
    n = int(lib_mz.size)
    q_proj = np.zeros(n, dtype=float)
    ppm_sel = np.full(n, np.nan, dtype=float)

    # Small n (<= ~10-20) is typical after max_library_peaks filtering; O(n*m) is fine.
    q_mz = group["queryMz"].to_numpy(dtype=float, copy=False)
    q_int = group["queryIntensity"].to_numpy(dtype=float, copy=False)
    ppm = group["ppmDifference"].to_numpy(dtype=float, copy=False)

    for qm, qi, pp in zip(q_mz, q_int, ppm):
        ref_mz = _reconstruct_library_mz(qm, pp)
        if not np.isfinite(ref_mz):
            continue

        # Find closest library bin
        j = int(np.argmin(np.abs(lib_mz - ref_mz)))
        # Safety check: ensure within tolerance
        # (Use original pp; it was computed from the library peak during matching.)
        if abs(float(pp) - float(ppm_center)) > float(ppm_tolerance) * 1.05:
            continue

        # Keep best (maximum) intensity for this library bin
        if qi > q_proj[j]:
            q_proj[j] = float(qi)
            ppm_sel[j] = float(pp)

    return q_proj, ppm_sel


def _group_metrics(
    group: pd.DataFrame,
    query_meta: dict,
    library_peaks_by_idx: dict | None,
    ppm_tolerance: float,
    ppm_center: float,
    ppm_sigma: float,
    mode: str,
    use_projected: bool,
    idf_bin_width_da: float,
    idf_bin_to_w: dict | None,
    idf_default_w: float,
) -> pd.Series:
    """Compute per-(libraryIdx, queryIdx) metrics."""
    scan = int(group["queryIdx"].iloc[0])
    lib_idx = int(group["libraryIdx"].iloc[0])

    # ---- Build vectors for cosine / MaCC ----
    if use_projected and library_peaks_by_idx is not None and lib_idx in library_peaks_by_idx:
        lib_peaks = library_peaks_by_idx[lib_idx]
        lib_mz = np.asarray([p[0] for p in lib_peaks], dtype=float)
        lib_int = np.asarray([p[1] for p in lib_peaks], dtype=float)

        q_proj, ppm_sel = _project_to_library_bins(group, lib_mz, lib_int, ppm_tolerance, ppm_center)

        shared = int(np.count_nonzero(q_proj > 0))
        lib_vec = np.sqrt(np.maximum(lib_int, 0.0))
        que_vec = np.sqrt(np.maximum(q_proj, 0.0))

        # Optional IDF weighting: down-weight ubiquitous fragment m/z bins
        if idf_bin_to_w is not None and float(idf_bin_width_da) > 0.0:
            bins = np.rint(lib_mz / float(idf_bin_width_da)).astype(np.int64, copy=False)
            w = np.fromiter((float(idf_bin_to_w.get(int(b), idf_default_w)) for b in bins), dtype=float)
            lib_vec = lib_vec * w
            que_vec = que_vec * w
        cos = cosine_score(lib_vec, que_vec)

        ppm_used = ppm_sel[np.isfinite(ppm_sel)]
        ppm_abs_mean = float(np.mean(np.abs(ppm_used - ppm_center))) if ppm_used.size else 999.0

        matched_int = float(np.sum(q_proj)) if shared else 0.0
        below_mask = None
        prec_mz = float(query_meta[scan].precursor_mz) if scan in query_meta else 0.0
        if prec_mz > 0:
            # We don't have the *projected* m/z per bin; use observed queryMz for "below" accounting.
            below_mask = group["queryMz"].to_numpy(dtype=float) < prec_mz
        matched_below = float(np.sum(group.loc[below_mask, "queryIntensity"].to_numpy(dtype=float))) if below_mask is not None else matched_int

    else:
        # Fallback: treat raw matched pairs as vectors (less accurate)
        shared = int(group.shape[0])
        lib_vec = np.sqrt(group["libraryIntensity"].to_numpy(dtype=float))
        que_vec = np.sqrt(group["queryIntensity"].to_numpy(dtype=float))

        # Optional IDF weighting using reconstructed library m/z per matched pair
        if idf_bin_to_w is not None and float(idf_bin_width_da) > 0.0:
            q_mz = group["queryMz"].to_numpy(dtype=float, copy=False)
            ppm = group["ppmDifference"].to_numpy(dtype=float, copy=False)
            ref_mz = np.array([_reconstruct_library_mz(qm, pp) for qm, pp in zip(q_mz, ppm)], dtype=float)
            bins = np.rint(ref_mz / float(idf_bin_width_da)).astype(np.int64, copy=False)
            w = np.fromiter((float(idf_bin_to_w.get(int(b), idf_default_w)) for b in bins), dtype=float)
            lib_vec = lib_vec * w
            que_vec = que_vec * w
        cos = cosine_score(lib_vec, que_vec)
        ppm_abs_mean = float(np.mean(np.abs(group["ppmDifference"].to_numpy(dtype=float) - ppm_center))) if shared else 999.0

        matched_int = float(np.sum(group["queryIntensity"].to_numpy(dtype=float))) if shared else 0.0
        prec_mz = float(query_meta[scan].precursor_mz) if scan in query_meta else 0.0
        below = group[group["queryMz"] < prec_mz] if prec_mz > 0 else group
        matched_below = float(np.sum(below["queryIntensity"].to_numpy(dtype=float))) if len(below) else 0.0

    macc = macc_score(shared, cos)

    # ---- rank purely by MaCC ----
    if mode == "macc":
        return pd.Series(
            {
                "shared": shared,
                "cosineScore": float(cos),
                "maccScore": float(macc),
                "ppmAbsMean": float(ppm_abs_mean),
                "ppmPenalty": 1.0,
                "ionCount": float(matched_int),
                "explained": 0.0,
                "explainedAll": 0.0,
                "finalScore": float(macc),
            }
        )

    # ---- Enhanced v2: keep MaCC as the core; use ppm/explained as *tie-breakers* ----
    # A narrower sigma increases discriminative power, but we don't want to nuke DI IDs.
    # Use half the tolerance (min 1 ppm) as a reasonable default.
    ppen = ppm_penalty(ppm_abs_mean, max(1.0, 0.5 * float(ppm_sigma)))

    top_tic = float(query_meta[scan].top_tic) if scan in query_meta else 0.0
    explained_all = (matched_int / top_tic) if top_tic > 0 else 0.0
    explained_below = (matched_below / top_tic) if top_tic > 0 else 0.0
    explained_all = float(np.clip(explained_all, 0.0, 1.0))
    explained_below = float(np.clip(explained_below, 0.0, 1.0))

    # Prefer "below precursor" matching when precursor exists, otherwise use all matched signal.
    explained = explained_below if prec_mz > 0 else explained_all

    # Convert penalties into gentle tie-breaker factors.
    # - ppmPenalty is in (0,1]; map to ~[0.85, 1.0]
    # - explained is often tiny in direct infusion; avoid halving scores.
    ppm_factor = _tie_breaker_factor(ppen, lo=0.85, hi=1.0)
    expl_factor = _tie_breaker_factor(np.sqrt(explained), lo=0.85, hi=1.0)

    final = macc * ppm_factor * expl_factor

    return pd.Series(
        {
            "shared": shared,
            "cosineScore": float(cos),
            "maccScore": float(macc),
            "ppmAbsMean": float(ppm_abs_mean),
            "ppmPenalty": float(ppen),
            "ionCount": float(matched_below),
            "explained": float(explained),
            "explainedAll": float(explained_all),
            "finalScore": float(final),
        }
    )


def score_matches(
    match_df: pd.DataFrame,
    query_meta: dict,
    ppm_tolerance: float,
    score_mode: str = "enhanced",
    library_peaks_by_idx: dict | None = None,
    use_projected: bool = True,
    ppm_center: float = 0.0,
    idf_bin_width_da: float = 0.0,
    idf_bin_to_w: dict | None = None,
    idf_default_w: float = 1.0,
) -> pd.DataFrame:
    """
    Faster replacement for the original score_matches() that avoids
    Pandas groupby().apply(...), while keeping the SAME scoring logic
    and output columns.

    It uses a stable sort on packed (libraryIdx, queryIdx) keys so that
    row order *within each group* is preserved (important for tie handling).
    """
    if len(match_df) == 0:
        return pd.DataFrame(
            columns=[
                "libraryIdx",
                "queryIdx",
                "shared",
                "cosineScore",
                "maccScore",
                "ppmAbsMean",
                "ppmPenalty",
                "ionCount",
                "explained",
                "explainedAll",
                "finalScore",
            ]
        )

    mode = (score_mode or "enhanced").strip().lower()
    if mode not in ("enhanced", "macc"):
        mode = "enhanced"

    # Enhanced v2: keep MaCC as the core; use ppm/explained primarily as tie-breakers.
    # We still pass ppm_tolerance through as a scale parameter, but we narrow it inside
    # the penalty to increase discrimination without killing sensitivity.
    ppm_sigma = float(ppm_tolerance) if mode == "enhanced" else 0.0

    # ---- Pull columns once (fast views) ----
    lib_idx_all = match_df["libraryIdx"].to_numpy(dtype=np.int64, copy=False)
    scan_all = match_df["queryIdx"].to_numpy(dtype=np.int64, copy=False)

    lib_int_all = match_df["libraryIntensity"].to_numpy(dtype=float, copy=False)
    q_int_all = match_df["queryIntensity"].to_numpy(dtype=float, copy=False)
    q_mz_all = match_df["queryMz"].to_numpy(dtype=float, copy=False)
    ppm_all = match_df["ppmDifference"].to_numpy(dtype=float, copy=False)

    # Pack key: stable grouping identifier (libraryIdx, queryIdx)
    # (Same packing idea you used in eliminate_low_count_matches) :contentReference[oaicite:3]{index=3}
    keys = (lib_idx_all << 32) ^ scan_all

    # Stable sort by key so rows within the same key keep original order
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]

    lib_idx_s = lib_idx_all[order]
    scan_s = scan_all[order]
    lib_int_s = lib_int_all[order]
    q_int_s = q_int_all[order]
    q_mz_s = q_mz_all[order]
    ppm_s = ppm_all[order]

    # Find group boundaries in the sorted keys
    # starts: indices where a new key begins
    change = np.empty(keys_s.size, dtype=bool)
    change[0] = True
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = keys_s.size

    # Output accumulators
    out_lib = np.empty(starts.size, dtype=np.int64)
    out_scan = np.empty(starts.size, dtype=np.int64)
    shared_arr = np.empty(starts.size, dtype=np.int64)
    cos_arr = np.empty(starts.size, dtype=float)
    macc_arr = np.empty(starts.size, dtype=float)
    ppmabs_arr = np.empty(starts.size, dtype=float)
    ppmpen_arr = np.empty(starts.size, dtype=float)
    ion_arr = np.empty(starts.size, dtype=float)
    expl_arr = np.empty(starts.size, dtype=float)
    explall_arr = np.empty(starts.size, dtype=float)
    final_arr = np.empty(starts.size, dtype=float)

    # Small helper: project query peaks into library bins using the SAME logic as _project_to_library_bins()
    def _project_to_library_bins_arrays(q_mz: np.ndarray, q_int: np.ndarray, ppm: np.ndarray, lib_mz: np.ndarray, ppm_tol: float, ppm_center_val: float):
        n = int(lib_mz.size)
        q_proj = np.zeros(n, dtype=float)
        ppm_sel = np.full(n, np.nan, dtype=float)

        for qm, qi, pp in zip(q_mz, q_int, ppm):
            # same reconstruction as _reconstruct_library_mz()
            denom = 1.0 - (float(pp) * 1e-6)
            if denom == 0.0:
                continue
            ref_mz = float(qm) / denom
            if not np.isfinite(ref_mz):
                continue

            j = int(np.argmin(np.abs(lib_mz - ref_mz)))

            # Same safety check as original :contentReference[oaicite:4]{index=4}
            if abs(float(pp) - float(ppm_center_val)) > float(ppm_tol) * 1.05:
                continue

            # Same strict ">" tie behavior as original :contentReference[oaicite:5]{index=5}
            if qi > q_proj[j]:
                q_proj[j] = float(qi)
                ppm_sel[j] = float(pp)

        return q_proj, ppm_sel

    # ---- Main loop over groups ----
    for gi, (s, e) in enumerate(zip(starts, ends)):
        lib_idx = int(lib_idx_s[s])
        scan = int(scan_s[s])

        out_lib[gi] = lib_idx
        out_scan[gi] = scan

        g_lib_int = lib_int_s[s:e]
        g_q_int = q_int_s[s:e]
        g_q_mz = q_mz_s[s:e]
        g_ppm = ppm_s[s:e]

        # ---- Build vectors for cosine / MaCC (same branching as _group_metrics) :contentReference[oaicite:6]{index=6}
        do_project = bool(use_projected) and (library_peaks_by_idx is not None) and (lib_idx in library_peaks_by_idx)

        if do_project:
            lib_peaks = library_peaks_by_idx[lib_idx]
            lib_mz = np.asarray([p[0] for p in lib_peaks], dtype=float)
            lib_int = np.asarray([p[1] for p in lib_peaks], dtype=float)

            q_proj, ppm_sel = _project_to_library_bins_arrays(
                g_q_mz, g_q_int, g_ppm, lib_mz, float(ppm_tolerance), float(ppm_center)
            )

            shared = int(np.count_nonzero(q_proj > 0))
            lib_vec = np.sqrt(np.maximum(lib_int, 0.0))
            que_vec = np.sqrt(np.maximum(q_proj, 0.0))

            # IDFscore
            if idf_bin_to_w is not None and float(idf_bin_width_da) > 0.0:
                bins = np.rint(lib_mz / float(idf_bin_width_da)).astype(np.int64, copy=False)
                w = np.fromiter(
                    (float(idf_bin_to_w.get(int(b), idf_default_w)) for b in bins),
                    dtype=float,
                )
                lib_vec = lib_vec * w
                que_vec = que_vec * w

            cos = cosine_score(lib_vec, que_vec)


            ppm_used = ppm_sel[np.isfinite(ppm_sel)]
            ppm_abs_mean = float(np.mean(np.abs(ppm_used - float(ppm_center)))) if ppm_used.size else 999.0

            matched_int = float(np.sum(q_proj)) if shared else 0.0

            prec_mz = float(query_meta[scan].precursor_mz) if scan in query_meta else 0.0
            if prec_mz > 0:
                below_mask = g_q_mz < prec_mz
                matched_below = float(np.sum(g_q_int[below_mask])) if below_mask.size else 0.0
            else:
                matched_below = matched_int

        else:
            shared = int(e - s)
            lib_vec = np.sqrt(g_lib_int)
            que_vec = np.sqrt(g_q_int)

            cos = cosine_score(lib_vec, que_vec)
            ppm_abs_mean = float(np.mean(np.abs(g_ppm - float(ppm_center)))) if shared else 999.0

            matched_int = float(np.sum(g_q_int)) if shared else 0.0
            prec_mz = float(query_meta[scan].precursor_mz) if scan in query_meta else 0.0
            if prec_mz > 0:
                below_mask = g_q_mz < prec_mz
                matched_below = float(np.sum(g_q_int[below_mask])) if below_mask.size else 0.0
            else:
                matched_below = matched_int

        macc = macc_score(shared, cos)

        # ---- Mode-specific final score logic (same as _group_metrics) :contentReference[oaicite:7]{index=7}
        if mode == "macc":
            shared_arr[gi] = shared
            cos_arr[gi] = float(cos)
            macc_arr[gi] = float(macc)
            ppmabs_arr[gi] = float(ppm_abs_mean)
            ppmpen_arr[gi] = 1.0
            ion_arr[gi] = float(matched_int)
            expl_arr[gi] = 0.0
            explall_arr[gi] = 0.0
            final_arr[gi] = float(macc)
            continue

        ppen = ppm_penalty(ppm_abs_mean, max(1.0, 0.5 * float(ppm_sigma)))

        top_tic = float(query_meta[scan].top_tic) if scan in query_meta else 0.0
        explained_all = (matched_int / top_tic) if top_tic > 0 else 0.0
        explained_below = (matched_below / top_tic) if top_tic > 0 else 0.0
        explained_all = float(np.clip(explained_all, 0.0, 1.0))
        explained_below = float(np.clip(explained_below, 0.0, 1.0))

        explained = explained_below if prec_mz > 0 else explained_all

        ppm_factor = _tie_breaker_factor(ppen, lo=0.85, hi=1.0)
        expl_factor = _tie_breaker_factor(np.sqrt(explained), lo=0.85, hi=1.0)

        final = macc * ppm_factor * expl_factor

        shared_arr[gi] = shared
        cos_arr[gi] = float(cos)
        macc_arr[gi] = float(macc)
        ppmabs_arr[gi] = float(ppm_abs_mean)
        ppmpen_arr[gi] = float(ppen)
        ion_arr[gi] = float(matched_below)
        expl_arr[gi] = float(explained)
        explall_arr[gi] = float(explained_all)
        final_arr[gi] = float(final)

    score_df = pd.DataFrame(
        {
            "libraryIdx": out_lib,
            "queryIdx": out_scan,
            "shared": shared_arr,
            "cosineScore": cos_arr,
            "maccScore": macc_arr,
            "ppmAbsMean": ppmabs_arr,
            "ppmPenalty": ppmpen_arr,
            "ionCount": ion_arr,
            "explained": expl_arr,
            "explainedAll": explall_arr,
            "finalScore": final_arr,
        }
    )

    return score_df.sort_values("finalScore", ascending=False).reset_index(drop=True)
