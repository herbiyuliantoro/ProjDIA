from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import os
import numpy as np
import pandas as pd
from bisect import bisect
import scipy.linalg as linalg

from .loaders import load_library, QueryMzxml
from .matching import match_library_to_query_pooled_spectra, eliminate_low_count_matches
from .scoring import score_matches
from .fdr import add_q_values, filter_q
from .rescore import RescoreSettings, percolator_lite_rescore
from .utils import Logger
from .idpicker import build_protein_table_from_peptides


def _parse_semicolon_list(x: object) -> list[str]:
    if x is None:
        return []
    s = str(x)
    if not s or s.lower() == "nan":
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def _maxlfq(sample_by_peptide_log: np.ndarray, min_num_differences: int = 2, tolerance: float = -10.0) -> np.ndarray:
    """MaxLFQ on a (samples x peptides) log-intensity matrix with 0 meaning missing."""
    X = np.array(sample_by_peptide_log, dtype=float, copy=True)
    X[X < tolerance] = 0.0
    sample_num = int(X.shape[0])
    A = np.zeros((sample_num, sample_num), dtype=float)
    B = np.zeros(sample_num, dtype=float)

    # Build A,B from pairwise median differences over shared peptides
    for i in range(sample_num):
        for j in range(i + 1, sample_num):
            s1 = X[i]
            s2 = X[j]
            matches = ~((s1 == 0.0) | (s2 == 0.0))
            n = int(np.count_nonzero(matches))
            if n < int(min_num_differences):
                continue
            diff = float(np.median(s1[matches] - s2[matches]))
            A[i, i] += 1.0
            A[j, j] += 1.0
            A[i, j] = A[j, i] = -1.0
            B[i] += diff
            B[j] -= diff

    unmatched = np.diagonal(A) == 0.0

    # Regularize (same spirit as MaxLFQ reference implementations)
    reg = np.array(np.diagonal(A), dtype=float)
    reg[reg < 2.0] = 1.0
    reg = reg * 0.0001
    A[np.diag_indices_from(A)] = np.diagonal(A) + reg
    B = B + np.amax(X, axis=1) * reg

    # Solve
    try:
        cho = linalg.cho_factor(A)
        out = linalg.cho_solve(cho, B)
    except Exception:
        # Fallback: least squares
        out, *_ = np.linalg.lstsq(A, B, rcond=None)
    out[unmatched] = 0.0
    return out


def _find_lib_keys_in_window(mz_window: Tuple, lib_keys_sorted: list) -> list:
    center_mz = float(mz_window[0])
    width = float(mz_window[1])
    top_mz = center_mz + width / 2
    bot_mz = center_mz - width / 2
    # keys are (precursorMz, peptideName); bisect on first element.
    top_i = bisect(lib_keys_sorted, (top_mz, "z"))
    bot_i = bisect(lib_keys_sorted, (bot_mz, ""))
    return lib_keys_sorted[bot_i:top_i]


def _pool_library_peaks(mz_window: Tuple, lib_dict: dict, lib_keys_sorted: list):
    keys = _find_lib_keys_in_window(mz_window, lib_keys_sorted)
    peaks = []
    for k in keys:
        peaks.extend(lib_dict[k]["peaks"])
    return sorted(peaks)


def _pool_query_peaks(scan_ids: List[int], reader, query_meta: dict, top_k_peaks: int, chunk_size: int):
    for i in range(0, len(scan_ids), chunk_size):
        chunk = scan_ids[i : i + chunk_size]
        pooled = []
        for scan in chunk:
            # pyteomics mzxml index uses spectrum 'id' strings; our window mapping uses scan numbers.
            sid = ""
            if scan in query_meta:
                sid = str(getattr(query_meta[scan], "spectrum_id", "") or "")
            candidates = [c for c in [sid, str(scan), f"scan={scan}"] if c]
            spec = None
            for cid in candidates:
                try:
                    spec = reader.get_by_id(cid)
                    break
                except KeyError:
                    continue
            if spec is None:
                continue

            mz = np.asarray(spec.get("m/z array", []), dtype=float)
            inten = np.asarray(spec.get("intensity array", []), dtype=float)
            if mz.size == 0 or inten.size == 0:
                continue

            if top_k_peaks and top_k_peaks < inten.size:
                idx = np.argpartition(inten, -top_k_peaks)[-top_k_peaks:]
                mz = mz[idx]
                inten = inten[idx]

            keep = inten > 0
            mz = mz[keep]
            inten = inten[keep]
            if mz.size == 0:
                continue

            tag = np.full_like(inten, int(scan), dtype=int)
            pooled.extend(list(zip(mz.tolist(), inten.tolist(), tag.tolist())))
        yield sorted(pooled)


def _robust_center_scale(ppm_values: np.ndarray) -> tuple[float, float]:
    """Robust estimate of (center, sigma) using median + MAD."""
    ppm_values = ppm_values[np.isfinite(ppm_values)]
    if ppm_values.size == 0:
        return 0.0, 0.0
    med = float(np.median(ppm_values))
    mad = float(np.median(np.abs(ppm_values - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(ppm_values)) if ppm_values.size > 1 else 0.0
    return med, sigma




class StopRequested(Exception):
    """Raised when a user requests cancellation via the GUI."""
    pass


@dataclass
class EngineSettings:
    # Matching
    ppm_tolerance: float = 20.0
    min_shared_peaks: int = 3

    # Peak filtering
    top_k_query_peaks: int = 200     # 0 disables (keep all)
    max_library_peaks: int = 150
    scan_chunk_size: int = 50

    # FDR
    q_cutoff: float = 0.01

    # Protein inference (IDPicker on peptide table)
    enable_protein_inference: bool = True
    protein_inference_q: float = 0.01  # peptides with q<=this are used for IDPicker

    # Output-only capping (IMPORTANT: applied AFTER FDR/ML; does NOT affect IDs)
    keep_best_per_scan: bool = False
    max_hits_per_scan: int = 0       # 0 disables

    # Scoring mode
    score_mode: str = "macc"        
    use_projected_scoring: bool = True  # project query into library bins for cosine/shared

    # Interference-aware scoring (run-adaptive IDF-like fragment weighting)
    enable_idf_weighting: bool = False
    idf_bin_width_da: float = 0.02   # fragment m/z bin width (Da)
    idf_smoothing: float = 1.0       # pseudo-count for stability
    idf_w_min: float = 0.05          # clamp weights
    idf_w_max: float = 2.0
    idf_power: float = 1.0           # <1 softer, >1 stronger

    # Mass-error correction pass 
    enable_mass_error_correction: bool = True
    correction_fit_q: float = 0.01
    correction_min_psms: int = 200
    correction_n_sigma: float = 3.0
    correction_min_tol: float = 4.0  # ppm

    # ML rescoring (Percolator-lite)
    use_ml_rescoring: bool = True
    ml_iterations: int = 4

    # ML rescoring controls (to reduce overfit)
    ml_enable_cv: bool = True
    ml_cv_folds: int = 3
    ml_min_psms: int = 2000
    ml_min_decoys: int = 200
    ml_min_targets: int = 200
    ml_min_positives: int = 200
    ml_dedup_positives_by_peptide: bool = True

    # Output controls
    # Default: write ONLY these 3 files per run:
    #   - <stem>.psm.q<q_cutoff>.tsv
    #   - <stem>.peptide.q<q_cutoff>.tsv
    #   - <stem>.protein.q<q_cutoff>.tsv
    write_minimal_outputs: bool = True
    write_full_psm: bool = False
    write_full_peptide: bool = False
    write_full_protein: bool = False
    write_capped_outputs: bool = False
    write_ppm_correction: bool = False

    # LFQ across multiple runs (requires >=2 mzXML files)
    enable_lfq: bool = False
    lfq_method: str = "sum"      # 'sum' or 'maxlfq'
    lfq_min_num_differences: int = 2


class ProjDIAEngine:
    def __init__(
        self,
        library_path: str,
        settings: EngineSettings = EngineSettings(),
        logger: Optional[Logger] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.settings = settings
        self.log = logger or Logger()
        self._stop_event = stop_event or threading.Event()

        self.log.log(f"Loading library: {os.path.basename(library_path)}")
        self.lib = load_library(library_path, max_library_peaks=settings.max_library_peaks)

        # Sorted keys by precursor m/z for window lookup
        self.lib_keys_sorted = sorted(self.lib.keys())

        # Build robust mappings keyed by the *internal* libraryIdx tag used in peaks
        self.lib_idx_to_key: Dict[int, Tuple[float, str]] = {}
        self.lib_peaks_by_idx: Dict[int, List[Tuple[float, float]]] = {}

        for key, entry in self.lib.items():
            peaks = entry.get("peaks", []) or []
            if not peaks:
                continue
            idx = int(peaks[0][2])
            self.lib_idx_to_key[idx] = key
            self.lib_peaks_by_idx[idx] = [(float(m), float(i)) for (m, i, _) in peaks]

        # Fallback if something went wrong
        if len(self.lib_idx_to_key) == 0:
            # assume tag order matches sorted keys
            for idx, key in enumerate(self.lib_keys_sorted):
                peaks = self.lib[key].get("peaks", []) or []
                self.lib_idx_to_key[idx] = key
                self.lib_peaks_by_idx[idx] = [(float(m), float(i)) for (m, i, _) in peaks]

        self.log.log(f"Library entries: {len(self.lib_keys_sorted)}")

        # Decoy scaling for FDR if target:decoy is not 1:1
        n_dec = sum(int(v.get("isDecoy", 0)) for v in self.lib.values())
        n_tar = len(self.lib) - n_dec
        self.decoy_scale = float(n_tar / max(n_dec, 1)) if n_dec else 1.0
        if self.decoy_scale != 1.0:
            self.log.log(f"Decoy scaling: targets={n_tar} decoys={n_dec} scale={self.decoy_scale:.4g}")


    def request_stop(self) -> None:
        self._stop_event.set()

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise StopRequested("Stop requested by user")

    def _decorate_scores(self, score_df: pd.DataFrame, query_meta: dict) -> pd.DataFrame:
        """Add peptide/protein metadata columns to a scored candidate dataframe."""
        df = score_df.copy()

        def lib_key(i: int):
            return self.lib_idx_to_key.get(int(i), None)

        df["MzLIB"] = df["libraryIdx"].apply(lambda i: float(lib_key(i)[0]) if lib_key(i) else float("nan"))
        df["peptide"] = df["libraryIdx"].apply(lambda i: str(lib_key(i)[1]) if lib_key(i) else "")
        df["protein"] = df["libraryIdx"].apply(
            lambda i: str(self.lib[lib_key(i)]["proteinName"]) if lib_key(i) in self.lib else ""
        )
        df["isDecoy"] = df["libraryIdx"].apply(
            lambda i: int(self.lib[lib_key(i)].get("isDecoy", 0)) if lib_key(i) in self.lib else 0
        )
        df["zLIB"] = df["libraryIdx"].apply(
            lambda i: int(self.lib[lib_key(i)].get("precursorCharge", 0)) if lib_key(i) in self.lib else 0
        )

        df["scan"] = df["queryIdx"].astype(int)
        df["MzEXP"] = df["scan"].apply(lambda s: float(query_meta[s].precursor_mz) if s in query_meta else float("nan"))
        df["totalWindowWidth"] = df["scan"].apply(lambda s: float(query_meta[s].window_width) if s in query_meta else float("nan"))
        df["retentionTime"] = df["scan"].apply(lambda s: str(query_meta[s].retention_time) if s in query_meta else "")
        df["CompensationVoltage"] = df["scan"].apply(lambda s: str(query_meta[s].cv) if s in query_meta else "")

        return df

    def _apply_output_capping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply output-only capping per scan (does NOT affect scoring/FDR)."""
        if df is None or len(df) == 0:
            return df
        d = df.sort_values("finalScore", ascending=False).reset_index(drop=True)
        if bool(self.settings.keep_best_per_scan):
            return d.groupby("scan", as_index=False).head(1).reset_index(drop=True)
        n = int(self.settings.max_hits_per_scan)
        if n > 0:
            return d.groupby("scan", as_index=False).head(n).reset_index(drop=True)
        return d

    def run_file(self, mzxml_path: str) -> Dict[str, pd.DataFrame]:
        self.log.log(f"Reading query file: {os.path.basename(mzxml_path)}")
        self._check_stop()
        q = QueryMzxml(mzxml_path)
        windows = q.map_scans_to_windows()
        self.log.log(f"Total DIA windows: {len(windows)}")

        sizes = [len(v) for v in windows.values()]
        if sizes:
            med = int(pd.Series(sizes).median())
            self.log.log(f"Scans per window (min/median/max): {min(sizes)}/{med}/{max(sizes)}")

        query_meta = q.extract_metadata(top_k_peaks=self.settings.top_k_query_peaks)

        # ---- Optional: run-adaptive interference suppression (IDF-like fragment weighting) ----
        use_idf = bool(getattr(self.settings, "enable_idf_weighting", False))
        idf_bw = float(getattr(self.settings, "idf_bin_width_da", 0.01) or 0.01)
        idf_bin_counts: Dict[int, int] | None = {} if use_idf else None
        if idf_bin_counts is not None:
            self.log.log(f"IDF fragment weighting enabled (bin_width={idf_bw:g} Da)")

        all_match_dfs = []
        with q.iter_scans_by_id() as reader:
            for w_i, (mz_window, scan_ids) in enumerate(windows.items(), start=1):
                self._check_stop()
                lib_peaks = _pool_library_peaks(mz_window, self.lib, self.lib_keys_sorted)
                if not lib_peaks:
                    continue

                for pooled_query in _pool_query_peaks(
                    scan_ids, reader, query_meta, self.settings.top_k_query_peaks, self.settings.scan_chunk_size
                ):
                    self._check_stop()
                    if not pooled_query:
                        continue
                    # Update fragment-frequency bins from pooled query peaks (used for IDF-like weighting)
                    if idf_bin_counts is not None:
                        mzs = np.fromiter((p[0] for p in pooled_query), dtype=float)
                        if mzs.size:
                            bins = np.rint(mzs / idf_bw).astype(np.int64)
                            u, c = np.unique(bins, return_counts=True)
                            for bb, cc in zip(u.tolist(), c.tolist()):
                                idf_bin_counts[int(bb)] = idf_bin_counts.get(int(bb), 0) + int(cc)
                    mdf = match_library_to_query_pooled_spectra(lib_peaks, pooled_query, self.settings.ppm_tolerance)
                    if len(mdf) == 0:
                        continue
                    mdf = eliminate_low_count_matches(mdf, min_num_matches=self.settings.min_shared_peaks)
                    if len(mdf):
                        all_match_dfs.append(mdf)

        if not all_match_dfs:
            raise ValueError("No matches found. Check ppm tolerance, library coverage, and query file content.")

        match_df = pd.concat(all_match_dfs, ignore_index=True)
        self.log.log(f"Matched peak pairs: {len(match_df)}")

        # Build IDF-like weights (down-weight ubiquitous fragment m/z bins)
        idf_bin_to_w: Dict[int, float] | None = None
        idf_default_w: float = 1.0
        if idf_bin_counts is not None:
            smooth = float(getattr(self.settings, "idf_smoothing", 1.0) or 1.0)
            wmin = float(getattr(self.settings, "idf_w_min", 0.05) or 0.05)
            wmax = float(getattr(self.settings, "idf_w_max", 2.0) or 2.0)
            power = float(getattr(self.settings, "idf_power", 1.0) or 1.0)

            # Default weight for unseen bins (freq=0)
            idf_default_w = float(np.clip(1.0 / np.log1p(smooth), wmin, wmax))
            if power != 1.0:
                idf_default_w = float(idf_default_w ** power)

            idf_bin_to_w = {}
            for b, cnt in idf_bin_counts.items():
                w = float(np.clip(1.0 / np.log1p(float(cnt) + smooth), wmin, wmax))
                if power != 1.0:
                    w = float(w ** power)
                idf_bin_to_w[int(b)] = w

            self.log.log(f"IDF bins observed: {len(idf_bin_to_w)}")

        # ---- PASS 1 scoring (projected by default) ----
        self._check_stop()
        score_df = score_matches(
            match_df,
            query_meta=query_meta,
            ppm_tolerance=float(self.settings.ppm_tolerance),
            score_mode=getattr(self.settings, "score_mode", "macc"),
            library_peaks_by_idx=self.lib_peaks_by_idx,
            use_projected=bool(getattr(self.settings, "use_projected_scoring", True)),
            ppm_center=0.0,
            idf_bin_width_da=idf_bw,
            idf_bin_to_w=idf_bin_to_w,
            idf_default_w=idf_default_w,
        )
        self.log.log(f"Scored candidate PSMs (pass1): {len(score_df)}")

        # Decorate to compute q-values for correction fitting
        scored_pass1 = self._decorate_scores(score_df, query_meta=query_meta)

        ppm_center = 0.0
        ppm_tol2 = float(self.settings.ppm_tolerance)

        # ---- Optional ----
        self._check_stop()
        if bool(getattr(self.settings, "enable_mass_error_correction", True)):
            tmp = add_q_values(
                scored_pass1,
                is_decoy_col="isDecoy",
                score_col="finalScore",
                decoy_scale=self.decoy_scale,
            )
            fit_q = float(getattr(self.settings, "correction_fit_q", 0.01))
            confident = tmp[(tmp["q"] <= fit_q) & (tmp["isDecoy"] == 0)].copy()

            if len(confident) >= int(getattr(self.settings, "correction_min_psms", 200)):
                # Use median ppm per (libraryIdx,queryIdx) to avoid overweighting duplicates
                conf_pairs = confident[["libraryIdx", "queryIdx"]].drop_duplicates()
                m2 = match_df.merge(conf_pairs, on=["libraryIdx", "queryIdx"], how="inner")

                pair_ppm = m2.groupby(["libraryIdx", "queryIdx"])["ppmDifference"].median().to_numpy(dtype=float)
                center, sigma = _robust_center_scale(pair_ppm)

                n_sigma = float(getattr(self.settings, "correction_n_sigma", 3.0))
                min_tol = float(getattr(self.settings, "correction_min_tol", 4.0))
                tol = max(min_tol, n_sigma * sigma)
                tol = min(tol, float(self.settings.ppm_tolerance))  # never exceed pass1 tolerance

                if np.isfinite(center) and np.isfinite(tol) and tol > 0:
                    ppm_center = float(center)
                    ppm_tol2 = float(tol)

                    self.log.log(
                        f"Mass-error correction: center={ppm_center:+.3f} ppm, sigma≈{sigma:.3f}, tol2={ppm_tol2:.3f} (from {len(confident)} confident PSMs)"
                    )

                    match_df2 = match_df[np.abs(match_df["ppmDifference"] - ppm_center) <= ppm_tol2].reset_index(drop=True)
                    self.log.log(f"Matched peak pairs after correction filter: {len(match_df2)}")

                    # Re-apply min-shared after pass2 filtering (groups can shrink below threshold)
                    match_df2 = eliminate_low_count_matches(match_df2, min_num_matches=int(self.settings.min_shared_peaks))
                    self.log.log(f"Matched peak pairs after pass2 min-shared filter: {len(match_df2)}")

                    if len(match_df2) == 0:
                        self.log.log("Pass-2 produced no remaining matches after min-shared filtering; using pass-1 scores.")
                        scored = scored_pass1
                    else:
                        score_df2 = score_matches(
                            match_df2,
                            query_meta=query_meta,
                            ppm_tolerance=float(ppm_tol2),
                            score_mode=getattr(self.settings, "score_mode", "macc"),
                            library_peaks_by_idx=self.lib_peaks_by_idx,
                            use_projected=bool(getattr(self.settings, "use_projected_scoring", True)),
                            ppm_center=float(ppm_center),
                        )
                        self.log.log(f"Scored candidate PSMs (pass2): {len(score_df2)}")

                        scored = self._decorate_scores(score_df2, query_meta=query_meta)
                else:
                    scored = scored_pass1
            else:
                self.log.log(f"Mass-error correction skipped: only {len(confident)} confident PSMs at q<={fit_q:g}")
                scored = scored_pass1
        else:
            scored = scored_pass1

        # ---- ML rescoring (optional) ----
        psm_df = scored.sort_values("finalScore", ascending=False).reset_index(drop=True)
        score_col = "finalScore"

        self._check_stop()
        if bool(self.settings.use_ml_rescoring):
            ml_settings = RescoreSettings(
                enabled=True,
                iterations=int(self.settings.ml_iterations),
                enable_cv=bool(getattr(self.settings, 'ml_enable_cv', True)),
                cv_folds=int(getattr(self.settings, 'ml_cv_folds', 3)),
                min_psms=int(getattr(self.settings, 'ml_min_psms', 2000)),
                min_decoys=int(getattr(self.settings, 'ml_min_decoys', 200)),
                min_targets=int(getattr(self.settings, 'ml_min_targets', 200)),
                min_positives=int(getattr(self.settings, 'ml_min_positives', 200)),
                dedup_positives_by_peptide=bool(getattr(self.settings, 'ml_dedup_positives_by_peptide', True)),
            )
            ml_score, info = percolator_lite_rescore(
                psm_df,
                base_score_col="finalScore",
                is_decoy_col="isDecoy",
                scan_col="scan",
                settings=ml_settings,
                decoy_scale=self.decoy_scale,
            )
            psm_df["mlScore"] = ml_score

            # Log truthfully and summarize positives (CV vs non-CV)
            if not bool(info.get("enabled", False)):
                self.log.log(f"ML rescoring skipped: {info.get('reason', 'unknown')}")
                score_col = "finalScore"
            else:
                score_col = "mlScore"
                iters_req = int(info.get("iterations", int(self.settings.ml_iterations)))

                def _last_pos_summary(info: dict, iters_req: int):
                    import re
                    if bool(info.get("enable_cv", False)):
                        patt = re.compile(r"fold_(\d+)_iter_(\d+)_positives\Z")
                        rec = []
                        for k, v in info.items():
                            m = patt.match(str(k))
                            if m and isinstance(v, (int, np.integer)):
                                fold = int(m.group(1))
                                it = int(m.group(2))
                                rec.append((it, fold, int(v)))
                        if not rec:
                            return "n/a"
                        # Prefer requested iteration if present, otherwise use max available
                        its = [it for it, _, _ in rec]
                        use_it = iters_req if iters_req in its else max(its)
                        vals = [v for it, _, v in rec if it == use_it]
                        if not vals:
                            return "n/a"
                        mean = sum(vals) / len(vals)
                        return f"iter={use_it} cv_mean={mean:.1f} folds={len(vals)}"
                    else:
                        patt = re.compile(r"iter_(\d+)_positives\Z")
                        rec = []
                        for k, v in info.items():
                            m = patt.match(str(k))
                            if m and isinstance(v, (int, np.integer)):
                                it = int(m.group(1))
                                rec.append((it, int(v)))
                        if not rec:
                            return "n/a"
                        its = [it for it, _ in rec]
                        use_it = iters_req if iters_req in its else max(its)
                        for it, v in rec:
                            if it == use_it:
                                return f"iter={use_it} positives={v}"
                        return "n/a"

                pos_final = _last_pos_summary(info, iters_req)
                self.log.log(f"ML rescoring enabled. iterations={iters_req} positives_last={pos_final}")

        # ---- FDR/q on chosen score (NO per-scan cap before this point) ----
        self._check_stop()
        psm_df = add_q_values(psm_df, is_decoy_col="isDecoy", score_col=score_col, decoy_scale=self.decoy_scale)
        psm_q = filter_q(psm_df, q_cut=self.settings.q_cutoff)

        # ---- Quant-friendly peptide aggregates (from PSM table) ----
        # Keep per-peptide sums so protein tables can report summed intensity (useful for LFQ / quant).
        psm_stats = (
            psm_df.groupby("peptide", as_index=False)
            .agg(
                ionCountSumPSM=("ionCount", "sum"),
                ionCountMaxPSM=("ionCount", "max"),
                nPSM=("ionCount", "size"),
                nScans=("scan", "nunique"),
            )
        )

        # peptide-level: best hit per peptide
        pep_best = psm_df.sort_values(score_col, ascending=False).drop_duplicates("peptide").reset_index(drop=True)
        pep_best = add_q_values(pep_best, is_decoy_col="isDecoy", score_col=score_col, decoy_scale=self.decoy_scale)
        pep_q = filter_q(pep_best, q_cut=self.settings.q_cutoff)

        # Attach per-peptide quant aggregates
        pep_best = pep_best.merge(psm_stats, on="peptide", how="left")
        pep_q = pep_q.merge(psm_stats, on="peptide", how="left")


        # ---- Protein inference (IDPicker; optional) ----
        prot_cols = ["protein", "nPeptides", "peptides", "bestPeptideScore", "minPeptideQ", "ionCountSum", "ionCountSumPSM"]
        prot_df = pd.DataFrame(columns=prot_cols)
        prot_q = pd.DataFrame(columns=prot_cols)
        self._check_stop()
        if bool(getattr(self.settings, "enable_protein_inference", True)):
            prot_q_src = pep_q[pep_q["isDecoy"] == 0].reset_index(drop=True) if "isDecoy" in pep_q.columns else pep_q
            prot_q = build_protein_table_from_peptides(
                prot_q_src,
                score_col=score_col,
                q_col="q",
                protein_col="protein",
                peptide_col="peptide",
            )

            infer_q = float(getattr(self.settings, "protein_inference_q", float(self.settings.q_cutoff)))
            prot_src = pep_best.copy()
            if "q" in prot_src.columns:
                prot_src = prot_src[prot_src["q"] <= infer_q]
            if "isDecoy" in prot_src.columns:
                prot_src = prot_src[prot_src["isDecoy"] == 0]
            prot_src = prot_src.reset_index(drop=True)

            prot_df = build_protein_table_from_peptides(
                prot_src,
                score_col=score_col,
                q_col="q",
                protein_col="protein",
                peptide_col="peptide",
            )

        # Optional output-only capping (does not change counts above)
        psm_df_capped = self._apply_output_capping(psm_df)
        psm_q_capped = self._apply_output_capping(psm_q)

        return {
            "full_psm": psm_df,
            "full_psm_capped": psm_df_capped,
            "full_psm_qcut": psm_q,
            "full_psm_qcut_capped": psm_q_capped,
            "peptide": pep_best,
            "peptide_qcut": pep_q,
            "protein": prot_df,
            "protein_qcut": prot_q,
            "ppm_center": pd.DataFrame({"ppm_center":[ppm_center], "ppm_tolerance_pass2":[ppm_tol2]}),
        }

    def run(self, mzxml_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for fp in mzxml_paths:
            self._check_stop()
            try:
                res = self.run_file(fp)
            except StopRequested:
                self.log.log("Stop requested. Ending run early.")
                break
            results[os.path.basename(fp)] = res
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(fp))[0]

                qtag = f"q{self.settings.q_cutoff:g}"

                # Minimal outputs (default)
                if bool(getattr(self.settings, "write_minimal_outputs", True)):
                    res["full_psm_qcut"].to_csv(os.path.join(output_dir, f"{stem}.psm.{qtag}.tsv"), sep="\t", index=False)
                    res["peptide_qcut"].to_csv(os.path.join(output_dir, f"{stem}.peptide.{qtag}.tsv"), sep="\t", index=False)
                    res["protein_qcut"].to_csv(os.path.join(output_dir, f"{stem}.protein.{qtag}.tsv"), sep="\t", index=False)

                # Optional extras
                if bool(getattr(self.settings, "write_full_psm", False)):
                    res["full_psm"].to_csv(os.path.join(output_dir, f"{stem}.psm.tsv"), sep="\t", index=False)
                if bool(getattr(self.settings, "write_full_peptide", False)):
                    res["peptide"].to_csv(os.path.join(output_dir, f"{stem}.peptide.tsv"), sep="\t", index=False)
                if bool(getattr(self.settings, "write_full_protein", False)):
                    res["protein"].to_csv(os.path.join(output_dir, f"{stem}.protein.tsv"), sep="\t", index=False)

                if bool(getattr(self.settings, "write_capped_outputs", False)):
                    res["full_psm_capped"].to_csv(os.path.join(output_dir, f"{stem}.psm.capped.tsv"), sep="\t", index=False)
                    res["full_psm_qcut_capped"].to_csv(os.path.join(output_dir, f"{stem}.psm.{qtag}.capped.tsv"), sep="\t", index=False)

                if bool(getattr(self.settings, "write_ppm_correction", False)):
                    res["ppm_center"].to_csv(os.path.join(output_dir, f"{stem}.ppm_correction.tsv"), sep="\t", index=False)

        # Optional: LFQ across multiple runs
        if output_dir and bool(getattr(self.settings, "enable_lfq", False)) and len(results) >= 2 and (not self._stop_event.is_set()):
            qtag = f"q{self.settings.q_cutoff:g}"

            # ---- Peptide matrix ----
            pep_ids = set()
            pep_maps: Dict[str, dict] = {}
            for run_name, r in results.items():
                df = r.get("peptide_qcut", pd.DataFrame())
                if len(df) == 0:
                    pep_maps[run_name] = {}
                    continue
                if "isDecoy" in df.columns:
                    df = df[df["isDecoy"] == 0]
                val_col = "ionCountSumPSM" if "ionCountSumPSM" in df.columns else "ionCount"
                m = pd.Series(df[val_col].values, index=df["peptide"]).to_dict()
                pep_maps[run_name] = m
                pep_ids.update(m.keys())
            pep_ids = sorted(pep_ids)
            pep_mat = pd.DataFrame(
                [[float(pep_maps[r].get(p, 0.0) or 0.0) for p in pep_ids] for r in pep_maps.keys()],
                index=list(pep_maps.keys()),
                columns=pep_ids,
            )
            pep_mat.to_csv(os.path.join(output_dir, f"LFQ.peptide_matrix.{qtag}.tsv"), sep="\t")

            # ---- Protein-peptide map (union across runs) ----
            prot_to_peps: Dict[str, set] = {}
            for _, r in results.items():
                pdf = r.get("protein_qcut", pd.DataFrame())
                if len(pdf) == 0:
                    continue
                for _, row in pdf.iterrows():
                    prot = str(row.get("protein", ""))
                    if not prot:
                        continue
                    peps = _parse_semicolon_list(row.get("peptides", ""))
                    if prot not in prot_to_peps:
                        prot_to_peps[prot] = set()
                    prot_to_peps[prot].update(peps)

            prot_ids = sorted(prot_to_peps.keys())
            if prot_ids:
                # Protein SUM (simple, robust)
                prot_sum = []
                for run_name in pep_mat.index:
                    row = []
                    for prot in prot_ids:
                        peps = prot_to_peps.get(prot, set())
                        row.append(float(pep_mat.loc[run_name, list(peps)].sum()) if peps else 0.0)
                    prot_sum.append(row)
                prot_sum_df = pd.DataFrame(prot_sum, index=pep_mat.index, columns=prot_ids)
                prot_sum_df.to_csv(os.path.join(output_dir, f"LFQ.protein_sum.{qtag}.tsv"), sep="\t")

                # Protein MaxLFQ (optional)
                if str(getattr(self.settings, "lfq_method", "sum")).strip().lower() == "maxlfq":
                    min_d = int(getattr(self.settings, "lfq_min_num_differences", 2))
                    prot_max = {}
                    for prot in prot_ids:
                        peps = sorted(prot_to_peps.get(prot, set()))
                        if not peps:
                            continue
                        sub = pep_mat.reindex(columns=peps).fillna(0.0)
                        X = sub.to_numpy(dtype=float)
                        X_log = np.zeros_like(X)
                        m = X > 0
                        X_log[m] = np.log(X[m])

                        prot_log = _maxlfq(X_log, min_num_differences=min_d, tolerance=-10.0)
                        prot_vals = np.exp(prot_log)
                        prot_vals[(prot_vals == 1.0) | (~np.isfinite(prot_vals))] = 0.0
                        prot_max[prot] = prot_vals

                    prot_max_df = pd.DataFrame(index=pep_mat.index)
                    for prot in prot_ids:
                        if prot in prot_max:
                            prot_max_df[prot] = prot_max[prot]
                        else:
                            prot_max_df[prot] = 0.0
                    prot_max_df.to_csv(os.path.join(output_dir, f"LFQ.protein_maxlfq.{qtag}.tsv"), sep="\t")

        return results
