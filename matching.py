from __future__ import annotations

import numpy as np
print(">>> USING NEW matching.py (native projected)")


import pandas as pd
from numba import njit

NEITHER = 0
INC_LIB = 1
INC_QUERY = 2


@njit(cache=True, fastmath=True)
def ppm_diff(ref_mz: float, target_mz: float) -> float:
    return (ref_mz - target_mz) * 1e6 / ref_mz


@njit(cache=True, fastmath=True)
def within(ppm: float, tol: float) -> bool:
    return abs(ppm) <= tol


@njit(cache=True, fastmath=True)
def step_decision(lib_mz: float, q_mz: float, tol: float) -> int:
    p = ppm_diff(lib_mz, q_mz)
    if within(p, tol):
        return NEITHER
    if lib_mz > q_mz:
        return INC_QUERY
    return INC_LIB


@njit(cache=True, fastmath=True)
def numba_match_count(lib, q, ppm_tolerance) -> int:
    mz_idx, tag_idx = 0, 2
    count = 0
    lib_i = 0
    q_i = 0

    while lib_i < len(lib) and q_i < len(q):
        inc = step_decision(lib[lib_i][mz_idx], q[q_i][mz_idx], ppm_tolerance)
        if inc == INC_LIB:
            lib_i += 1
            continue
        if inc == INC_QUERY:
            q_i += 1
            continue

        # Within tolerance: scan forward in library while still within tolerance for this query peak
        tmp_lib = lib_i
        q_mz = q[q_i][mz_idx]
        while tmp_lib < len(lib):
            p = ppm_diff(lib[tmp_lib][mz_idx], q_mz)
            if not within(p, ppm_tolerance):
                break
            count += 1
            tmp_lib += 1

        q_i += 1

    return count


@njit(cache=True, fastmath=True)
def numba_match_fill(lib, q, ppm_tolerance, out):
    mz_idx, inten_idx, tag_idx = 0, 1, 2
    k = 0
    lib_i = 0
    q_i = 0

    while lib_i < len(lib) and q_i < len(q):
        inc = step_decision(lib[lib_i][mz_idx], q[q_i][mz_idx], ppm_tolerance)
        if inc == INC_LIB:
            lib_i += 1
            continue
        if inc == INC_QUERY:
            q_i += 1
            continue

        tmp_lib = lib_i
        q_tag = q[q_i][tag_idx]
        q_int = q[q_i][inten_idx]
        q_mz = q[q_i][mz_idx]

        while tmp_lib < len(lib):
            p = ppm_diff(lib[tmp_lib][mz_idx], q_mz)
            if not within(p, ppm_tolerance):
                break

            out[k, 0] = lib[tmp_lib][tag_idx]     # libraryIdx
            out[k, 1] = lib[tmp_lib][inten_idx]   # libraryIntensity
            out[k, 2] = q_tag                     # queryIdx
            out[k, 3] = q_int                     # queryIntensity
            out[k, 4] = q_mz                      # queryMz
            out[k, 5] = p                         # ppmDifference
            k += 1
            tmp_lib += 1

        q_i += 1

    return k


def match_library_to_query_pooled_spectra(library_peaks, query_peaks, ppm_tolerance: float) -> pd.DataFrame:
    """
    Fast matching without chunk caps and without drop_duplicates().
    Output columns unchanged.
    """
    cols = ["libraryIdx", "libraryIntensity", "queryIdx", "queryIntensity", "queryMz", "ppmDifference"]
    if len(library_peaks) == 0 or len(query_peaks) == 0:
        return pd.DataFrame(columns=cols)

    lib = np.asarray(library_peaks, dtype=np.float64)
    q = np.asarray(query_peaks, dtype=np.float64)

    # Pass 1: count
    n = numba_match_count(lib, q, float(ppm_tolerance))
    if n == 0:
        return pd.DataFrame(columns=cols)

    # Pass 2: fill
    out = np.empty((n, 6), dtype=np.float64)
    k = numba_match_fill(lib, q, float(ppm_tolerance), out)

    # k should equal n, but slice safely
    out = out[:k]

    df = pd.DataFrame(out, columns=cols)
    df[["libraryIdx", "queryIdx"]] = df[["libraryIdx", "queryIdx"]].astype(np.int64)
    return df.reset_index(drop=True)


def eliminate_low_count_matches(match_df: pd.DataFrame, min_num_matches: int = 3) -> pd.DataFrame:
    """
    Much faster than groupby().filter(lambda ...).
    Uses NumPy unique counting on packed keys.
    """
    if len(match_df) == 0:
        return match_df

    lib = match_df["libraryIdx"].to_numpy(dtype=np.int64, copy=False)
    q = match_df["queryIdx"].to_numpy(dtype=np.int64, copy=False)

    # Pack into a single 64-bit key: (lib<<32) | query
    keys = (lib << 32) ^ q

    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    mask = counts[inv] >= int(min_num_matches)
    return match_df.loc[mask].reset_index(drop=True)
