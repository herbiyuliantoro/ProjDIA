from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pyteomics import mzxml, mgf

ZKey = Tuple[float, str]  # (precursorMz, peptideName)
Peak = Tuple[float, float, int]  # (mz, intensity, tag)

NEW_COLUMNS = [
    "precursorMz",
    "peptideName",
    "peakMz",
    "peakIntensity",
    "precursorCharge",
    "identification",
    "proteinName",
    "fragmentType",
    "fragmentNumber",
    "fragmentCharge",
]

OLD_COLUMNS_SPECTRAST = [
    "PrecursorMz",
    "FullUniModPeptideName",
    "ProductMz",
    "LibraryIntensity",
    "PrecursorCharge",
    "transition_group_id",
    "ProteinName",
    "FragmentType",
    "FragmentSeriesNumber",
    "FragmentCharge",
]

OLD_COLUMNS_FRAGPIPE = [
    "PrecursorMz",
    "ModifiedPeptideSequence",
    "ProductMz",
    "LibraryIntensity",
    "PrecursorCharge",
    "PeptideSequence",
    "ProteinId",
    "FragmentType",
    "FragmentSeriesNumber",
    "FragmentCharge",
]

OLD_COLUMNS_PROSIT = [
    "PrecursorMz",
    "LabeledPeptide",
    "FragmentMz",
    "RelativeIntensity",
    "PrecursorCharge",
    "StrippedPeptide",
    "FragmentLossType",
    "FragmentType",
    "FragmentNumber",
    "FragmentCharge",
]

def _determine_library_source_from_header_line(first_line: str) -> str:
    """Infer table-library source from the header line (case-insensitive).

    Returns one of: spectrast, fragpipe, prosit, native.
    """
    line = (first_line or "").strip().lower()

    # "native" schema: already standardized (or close to it)
    if "peptidename" in line and ("peakmz" in line or "peak_mz" in line):
        return "native"

    # Known exporters
    if "transition_group_id" in line:
        return "spectrast"
    # FragPipe often has ProteinId, but allow common variants
    if "proteinid" in line or "protein_id" in line:
        return "fragpipe"
    if "relativeintensity" in line:
        return "prosit"

    raise ValueError("Unrecognized table library format.")

def _set_old_to_new_column_dict(file_path: str) -> Dict[str, str]:
    """Legacy helper kept for backward compatibility.

    The current loader does not rely on a strict position-based mapping anymore,
    but some older code paths may still call this.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    src = _determine_library_source_from_header_line(first_line)
    if src == "spectrast":
        return dict(zip(OLD_COLUMNS_SPECTRAST, NEW_COLUMNS))
    if src == "fragpipe":
        return dict(zip(OLD_COLUMNS_FRAGPIPE, NEW_COLUMNS))
    if src == "prosit":
        return dict(zip(OLD_COLUMNS_PROSIT, NEW_COLUMNS))
    if src == "native":
        # Identity mapping when the file already uses the standardized columns.
        return {c: c for c in NEW_COLUMNS}
    raise RuntimeError("Unexpected library source.")

def _remove_low_intensity_peaks(peaks: List[Peak], max_peak_num: int) -> List[Peak]:
    peaks = list(peaks)
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:max_peak_num]

def _create_peaks(mz: Iterable[float], inten: Iterable[float], tag: int) -> List[Peak]:
    mz = list(mz)
    inten = list(inten)
    return list(zip(mz, inten, [tag] * len(mz)))

def load_library(library_path: str, max_library_peaks: int = 50) -> Dict[ZKey, dict]:
    if library_path.lower().endswith((".tsv", ".csv")):
        return _load_table_library(library_path, max_library_peaks=max_library_peaks)
    if library_path.lower().endswith(".mgf"):
        return _load_mgf_library(library_path, max_library_peaks=max_library_peaks)
    raise ValueError("Unsupported library format. Use .csv, .tsv, or .mgf")

def _load_table_library(file_path: str, max_library_peaks: int = 50) -> Dict[ZKey, dict]:
    """Load a tabular library (.tsv/.csv) into the internal dictionary format.

    Goals:
      - Preserve modified-peptide identifiers when available (first-gen behavior)
      - Be tolerant to case and minor column-name variations (more robust)
    """
    sep = "\t" if file_path.lower().endswith(".tsv") else ","
    df_raw = pd.read_csv(file_path, sep=sep)

    # Resolve columns case-insensitively
    col_lut = {}
    for c in df_raw.columns:
        cl = str(c).strip().lower()
        if cl not in col_lut:
            col_lut[cl] = c

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand is None:
                continue
            key = str(cand).strip().lower()
            if key in col_lut:
                return col_lut[key]
        return None

    # Determine source from header line (more reliable than guessing from df columns)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
    src = _determine_library_source_from_header_line(header_line)

    # Build a standardized dataframe explicitly (more robust than strict rename+zip)
    if src == "native":
        c_prec = pick("precursorMz")
        c_pep = pick("peptideName")
        c_mz = pick("peakMz")
        c_int = pick("peakIntensity")
        c_z = pick("precursorCharge")
        c_id = pick("identification")
        c_prot = pick("proteinName")
        c_ft = pick("fragmentType")
        c_fn = pick("fragmentNumber")
        c_fc = pick("fragmentCharge")
    elif src == "spectrast":
        c_prec = pick("PrecursorMz")
        # Prefer modified peptide identifiers when available (first-gen behavior)
        c_pep = pick("FullUniModPeptideName", "ModifiedPeptideSequence", "PeptideSequence")
        c_mz = pick("ProductMz", "FragmentMz")
        c_int = pick("LibraryIntensity", "RelativeIntensity")
        c_z = pick("PrecursorCharge")
        c_id = pick("transition_group_id", "PeptideSequence", "FullUniModPeptideName")
        c_prot = pick("ProteinName", "ProteinId")
        c_ft = pick("FragmentType")
        c_fn = pick("FragmentSeriesNumber", "FragmentNumber")
        c_fc = pick("FragmentCharge")
    elif src == "fragpipe":
        c_prec = pick("PrecursorMz")
        c_pep = pick("ModifiedPeptideSequence", "FullUniModPeptideName", "PeptideSequence")
        c_mz = pick("ProductMz", "FragmentMz")
        c_int = pick("LibraryIntensity", "RelativeIntensity")
        c_z = pick("PrecursorCharge")
        # Often a stripped sequence; keep as "identification" like first-gen
        c_id = pick("PeptideSequence", "StrippedPeptide", "ModifiedPeptideSequence")
        c_prot = pick("ProteinId", "ProteinName")
        c_ft = pick("FragmentType")
        c_fn = pick("FragmentSeriesNumber", "FragmentNumber")
        c_fc = pick("FragmentCharge")
    elif src == "prosit":
        c_prec = pick("PrecursorMz")
        c_pep = pick("LabeledPeptide", "FullUniModPeptideName", "PeptideSequence")
        c_mz = pick("FragmentMz", "ProductMz")
        c_int = pick("RelativeIntensity", "LibraryIntensity")
        c_z = pick("PrecursorCharge")
        c_id = pick("StrippedPeptide", "PeptideSequence", "LabeledPeptide")
        # Prosit libs may not have a protein column; keep compatibility with legacy mapping
        c_prot = pick("ProteinName", "FragmentLossType")
        c_ft = pick("FragmentType", "FragmentLossType")
        c_fn = pick("FragmentNumber", "FragmentSeriesNumber")
        c_fc = pick("FragmentCharge")
    else:
        raise ValueError(f"Unsupported table library source: {src}")

    required = [("precursorMz", c_prec), ("peptideName", c_pep), ("peakMz", c_mz), ("peakIntensity", c_int)]
    missing_req = [name for name, col in required if not col]
    if missing_req:
        raise ValueError(f"Table library missing required column(s) for '{src}': {missing_req}")

    df = pd.DataFrame({
        "precursorMz": df_raw[c_prec].astype(float),
        "peptideName": df_raw[c_pep].astype(str),
        "peakMz": df_raw[c_mz].astype(float),
        "peakIntensity": df_raw[c_int].astype(float),
        "precursorCharge": df_raw[c_z].fillna(0).astype(int) if c_z else 0,
        "identification": df_raw[c_id].astype(str) if c_id else df_raw[c_pep].astype(str),
        "proteinName": df_raw[c_prot].astype(str) if c_prot else "",
        "fragmentType": df_raw[c_ft].astype(str) if c_ft else "",
        "fragmentNumber": df_raw[c_fn].fillna(0).astype(int) if c_fn else 0,
        "fragmentCharge": df_raw[c_fc].fillna(0).astype(int) if c_fc else 0,
    })

    # Drop rows with missing essentials
    df = df.replace({np.nan: None})
    df = df[df["precursorMz"].notnull() & df["peptideName"].notnull() & df["peakMz"].notnull() & df["peakIntensity"].notnull()]

    df["zkey"] = list(zip(df["precursorMz"].astype(float), df["peptideName"].astype(str)))
    keys = sorted(set(df["zkey"]))

    mzs = df.groupby("zkey")["peakMz"].apply(list).to_dict()
    intens = df.groupby("zkey")["peakIntensity"].apply(list).to_dict()
    meta = (
        df.drop_duplicates("zkey")
          .set_index("zkey")[["precursorCharge","identification","proteinName"]]
          .to_dict(orient="index")
    )

    out: Dict[ZKey, dict] = {}
    for lib_idx, key in enumerate(keys):
        peaks = _create_peaks(mzs[key], intens[key], lib_idx)
        peaks = _remove_low_intensity_peaks(peaks, max_library_peaks)
        peaks = [(m, i, lib_idx) for (m, i, _) in peaks]
        protein = str(meta[key].get("proteinName","") or "")
        ident = str(meta[key].get("identification","") or "")
        is_decoy = int(("decoy" in protein.lower()) or ("decoy" in ident.lower()))
        out[key] = {
            "precursorCharge": int(meta[key].get("precursorCharge", 0) or 0),
            "identification": ident,
            "proteinName": protein,
            "peaks": sorted(peaks),
            "peakCount": int(len(peaks)),
            "totalIntensity": float(sum([p[1] for p in peaks])) if peaks else 0.0,
            "isDecoy": is_decoy,
        }
    return out

def _load_mgf_library(file_path: str, max_library_peaks: int = 50) -> Dict[ZKey, dict]:
    spectra = list(mgf.read(file_path))
    tmp: Dict[ZKey, dict] = {}
    for lib_idx, spec in enumerate(spectra):
        params = spec.get("params", {})
        pepmass = params.get("pepmass", (None,))[0]
        seq = params.get("seq", None)
        if pepmass is None or seq is None:
            continue
        key: ZKey = (float(pepmass), str(seq))
        title = str(params.get("title","") or "")
        protein = str(params.get("protein","") or "")
        charge_field = params.get("charge", [0])
        charge = int(re.sub(r"[+\-]", "", str(charge_field[0]))) if charge_field else 0
        is_decoy = int(("DECOY" in title.upper()) or ("DECOY" in protein.upper()) or ("DECOY" in str(seq).upper()))

        peaks = _create_peaks(spec["m/z array"], spec["intensity array"], lib_idx)
        peaks = _remove_low_intensity_peaks(peaks, max_library_peaks)
        peaks = [(m, i, lib_idx) for (m, i, _) in peaks]
        tmp[key] = {
            "precursorCharge": charge,
            "identification": title,
            "proteinName": protein,
            "peaks": sorted(peaks),
            "peakCount": int(len(peaks)),
            "totalIntensity": float(sum([p[1] for p in peaks])) if peaks else 0.0,
            "isDecoy": is_decoy,
        }

    keys = sorted(tmp.keys())
    out: Dict[ZKey, dict] = {}
    for new_idx, key in enumerate(keys):
        v = tmp[key]
        peaks = [(m, inten, new_idx) for (m, inten, _) in v["peaks"]]
        out[key] = {**v, "peaks": sorted(peaks)}
    return out

@dataclass
class QueryScanMeta:
    scan: int
    spectrum_id: str
    precursor_mz: float
    window_width: float
    peaks_count: int
    retention_time: str
    cv: str
    top_tic: float
    tic: float

class QueryMzxml:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def map_scans_to_windows(self) -> Dict[Tuple[float, float, str], List[int]]:
        # Group scans by (windowCenterMz, windowWidth, compensationVoltage).
        # Including CV avoids mixing FAIMS planes (if present).
        mzWindowToScanId = defaultdict(list)
        with mzxml.read(self.file_path) as spectra:
            for spec in spectra:
                scan = int(spec["num"])
                if "precursorMz" not in spec:
                    continue

                # normalize nameValue fields into top-level keys (pyteomics convention)
                if "nameValue" in spec:
                    for k, v in spec["nameValue"].items():
                        spec[k] = v

                cv = str(spec.get("compensationVoltage", "")) if "compensationVoltage" in spec else ""
                prec = float(spec["precursorMz"][0]["precursorMz"])
                w = float(spec["precursorMz"][0]["windowWideness"])
                mzWindowToScanId[(prec, w, cv)].append(scan)
        return dict(mzWindowToScanId)

    def iter_scans_by_id(self):
        # pyteomics builds an index of spectrum 'id' strings when use_index=True
        return mzxml.read(self.file_path, use_index=True)

    def extract_metadata(self, top_k_peaks: int = 50) -> Dict[int, QueryScanMeta]:
        meta: Dict[int, QueryScanMeta] = {}
        with mzxml.read(self.file_path) as spectra:
            for spec in spectra:
                if "precursorMz" not in spec:
                    continue

                # normalize nameValue fields into top-level keys
                if "nameValue" in spec:
                    for k, v in spec["nameValue"].items():
                        spec[k] = v

                scan = int(spec["num"])
                spec_id = str(spec.get("id", str(scan)) or str(scan))

                prec = float(spec["precursorMz"][0]["precursorMz"])
                w = float(spec["precursorMz"][0]["windowWideness"])
                peaks_count = int(spec.get("peaksCount", 0))
                rt = str(spec.get("retentionTime", ""))

                cv = str(spec.get("compensationVoltage", "")) if "compensationVoltage" in spec else ""

                intens = np.asarray(spec.get("intensity array", []), dtype=float)
                tic = float(np.sum(intens)) if intens.size else 0.0
                if intens.size and top_k_peaks and top_k_peaks < intens.size:
                    idx = np.argpartition(intens, -top_k_peaks)[-top_k_peaks:]
                    top_tic = float(np.sum(intens[idx]))
                else:
                    top_tic = tic

                meta[scan] = QueryScanMeta(
                    scan=scan,
                    spectrum_id=spec_id,
                    precursor_mz=prec,
                    window_width=w,
                    peaks_count=peaks_count,
                    retention_time=rt,
                    cv=cv,
                    top_tic=top_tic,
                    tic=tic,
                )
        return meta

    @staticmethod
    def scan_to_peaks(spec: dict, scan_id: int, top_k_peaks: int = 50) -> List[Peak]:
        mz = np.asarray(spec.get("m/z array", []), dtype=float)
        inten = np.asarray(spec.get("intensity array", []), dtype=float)
        if mz.size == 0 or inten.size == 0:
            return []

        if top_k_peaks and top_k_peaks < inten.size:
            idx = np.argpartition(inten, -top_k_peaks)[-top_k_peaks:]
            mz = mz[idx]
            inten = inten[idx]

        keep = inten > 0
        mz = mz[keep]
        inten = inten[keep]
        tag = np.full_like(inten, int(scan_id), dtype=int)

        return list(zip(mz.tolist(), inten.tolist(), tag.tolist()))
