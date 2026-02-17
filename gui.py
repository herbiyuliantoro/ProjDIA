from __future__ import annotations

import json
import os
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import asdict
from tkinter import filedialog, messagebox
from tkinter import ttk
from typing import Dict, Optional

import pandas as pd

from .engine import EngineSettings, ProjDIAEngine
from .utils import Logger, Timer


APP_TITLE = "ProjDIA"
SETTINGS_VERSION = 1


def _safe_float(x: str, default: float) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)


def _safe_int(x: str, default: int) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return int(default)


class ProjDIAApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master

        self._running = False
        self._log_lines = []
        self._log_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._engine: Optional[ProjDIAEngine] = None

        self.last_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.last_output_dir: Optional[str] = None

        self._build_style()
        self._build_vars()
        self._build_layout()

        self.pack(fill="both", expand=True)
        self._tick_clock()

    # ---------------- UI ----------------

    def _build_style(self):
        style = ttk.Style(self.master)
        # Prefer a modern ttk theme when available
        for theme in ("clam", "vista", "xpnative", "default"):
            try:
                style.theme_use(theme)
                break
            except Exception:
                continue

        self.master.title(APP_TITLE)
        self.master.geometry("1160x720")
        self.master.minsize(980, 640)

        # Slightly larger default font
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(size=10)
            self.master.option_add("*Font", default_font)
        except Exception:
            pass

    def _build_vars(self):
        # Inputs
        self.var_library = tk.StringVar(value="")
        self.var_output_dir = tk.StringVar(value="")
        self.var_mzxml_count = tk.StringVar(value="0 files selected")

        # Presets
        self.var_preset = tk.StringVar(value="Custom")

        # Settings vars (mirrors EngineSettings)
        self.v_ppm = tk.StringVar(value="30")
        self.v_min_shared = tk.StringVar(value="3")

        self.v_topk = tk.StringVar(value="0")
        self.v_max_lib_peaks = tk.StringVar(value="10")
        self.v_scan_chunk = tk.StringVar(value="50")

        self.v_qcut = tk.StringVar(value="0.01")

        self.v_keep_best = tk.BooleanVar(value=False)
        self.v_max_hits_scan = tk.StringVar(value="0")

        self.v_score_mode = tk.StringVar(value="macc")
        self.v_use_projected = tk.BooleanVar(value=True)

        self.v_mass_corr = tk.BooleanVar(value=True)
        self.v_corr_fit_q = tk.StringVar(value="0.01")
        self.v_corr_min_psms = tk.StringVar(value="200")
        self.v_corr_n_sigma = tk.StringVar(value="3.0")
        self.v_corr_min_tol = tk.StringVar(value="4.0")

        self.v_ml = tk.BooleanVar(value=True)
        self.v_ml_iters = tk.StringVar(value="4")

        self.v_protein = tk.BooleanVar(value=True)
        self.v_protein_q = tk.StringVar(value="0.01")

        # Output controls
        self.v_write_minimal = tk.BooleanVar(value=True)
        self.v_write_full_psm = tk.BooleanVar(value=False)
        self.v_write_full_peptide = tk.BooleanVar(value=False)
        self.v_write_full_protein = tk.BooleanVar(value=False)
        self.v_write_capped_outputs = tk.BooleanVar(value=False)
        self.v_write_ppm_table = tk.BooleanVar(value=False)

        # LFQ across multiple runs
        self.v_enable_lfq = tk.BooleanVar(value=False)
        self.v_lfq_method = tk.StringVar(value="sum")
        self.v_lfq_min_diffs = tk.StringVar(value="2")

        # Results UI
        self.var_result_file = tk.StringVar(value="")
        self.var_result_table = tk.StringVar(value="protein_qcut")
        self.var_clock = tk.StringVar(value="")

    def _build_layout(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill="x", padx=14, pady=(14, 6))

        title = ttk.Label(header, text="ProjDIA", font=("TkDefaultFont", 16, "bold"))
        title.pack(side="left")

        right = ttk.Frame(header)
        right.pack(side="right")
        ttk.Label(right, text="Preset:").pack(side="left", padx=(0, 6))
        preset = ttk.Combobox(
            right,
            textvariable=self.var_preset,
            values=["Custom", "Direct infusion"],
            state="readonly",
            width=18,
        )
        preset.pack(side="left")
        preset.bind("<<ComboboxSelected>>", lambda e: self._apply_preset(self.var_preset.get()))

        ttk.Button(right, text="Load settings…", command=self._load_settings_json).pack(side="left", padx=(12, 0))
        ttk.Button(right, text="Save settings…", command=self._save_settings_json).pack(side="left", padx=(8, 0))

        # Notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=14, pady=(6, 14))

        self.tab_inputs = ttk.Frame(self.nb)
        self.tab_params = ttk.Frame(self.nb)
        self.tab_advanced = ttk.Frame(self.nb)
        self.tab_run = ttk.Frame(self.nb)
        self.tab_results = ttk.Frame(self.nb)

        self.nb.add(self.tab_inputs, text="Inputs")
        self.nb.add(self.tab_params, text="Parameters")
        self.nb.add(self.tab_advanced, text="Advanced")
        self.nb.add(self.tab_run, text="Run & Log")
        self.nb.add(self.tab_results, text="Results")

        self._build_inputs_tab()
        self._build_params_tab()
        self._build_advanced_tab()
        self._build_run_tab()
        self._build_results_tab()

    def _tick_clock(self):
        # Live clock shown in the Run tab header
        try:
            self.var_clock.set(time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
        self.master.after(500, self._tick_clock)

    def _build_inputs_tab(self):
        root = self.tab_inputs
        root.columnconfigure(0, weight=1)

        # Library
        lf = ttk.Labelframe(root, text="Library")
        lf.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        lf.columnconfigure(0, weight=1)

        entry = ttk.Entry(lf, textvariable=self.var_library)
        entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ttk.Button(lf, text="Browse…", command=self._pick_library).grid(row=0, column=1, padx=(0, 10), pady=10)

        # mzXML list
        lf2 = ttk.Labelframe(root, text="mzXML files")
        lf2.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        root.rowconfigure(1, weight=1)
        lf2.columnconfigure(0, weight=1)
        lf2.rowconfigure(0, weight=1)

        self.mzxml_list = tk.Listbox(lf2, height=10, activestyle="none")
        self.mzxml_list.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)

        btns = ttk.Frame(lf2)
        btns.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)

        ttk.Button(btns, text="Add…", command=self._add_mzxml).pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Remove", command=self._remove_selected_mzxml).pack(fill="x", pady=6)
        ttk.Button(btns, text="Clear", command=self._clear_mzxml).pack(fill="x", pady=6)

        ttk.Label(lf2, textvariable=self.var_mzxml_count).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        # Output
        lf3 = ttk.Labelframe(root, text="Output")
        lf3.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        lf3.columnconfigure(0, weight=1)

        out_entry = ttk.Entry(lf3, textvariable=self.var_output_dir)
        out_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ttk.Button(lf3, text="Browse…", command=self._pick_output_dir).grid(row=0, column=1, padx=(0, 10), pady=10)

        ttk.Label(
            lf3,
            text="Default outputs per run: psm.q<q> / peptide.q<q> / protein.q<q> (q from your q-value cutoff).",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        opts = ttk.Frame(lf3)
        opts.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        opts.columnconfigure(0, weight=1)

        ttk.Checkbutton(opts, text="Write minimal outputs only (3 files per run)", variable=self.v_write_minimal).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        extras = ttk.Frame(opts)
        extras.grid(row=1, column=0, sticky="ew")

        ttk.Checkbutton(extras, text="Also write full PSM table", variable=self.v_write_full_psm).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(extras, text="Also write full peptide table", variable=self.v_write_full_peptide).grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Checkbutton(extras, text="Also write full protein table", variable=self.v_write_full_protein).grid(row=0, column=2, sticky="w")

        extras2 = ttk.Frame(opts)
        extras2.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(extras2, text="Write capped tables (readability)", variable=self.v_write_capped_outputs).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(extras2, text="Write ppm correction summary", variable=self.v_write_ppm_table).grid(row=0, column=1, sticky="w")

        lfq = ttk.Labelframe(lf3, text="LFQ across multiple runs (optional)")
        lfq.grid(row=3, column=0, columnspan=2, sticky="ew", padx=0, pady=(0, 10))
        lfq.columnconfigure(2, weight=1)
        ttk.Checkbutton(lfq, text="Enable LFQ outputs (requires ≥2 mzXML files)", variable=self.v_enable_lfq).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4)
        )
        ttk.Label(lfq, text="Method").grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        ttk.Combobox(lfq, textvariable=self.v_lfq_method, values=["sum", "maxlfq"], state="readonly", width=10).grid(
            row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 8)
        )
        ttk.Label(lfq, text="Min shared peptides (MaxLFQ)").grid(row=1, column=2, sticky="e", padx=(0, 6), pady=(0, 8))
        ttk.Entry(lfq, textvariable=self.v_lfq_min_diffs, width=8).grid(row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 8))

    def _build_params_tab(self):
        root = self.tab_params
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # Core
        core = ttk.Labelframe(root, text="Core settings")
        core.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=(10, 8))
        root.rowconfigure(0, weight=1)
        core.columnconfigure(1, weight=1)

        r = 0
        r = self._add_labeled_entry(core, r, "PPM tolerance", self.v_ppm, "float", hint="default: 30")
        r = self._add_labeled_entry(core, r, "Min shared peaks", self.v_min_shared, "int", hint="default: 3")
        r = self._add_labeled_entry(core, r, "Top-K query peaks (0 = keep all)", self.v_topk, "int", hint="0 recommended for direct infusion")
        r = self._add_labeled_entry(core, r, "Max library peaks / precursor", self.v_max_lib_peaks, "int", hint="default: 10")
        r = self._add_labeled_entry(core, r, "Scan chunk size", self.v_scan_chunk, "int", hint="e.g., 50")
        r = self._add_labeled_entry(core, r, "q-value cutoff", self.v_qcut, "float", hint="e.g., 0.01")

        # Scoring
        scoring = ttk.Labelframe(root, text="Scoring & rescoring")
        scoring.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=(10, 8))
        scoring.columnconfigure(1, weight=1)

        rr = 0
        ttk.Label(scoring, text="Score mode").grid(row=rr, column=0, sticky="w", padx=10, pady=(10, 4))
        cmb = ttk.Combobox(scoring, textvariable=self.v_score_mode, values=["macc", "enhanced"], state="readonly", width=16)
        cmb.grid(row=rr, column=1, sticky="w", padx=10, pady=(10, 4))
        rr += 1

        ttk.Checkbutton(scoring, text="Projected scoring (fragment-bin evidence)", variable=self.v_use_projected).grid(
            row=rr, column=0, columnspan=2, sticky="w", padx=10, pady=6
        )
        rr += 1

        ttk.Separator(scoring).grid(row=rr, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        rr += 1

        ttk.Checkbutton(scoring, text="Enable mass-error correction (2-pass)", variable=self.v_mass_corr).grid(
            row=rr, column=0, columnspan=2, sticky="w", padx=10, pady=6
        )
        rr += 1

        ttk.Checkbutton(scoring, text="Enable ML rescoring (Percolator-lite)", variable=self.v_ml).grid(
            row=rr, column=0, columnspan=2, sticky="w", padx=10, pady=6
        )
        rr += 1

        rr = self._add_labeled_entry(scoring, rr, "ML iterations", self.v_ml_iters, "int", hint="e.g., 4")

        ttk.Separator(scoring).grid(row=rr, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        rr += 1

        ttk.Checkbutton(scoring, text="Enable protein inference (IDPicker)", variable=self.v_protein).grid(
            row=rr, column=0, columnspan=2, sticky="w", padx=10, pady=6
        )
        rr += 1
        rr = self._add_labeled_entry(scoring, rr, "Protein inference peptide q", self.v_protein_q, "float", hint="peptides with q<=this go into IDPicker")

        # Output presentation
        out = ttk.Labelframe(root, text="Output presentation (does not affect FDR)")
        out.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        out.columnconfigure(1, weight=1)

        r2 = 0
        ttk.Checkbutton(out, text="Keep best hit per scan (for readability)", variable=self.v_keep_best).grid(
            row=r2, column=0, sticky="w", padx=10, pady=10
        )
        ttk.Label(out, text="or").grid(row=r2, column=1, sticky="w", pady=10)
        ttk.Label(out, text="Max hits / scan:").grid(row=r2, column=2, sticky="e", padx=(0, 6), pady=10)
        ttk.Entry(out, textvariable=self.v_max_hits_scan, width=10).grid(row=r2, column=3, sticky="w", padx=(0, 10), pady=10)

    def _build_advanced_tab(self):
        root = self.tab_advanced
        root.columnconfigure(0, weight=1)

        lf = ttk.Labelframe(root, text="Mass-error correction tuning")
        lf.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 10))
        lf.columnconfigure(1, weight=1)

        r = 0
        r = self._add_labeled_entry(lf, r, "Fit using target PSMs with q <=", self.v_corr_fit_q, "float", hint="default 0.01")
        r = self._add_labeled_entry(lf, r, "Minimum confident PSMs required", self.v_corr_min_psms, "int", hint="default 200")
        r = self._add_labeled_entry(lf, r, "Sigma multiplier (n_sigma)", self.v_corr_n_sigma, "float", hint="default 3.0")
        r = self._add_labeled_entry(lf, r, "Minimum pass-2 tolerance (ppm)", self.v_corr_min_tol, "float", hint="default 4.0")

        lf2 = ttk.Labelframe(root, text="Notes")
        lf2.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        root.rowconfigure(1, weight=1)
        msg = (
            
        )
        ttk.Label(lf2, text=msg, justify="left").pack(anchor="w", padx=10, pady=10)

    def _build_run_tab(self):
        root = self.tab_run
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.columnconfigure(2, weight=1)

        self.btn_run = ttk.Button(top, text="Run ProjDIA", command=self._on_run)
        self.btn_run.grid(row=0, column=0, sticky="w")

        self.btn_stop = ttk.Button(top, text="Stop", command=self._on_stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=10)

        ttk.Button(top, text="Open output folder", command=self._open_output_dir).grid(row=0, column=3, sticky="e")

        self.lbl_clock = ttk.Label(top, textvariable=self.var_clock, foreground="#555")
        self.lbl_clock.grid(row=0, column=4, sticky="e", padx=(10, 0))

        # Log box
        self.log_text = tk.Text(root, height=18, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_text.configure(state="disabled")

        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)
        ttk.Button(bottom, text="Clear log", command=self._clear_log).pack(side="left")
        ttk.Button(bottom, text="Copy log", command=self._copy_log).pack(side="left", padx=(8, 0))

    def _build_results_tab(self):
        root = self.tab_results
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        # Controls
        ctl = ttk.Frame(root)
        ctl.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ctl.columnconfigure(5, weight=1)

        ttk.Label(ctl, text="File:").grid(row=0, column=0, sticky="w")
        self.cmb_file = ttk.Combobox(ctl, textvariable=self.var_result_file, values=[], state="readonly", width=28)
        self.cmb_file.grid(row=0, column=1, sticky="w", padx=(6, 14))
        self.cmb_file.bind("<<ComboboxSelected>>", lambda e: self._refresh_results_view())

        ttk.Label(ctl, text="Table:").grid(row=0, column=2, sticky="w")
        self.cmb_table = ttk.Combobox(
            ctl,
            textvariable=self.var_result_table,
            values=["protein_qcut", "peptide_qcut", "full_psm_qcut", "protein", "peptide", "full_psm"],
            state="readonly",
            width=18,
        )
        self.cmb_table.grid(row=0, column=3, sticky="w", padx=(6, 14))
        self.cmb_table.bind("<<ComboboxSelected>>", lambda e: self._refresh_results_view())

        ttk.Button(ctl, text="Export table…", command=self._export_current_table).grid(row=0, column=4, sticky="e")

        # Summary
        self.lbl_summary = ttk.Label(root, text="Run a search to see results here.")
        self.lbl_summary.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

        # Table view
        frame = ttk.Frame(root)
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(frame, columns=(), show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=hsb.set)

        ttk.Label(
            root,
            text="Tip: protein_qcut is generated via IDPicker parsimony on q-filtered target peptides.",
        ).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 10))

    def _add_labeled_entry(self, parent, row: int, label: str, var: tk.StringVar, kind: str, hint: str = "") -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=(10 if row == 0 else 6, 2))
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, sticky="ew", padx=10, pady=(10 if row == 0 else 6, 2))
        if hint:
            ttk.Label(parent, text=hint, foreground="#666").grid(row=row + 1, column=1, sticky="w", padx=10, pady=(0, 2))
            return row + 2
        return row + 1

    # ---------------- Inputs ----------------

    def _pick_library(self):
        fp = filedialog.askopenfilename(
            title="Select spectral library",
            filetypes=[("Library files", "*.tsv *.csv *.mgf"), ("All files", "*.*")],
        )
        if fp:
            self.var_library.set(fp)

    def _pick_output_dir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.var_output_dir.set(d)

    def _add_mzxml(self):
        fps = filedialog.askopenfilenames(
            title="Select mzXML files",
            filetypes=[("mzXML files", "*.mzXML *.mzxml"), ("All files", "*.*")],
        )
        for fp in fps:
            self.mzxml_list.insert("end", fp)
        self._update_mzxml_count()

    def _remove_selected_mzxml(self):
        sel = list(self.mzxml_list.curselection())
        for idx in reversed(sel):
            self.mzxml_list.delete(idx)
        self._update_mzxml_count()

    def _clear_mzxml(self):
        self.mzxml_list.delete(0, "end")
        self._update_mzxml_count()

    def _update_mzxml_count(self):
        n = self.mzxml_list.size()
        self.var_mzxml_count.set(f"{n} file(s) selected")

    def _open_output_dir(self):
        d = self.var_output_dir.get().strip() or self.last_output_dir
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Open output folder", "No valid output folder available yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(d)  # type: ignore[attr-defined]
            elif os.name == "posix":
                # macOS uses 'open', linux uses 'xdg-open'
                import subprocess

                cmd = ["open", d] if sys.platform == "darwin" else ["xdg-open", d]
                subprocess.Popen(cmd)
        except Exception:
            messagebox.showinfo("Open output folder", f"Output folder:\n{d}")

    # ---------------- Settings ----------------

    def _apply_preset(self, preset: str):
        # Leave inputs unchanged; only touch settings.
        
        if preset == "Direct infusion":
            self.v_topk.set("0")
            self.v_min_shared.set("2")
            self.v_ppm.set("20")
            self.v_score_mode.set("macc")
            self.v_keep_best.set(False)
            self.v_max_hits_scan.set("0")
            self.v_use_projected.set(True)
            self.v_mass_corr.set(True)
            self.v_ml.set(True)
        elif preset == "DIA mixture":
            self.v_topk.set("200")
            self.v_min_shared.set("3")
            self.v_ppm.set("20")
            self.v_score_mode.set("enhanced")
            self.v_keep_best.set(False)
            self.v_max_hits_scan.set("100")
            self.v_use_projected.set(True)
            self.v_mass_corr.set(True)
            self.v_ml.set(True)
        else:
            # Custom: no action
            return

    def _gather_engine_settings(self) -> EngineSettings:
        s = EngineSettings(
            ppm_tolerance=_safe_float(self.v_ppm.get(), 30.0),
            min_shared_peaks=_safe_int(self.v_min_shared.get(), 3),
            top_k_query_peaks=_safe_int(self.v_topk.get(), 0),
            max_library_peaks=_safe_int(self.v_max_lib_peaks.get(), 10),
            scan_chunk_size=_safe_int(self.v_scan_chunk.get(), 50),
            q_cutoff=_safe_float(self.v_qcut.get(), 0.01),
            keep_best_per_scan=bool(self.v_keep_best.get()),
            max_hits_per_scan=_safe_int(self.v_max_hits_scan.get(), 0),
            score_mode=str(self.v_score_mode.get()).strip() or "macc",
            use_projected_scoring=bool(self.v_use_projected.get()),
            enable_mass_error_correction=bool(self.v_mass_corr.get()),
            correction_fit_q=_safe_float(self.v_corr_fit_q.get(), 0.01),
            correction_min_psms=_safe_int(self.v_corr_min_psms.get(), 200),
            correction_n_sigma=_safe_float(self.v_corr_n_sigma.get(), 3.0),
            correction_min_tol=_safe_float(self.v_corr_min_tol.get(), 4.0),
            use_ml_rescoring=bool(self.v_ml.get()),
            ml_iterations=_safe_int(self.v_ml_iters.get(), 4),
            enable_protein_inference=bool(self.v_protein.get()),
            protein_inference_q=_safe_float(self.v_protein_q.get(), 0.01),

            # output
            write_minimal_outputs=bool(self.v_write_minimal.get()),
            write_full_psm=bool(self.v_write_full_psm.get()),
            write_full_peptide=bool(self.v_write_full_peptide.get()),
            write_full_protein=bool(self.v_write_full_protein.get()),
            write_capped_outputs=bool(self.v_write_capped_outputs.get()),
            write_ppm_correction=bool(self.v_write_ppm_table.get()),

            # lfq
            enable_lfq=bool(self.v_enable_lfq.get()),
            lfq_method=str(self.v_lfq_method.get()).strip() or "sum",
            lfq_min_num_differences=_safe_int(self.v_lfq_min_diffs.get(), 2),
        )
        return s

    def _save_settings_json(self):
        settings = self._gather_engine_settings()
        payload = {
            "app": APP_TITLE,
            "settings_version": SETTINGS_VERSION,
            "engine_settings": asdict(settings),
        }
        fp = filedialog.asksaveasfilename(
            title="Save settings",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not fp:
            return
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        messagebox.showinfo("Saved", f"Settings saved:\n{fp}")

    def _load_settings_json(self):
        fp = filedialog.askopenfilename(title="Load settings", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not fp:
            return
        try:
            payload = json.load(open(fp, "r", encoding="utf-8"))
            es = payload.get("engine_settings", {})
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not read settings JSON.\n\n{e}")
            return

        # Apply known keys defensively
        def set_if(key, var):
            if key in es:
                var.set(str(es[key]) if isinstance(var, tk.StringVar) else bool(es[key]))

        set_if("ppm_tolerance", self.v_ppm)
        set_if("min_shared_peaks", self.v_min_shared)
        set_if("top_k_query_peaks", self.v_topk)
        set_if("max_library_peaks", self.v_max_lib_peaks)
        set_if("scan_chunk_size", self.v_scan_chunk)
        set_if("q_cutoff", self.v_qcut)
        if "keep_best_per_scan" in es:
            self.v_keep_best.set(bool(es["keep_best_per_scan"]))
        set_if("max_hits_per_scan", self.v_max_hits_scan)
        if "score_mode" in es:
            self.v_score_mode.set(str(es["score_mode"]))
        if "use_projected_scoring" in es:
            self.v_use_projected.set(bool(es["use_projected_scoring"]))
        if "enable_mass_error_correction" in es:
            self.v_mass_corr.set(bool(es["enable_mass_error_correction"]))
        set_if("correction_fit_q", self.v_corr_fit_q)
        set_if("correction_min_psms", self.v_corr_min_psms)
        set_if("correction_n_sigma", self.v_corr_n_sigma)
        set_if("correction_min_tol", self.v_corr_min_tol)
        if "use_ml_rescoring" in es:
            self.v_ml.set(bool(es["use_ml_rescoring"]))
        set_if("ml_iterations", self.v_ml_iters)

        if "enable_protein_inference" in es:
            self.v_protein.set(bool(es["enable_protein_inference"]))
        set_if("protein_inference_q", self.v_protein_q)

        # Output
        if "write_minimal_outputs" in es:
            self.v_write_minimal.set(bool(es["write_minimal_outputs"]))
        if "write_full_psm" in es:
            self.v_write_full_psm.set(bool(es["write_full_psm"]))
        if "write_full_peptide" in es:
            self.v_write_full_peptide.set(bool(es["write_full_peptide"]))
        if "write_full_protein" in es:
            self.v_write_full_protein.set(bool(es["write_full_protein"]))
        if "write_capped_outputs" in es:
            self.v_write_capped_outputs.set(bool(es["write_capped_outputs"]))
        if "write_ppm_correction" in es:
            self.v_write_ppm_table.set(bool(es["write_ppm_correction"]))

        # LFQ
        if "enable_lfq" in es:
            self.v_enable_lfq.set(bool(es["enable_lfq"]))
        if "lfq_method" in es:
            self.v_lfq_method.set(str(es["lfq_method"]))
        set_if("lfq_min_num_differences", self.v_lfq_min_diffs)

        self.var_preset.set("Custom")
        messagebox.showinfo("Loaded", f"Settings loaded:\n{fp}")

    # ---------------- Run ----------------

    def _on_stop(self):
        if not self._running:
            return
        # Cooperative cancellation: request the engine to stop at the next safe checkpoint.
        try:
            self._stop_event.set()
            if self._engine is not None:
                self._engine.request_stop()
        except Exception:
            pass
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._log_lock:
            self._log_lines.append(f"{ts} | Stop requested…")
        try:
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass

    def _on_run(self):
        if self._running:
            return

        library = self.var_library.get().strip()
        out_dir = self.var_output_dir.get().strip()
        mzxmls = list(self.mzxml_list.get(0, "end"))

        if not library or not os.path.isfile(library):
            messagebox.showerror("Missing input", "Please select a valid library file.")
            self.nb.select(self.tab_inputs)
            return
        if not mzxmls:
            messagebox.showerror("Missing input", "Please add at least one mzXML file.")
            self.nb.select(self.tab_inputs)
            return
        if not out_dir:
            messagebox.showerror("Missing input", "Please select an output directory.")
            self.nb.select(self.tab_inputs)
            return

        settings = self._gather_engine_settings()
        self._clear_log()
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._log_lock:
            self._log_lines.append(f"{start_ts} | Starting ProjDIA…")
        self.nb.select(self.tab_run)

        self._stop_event.clear()

        self._running = True
        self.btn_run.configure(state="disabled")
        try:
            self.btn_stop.configure(state="normal")
        except Exception:
            pass
        self.progress.start(10)

        t = threading.Thread(
            target=self._run_thread,
            args=(library, mzxmls, out_dir, settings),
            daemon=True,
        )
        t.start()
        self.master.after(100, self._poll_log)

    def _run_thread(self, library: str, mzxmls, out_dir: str, settings: EngineSettings):
        def cb(msg: str):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with self._log_lock:
                self._log_lines.append(f"{ts} | {msg}")

        log = Logger(cb=cb)

        try:
            os.makedirs(out_dir, exist_ok=True)
            engine = ProjDIAEngine(library_path=library, settings=settings, logger=log, stop_event=self._stop_event)
            self._engine = engine
            with Timer() as tmr:
                res = engine.run(mzxmls, output_dir=out_dir)
            cb(f"Done in {tmr.dt:.2f} s. Wrote outputs to: {out_dir}")
            self.last_results = res
            self.last_output_dir = out_dir
        except Exception as e:
            cb(f"ERROR: {e}")
            self.last_results = {}
        finally:
            self._engine = None
            self._running = False

    def _poll_log(self):
        # Move pending log lines into the text widget
        moved = []
        with self._log_lock:
            if self._log_lines:
                moved = self._log_lines[:]
                self._log_lines.clear()

        if moved:
            self.log_text.configure(state="normal")
            for line in moved:
                self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        if self._running:
            self.master.after(120, self._poll_log)
            return

        # finished
        self.progress.stop()
        self.btn_run.configure(state="normal")
        try:
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass

        # refresh results UI
        self._refresh_results_sources()
        self.nb.select(self.tab_results)

    def _refresh_results_sources(self):
        if not self.last_results:
            self.lbl_summary.configure(text="No results available (run failed or produced no IDs).")
            self.cmb_file.configure(values=[])
            self.var_result_file.set("")
            self._show_df(pd.DataFrame())
            return

        files = sorted(self.last_results.keys())
        self.cmb_file.configure(values=files)
        if not self.var_result_file.get():
            self.var_result_file.set(files[0])
        self._refresh_results_view()

    def _refresh_results_view(self):
        fp = self.var_result_file.get().strip()
        if not fp or fp not in self.last_results:
            return
        res = self.last_results[fp]

        qcut = _safe_float(self.v_qcut.get(), 0.01)
        psm_n = len(res.get("full_psm_qcut", []))
        pep_n = len(res.get("peptide_qcut", []))
        prot_n = len(res.get("protein_qcut", []))
        self.lbl_summary.configure(text=f"{fp}  |  q<={qcut:g}: PSMs={psm_n}  Peptides={pep_n}  Proteins={prot_n}")

        key = self.var_result_table.get().strip()
        df = res.get(key, pd.DataFrame())
        self._show_df(df)

    def _show_df(self, df: pd.DataFrame, max_rows: int = 500):
        # Clear existing
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree.delete(*self.tree.get_children())

        if df is None or len(df) == 0:
            self.tree["columns"] = ("(empty)",)
            self.tree.heading("(empty)", text="(empty)")
            self.tree.column("(empty)", width=300, stretch=True)
            return

        cols = list(df.columns)
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, stretch=True)

        view = df.head(max_rows)
        for _, row in view.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            self.tree.insert("", "end", values=vals)

    def _export_current_table(self):
        fp = self.var_result_file.get().strip()
        if not fp or fp not in self.last_results:
            messagebox.showinfo("Export", "No results to export.")
            return
        key = self.var_result_table.get().strip()
        df = self.last_results[fp].get(key, pd.DataFrame())
        if df is None or len(df) == 0:
            messagebox.showinfo("Export", "Selected table is empty.")
            return

        out = filedialog.asksaveasfilename(
            title="Export table as TSV",
            defaultextension=".tsv",
            filetypes=[("TSV", "*.tsv"), ("CSV", "*.csv")],
        )
        if not out:
            return
        sep = "\t" if out.lower().endswith(".tsv") else ","
        df.to_csv(out, sep=sep, index=False)
        messagebox.showinfo("Exported", f"Saved:\n{out}")

    # ---------------- Log helpers ----------------

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _copy_log(self):
        txt = self.log_text.get("1.0", "end").strip()
        if not txt:
            return
        self.master.clipboard_clear()
        self.master.clipboard_append(txt)
        messagebox.showinfo("Copied", "Log copied to clipboard.")


def main():
    root = tk.Tk()
    app = ProjDIAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
