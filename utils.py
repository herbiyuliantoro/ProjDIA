from __future__ import annotations
from dataclasses import dataclass
from typing import List
import time

@dataclass
class Timer:
    t0: float = None
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0

class Logger:
    # Tiny logger that can write to GUI callback and/or stdout.
    def __init__(self, cb=None):
        self.cb = cb
    def log(self, msg: str):
        if self.cb:
            self.cb(msg)
        else:
            print(msg, flush=True)

def format_protein_string_to_list(protein_group: str) -> List[str]:
    # Supports formats like "3/P1/P2/P3" or "P1;P2" or "P1,P2"
    if protein_group is None:
        return []
    s = str(protein_group).strip()
    if not s:
        return []
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if parts and parts[0].isdigit():
            parts = parts[1:]
        return parts
    return [s]

def format_protein_list_to_string(proteins: List[str]) -> str:
    proteins = [p for p in proteins if p]
    return f"{len(proteins)}/" + "/".join(proteins) if proteins else ""
