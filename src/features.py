"""Common preprocessing for dacon-cancer-subtype-oz-team7 (7조).

Rules
- fit on train only; apply the same transform to test
- every feature here is computed per patient (row) or from train-only lists -> no leakage
- feature names use only [A-Za-z0-9_] so LightGBM/XGBoost accept them

Pipeline (E7, group-CV Macro F1 0.429 with untuned XGBoost):
  P1 binary 0/1 -> P2 drop genes never mutated in train -> P4 per-patient counts
  -> Top8-1 mutation-type counts/ratios -> Top8-2 amino-acid substitution patterns
  -> Top8-5 driver-gene truncating/missense 2-channel
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
PAT = re.compile(r"^([A-Z])(\d+)([A-Z*])$")          # R273H / R213* / R895R
SUBS = ["E_X", "S_X", "R_X", "Q_X", "W_X",          # X = stop (*)
        "R_Q", "R_W", "R_H", "G_D", "G_V", "P_S", "P_L"]
_SUB_KEY = {s: (s[0], "*" if s.endswith("_X") else s[-1]) for s in SUBS}
REFS = Path(__file__).resolve().parent.parent / "refs"


# ---------- column helpers ----------
def gene_columns(df: pd.DataFrame):
    """Gene columns = everything except ID / SUBCLASS. Always pass this explicitly."""
    return [c for c in df.columns if c not in ("ID", "SUBCLASS")]


def profile_hash(df: pd.DataFrame, genes):
    """Group id for StratifiedGroupKFold: identical mutation profiles share a group."""
    return df[genes].apply(lambda r: hashlib.md5("|".join(r.values).encode()).hexdigest(), axis=1).values


def load_driver_genes(genes):
    """Bailey 2018 driver list restricted to genes that exist in our columns (157/299)."""
    drv = pd.read_csv(REFS / "drivers_bailey2018.csv")
    return [g for g in drv["gene"] if g in set(genes)]


# ---------- P1 / P2 ----------
def fill_missing(df: pd.DataFrame, genes) -> pd.DataFrame:
    """test.csv has 237 empty cells (128 patients, 25 genes); train has none.
    We treat an empty cell as 'no information' = WT. Call this once right after read_csv."""
    out = df.copy()
    out[genes] = out[genes].fillna("WT")
    return out


def to_binary(df: pd.DataFrame, genes) -> pd.DataFrame:
    return (df[genes].notna() & (df[genes] != "WT")).astype(np.uint8)


def keep_informative(binary_train: pd.DataFrame):
    """Columns mutated in >=1 TRAIN patient. Fit on train (or train fold) only."""
    return binary_train.columns[binary_train.sum(axis=0) > 0].tolist()


# ---------- row-wise engineered features ----------
def classify_token(tok: str):
    """Return (kind, orig_aa, new_aa). kind in {T: truncating, M: missense, S: silent, O: other}."""
    if "fs" in tok:
        return "T", None, None
    if tok.endswith("*"):
        m = PAT.match(tok)
        return "T", (m.group(1) if m else None), "*"
    m = PAT.match(tok)
    if m:
        return ("S" if m.group(1) == m.group(3) else "M"), m.group(1), m.group(3)
    return "O", None, None          # del, 91_92NH>KY, etc.


def row_features(cells, genes, driver_genes):
    """Features for ONE patient. `cells` = iterable of that patient's gene values in `genes` order."""
    n = ns = nm = nt = nfs = 0
    sub = {s: 0 for s in SUBS}
    trunc = {g: 0 for g in driver_genes}
    mis = {g: 0 for g in driver_genes}
    dset = trunc.keys()
    for g, cell in zip(genes, cells):
        if cell == "WT" or not isinstance(cell, str):   # NaN (test only) -> treated as WT
            continue
        for tok in cell.split(" "):
            n += 1
            kind, a, b = classify_token(tok)
            if kind == "T":
                nt += 1
                if "fs" in tok:
                    nfs += 1
            elif kind == "S":
                ns += 1
            elif kind == "M":
                nm += 1
            if a is not None:
                for s, (sa, sb) in _SUB_KEY.items():
                    if a == sa and b == sb:
                        sub[s] += 1
            if g in dset:
                if kind == "T":
                    trunc[g] = 1
                elif kind == "M":
                    mis[g] = 1
    d = max(n, 1)
    out = {"n_mut": n, "log_n_mut": float(np.log1p(n)),
           "n_silent": ns, "n_missense": nm, "n_trunc": nt, "n_fs": nfs,
           "r_silent": ns / d, "r_missense": nm / d, "r_trunc": nt / d, "r_fs": nfs / d}
    for s in SUBS:
        out["sub_" + s] = sub[s]
        out["subr_" + s] = sub[s] / d
    for g in driver_genes:
        out["trunc_" + g] = trunc[g]
        out["mis_" + g] = mis[g]
    return out


def engineered(df: pd.DataFrame, genes, driver_genes) -> pd.DataFrame:
    rows = [row_features(r, genes, driver_genes) for r in df[genes].itertuples(index=False)]
    return pd.DataFrame(rows, index=df.index)


# ---------- full pipeline ----------
def build_features(train: pd.DataFrame, test: pd.DataFrame, level: str = "E7"):
    """Return X_train, X_test with identical columns.
    level: 'E5' (binary + n_mut) | 'E6a' (+type) | 'E6b' (+substitution) | 'E7' (+driver 2ch)
    """
    genes = gene_columns(train)
    train, test = fill_missing(train, genes), fill_missing(test, genes)
    drivers = load_driver_genes(genes) if level == "E7" else []
    b_tr, b_te = to_binary(train, genes), to_binary(test, genes)
    cols = keep_informative(b_tr)                       # train only
    f_tr, f_te = engineered(train, genes, drivers), engineered(test, genes, drivers)
    keep = ["n_mut"]
    if level in ("E6a", "E6b", "E7"):
        keep += ["log_n_mut", "n_silent", "n_missense", "n_trunc", "n_fs",
                 "r_silent", "r_missense", "r_trunc", "r_fs"]
    if level in ("E6b", "E7"):
        keep += [c for c in f_tr.columns if c.startswith("sub")]
    if level == "E7":
        keep += [c for c in f_tr.columns if c.startswith(("trunc_", "mis_"))]
    X_tr = pd.concat([b_tr[cols], f_tr[keep]], axis=1)
    X_te = pd.concat([b_te[cols], f_te[keep]], axis=1)
    return X_tr, X_te
