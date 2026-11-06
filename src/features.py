"""Common preprocessing for oz-cancer-subtype (7조).

Rules
- fit on train only; apply the same transform to test
- every feature here is computed per patient (row) or from train-only lists -> no leakage
"""
import re
import numpy as np
import pandas as pd

SEED = 42
SILENT_RE = re.compile(r"^([A-Z])\d+\1$")


def split_cols(df: pd.DataFrame):
    """Return gene column names (everything except ID / SUBCLASS)."""
    return [c for c in df.columns if c not in ("ID", "SUBCLASS")]


def to_binary(df: pd.DataFrame, gene_cols) -> pd.DataFrame:
    """P1: WT -> 0, any mutation -> 1 (uint8)."""
    return (df[gene_cols] != "WT").astype(np.uint8)


def keep_informative(binary_train: pd.DataFrame):
    """P2: columns mutated in at least one TRAIN patient (fit on train only)."""
    return binary_train.columns[binary_train.sum(axis=0) > 0].tolist()


def tokens(cell: str):
    return [] if cell == "WT" else cell.split(" ")


def mutation_type_counts(df: pd.DataFrame, gene_cols) -> pd.DataFrame:
    """P4 + Top8-#1: per-patient counts by mutation type (row-wise, no leakage)."""
    out = {"n_mut": [], "n_silent": [], "n_missense": [], "n_stop": [], "n_fs": []}
    for row in df[gene_cols].itertuples(index=False):
        n = s = m = st = fs = 0
        for cell in row:
            for t in tokens(cell):
                n += 1
                if "fs" in t:
                    fs += 1
                elif "*" in t:
                    st += 1
                elif SILENT_RE.match(t):
                    s += 1
                else:
                    m += 1
        out["n_mut"].append(n); out["n_silent"].append(s); out["n_missense"].append(m)
        out["n_stop"].append(st); out["n_fs"].append(fs)
    f = pd.DataFrame(out, index=df.index)
    f["log_n_mut"] = np.log1p(f["n_mut"])
    for c in ("n_silent", "n_missense", "n_stop", "n_fs"):
        f[c.replace("n_", "r_")] = f[c] / f["n_mut"].clip(lower=1)
    return f


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """E5 pipeline. Returns X_train, X_test (same columns)."""
    genes = split_cols(train)
    b_tr, b_te = to_binary(train, genes), to_binary(test, genes)
    cols = keep_informative(b_tr)
    X_tr = pd.concat([b_tr[cols], mutation_type_counts(train, genes)], axis=1)
    X_te = pd.concat([b_te[cols], mutation_type_counts(test, genes)], axis=1)
    return X_tr, X_te
