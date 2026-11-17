"""Tabelle 1 als LaTeX (booktabs) aus results/public/2026-10-20_SG9_seeds3-17/tabelle1.csv (eval_sg9.py).

Zeilen G, CW; Spalten Δ gegenüber frozen (Test-BA am Ende) für Identität Ende, Identität Best_val, Sigmoid Ende,
Orakel als Mittel [min, max] über die Datensätze, darunter die Anzahl Datensätze über frozen. Zahlen auf drei
Stellen ohne führende Null (.053). Steilheit der Sigmoide heißt β (Festlegung F01).
Optionen (Voreinstellung aus): wins=True – Zeile „sigmoid wins“ (Anzahl Datensätze, auf denen Sigmoid Ende die
jeweilige Spalte schlägt; Quelle tabelle1_seeds.csv); compact=True – nur Mittel, Kopf zweistufig.
"""
from pathlib import Path

import pandas as pd

from . import data as dt
from .layout import OUT

COLS = [("plus_id_end", "identity, end", "Identität Ende"),
        ("plus_id_best_val", "identity, best val.", "Identität Best_val"),
        ("plus_sig_end", r"sigmoid $\beta{=}10$, end", r"Sigmoid $\beta{=}10$ Ende"),
        ("plus_orakel", "oracle weights", "Orakel-Gewichte")]
# Vergleich „sigmoid wins“ je Spalte: (Spalte in tabelle1_seeds.csv, Vergleich); Sigmoid-Spalte gegen frozen
WINS = {"plus_id_end": ("id_end", ">"), "plus_id_best_val": ("id_best_val", ">"), "plus_sig_end": ("frozen", ">"),
        "plus_orakel": ("orakel", ">=")}
TOL = 1e-9


def f3(x, sign=True):
    """0.053 -> +.053, -0.012 -> -.012 (math mode in LaTeX)."""
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("0.", ".", 1)


def tabelle1_df(seeds="3–17"):
    """Zahlen der Tabelle (Mittel, min, max je Zelle; Anzahl über frozen) und Start/+ Prototypen für die Unterschrift."""
    t1, _ = dt.tabelle1()
    t = t1[t1.seeds == seeds]
    rows = []
    for lvl in ("G", "CW"):
        r = dict(level=lvl)
        for col, _, _ in COLS:
            x = t[(t.level == lvl) & (t.groesse == col)].iloc[0]
            r[col], r[col + "_min"], r[col + "_max"] = x.mittel, x["min"], x["max"]
            r[col + "_n"] = f"{int(x.n_ueber_frozen)}/{int(x.n)}"
        rows.append(r)
    base = {k: t[(t.level == "–") & (t.groesse == k)].iloc[0] for k in ("start", "plus_prototypen")}
    return pd.DataFrame(rows), base, int(t.n.iloc[0])


def _seed_list(seeds):
    a, b = (int(x) for x in seeds.split("–"))
    return list(range(a, b + 1))


def sigmoid_wins(seeds="3–17"):
    """Je Ebene und Spalte: Anzahl Datensätze, auf denen sig_end die Spalte schlägt (Identität Ende und Best_val: >,
    Orakel: ≥; in der Sigmoid-Spalte sig_end > frozen). Quelle tabelle1_seeds.csv (Seed-Mittel über K {2, 6, 12} × 5 reps)."""
    _, ts = dt.tabelle1()
    ts = ts[ts.seed.isin(_seed_list(seeds))]
    out = {}
    for lvl in ("G", "CW"):
        q = ts[ts.level == lvl]
        for col, (ref, op) in WINS.items():
            d = q.sig_end - q[ref]
            k = int((d >= -TOL).sum()) if op == ">=" else int((d > TOL).sum())
            out[(lvl, col)] = f"{k}/{len(q)}"
    return out


def tabelle1_latex(seeds="3–17", lang="en", out=None, name="tabelle1", wins=False, compact=False):
    """booktabs-Tabelle; schreibt <out>/<name>.tex und gibt (tex, DataFrame) zurück.
    wins=True: Zeile „sigmoid wins“ je Ebene; compact=True: nur Mittel (ohne [min, max]), Kopf zweistufig."""
    df, base, n = tabelle1_df(seeds)
    en = lang == "en"
    head = " & ".join(c[1] if en else c[2] for c in COLS)
    lab_n = rf"datasets $>$ frozen" if en else rf"Datensätze $>$ frozen"
    lab_w = "sigmoid wins" if en else "Sigmoid gewinnt"
    was = ("mean" if en else "Mittel") if compact else ("mean [min, max]" if en else "Mittel [min, max]")
    L = [r"\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}", r"\toprule",
         (r" & \multicolumn{%d}{c}{$\Delta$ test BA vs.\ frozen (%s over %d datasets)}\\" if en else
          r" & \multicolumn{%d}{c}{$\Delta$ Test-BA ggü.\ frozen (%s über %d Datensätze)}\\") % (len(COLS), was, n),
         r"\cmidrule(l){2-%d}" % (len(COLS) + 1)]
    if compact:
        L += [(r" & \multicolumn{2}{c}{identity} & sigmoid ($\beta{=}10$) & oracle\\" if en else
               r" & \multicolumn{2}{c}{Identität} & Sigmoid ($\beta{=}10$) & Orakel\\"),
              r"\cmidrule(lr){2-3}",
              (r" & end & best val. & end & weights\\" if en else r" & Ende & Best\_val & Ende & Gewichte\\"),
              r"\midrule"]
    else:
        L += [" & " + head + r"\\", r"\midrule"]
    sw = sigmoid_wins(seeds) if wins else None
    for _, r in df.iterrows():
        if compact:
            cells = [rf"${f3(r[c])}$" for c, _, _ in COLS]
        else:
            cells = [rf"${f3(r[c])}$\,[${f3(r[c + '_min'])}$, ${f3(r[c + '_max'])}$]" for c, _, _ in COLS]
        L.append(f"{r.level} & " + " & ".join(cells) + r"\\")
        L.append(r"\quad " + lab_n + " & " + " & ".join(r[c + "_n"] for c, _, _ in COLS) + r"\\")
        if wins:
            L.append(r"\quad " + lab_w + " & " + " & ".join(sw[(r.level, c)] for c, _, _ in COLS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(L) + "\n"
    note = (f"% start {base['start'].mittel:.3f} [{base['start']['min']:.3f}, {base['start']['max']:.3f}], "
            f"+ prototypes {f3(base['plus_prototypen'].mittel)} [{f3(base['plus_prototypen']['min'])}, "
            f"{f3(base['plus_prototypen']['max'])}] (seeds {seeds}; for the caption)\n")
    if wins:
        note += ("% sigmoid wins: sig_end > identity end / > identity best val. / > frozen (sigmoid column) / "
                 ">= oracle; per dataset, seed means\n")
    tex = note + tex
    out = Path(out) if out is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text(tex, encoding="utf-8")
    return tex, df
