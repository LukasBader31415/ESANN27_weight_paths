"""Versuchsplan SG7, Teil 5: Uebersicht Seeds 3–15 (Sigmoid xi 10 gegen Identitaet) und Bewertung der Erwartungen E1–E5.

Aufruf (Repo-Root): python experiments/TN_ksweep/eval_seeds.py [--out results/public/2026-10-20_SG7_seeds3-15]
Nutzt die Funktionen aus eval_sigmoid.py (load_fits, per_fit, paired); eval_sigmoid.py selbst bleibt unveraendert.
Quellen je Seed: Seed 3 Lauf 1 (SG_explore_H0) + TN/LR_explore als Identitaet (Train-BA aus den Hub-Archiven
cache/hist_5dfd.gz), Seed 4 Lauf 2 (SG_confirm_H0s4), Seeds 5–15 SG6/SG7 (SG_seed<s>_H0). Nur xi = 10.
Schwellen und Bewertungsregel aus experiments/TN_ksweep/predictions_sg7.yaml (festgelegt vor dem Lauf).
Ausgabe: <out>/uebersicht.md, kennzahlen_seeds.csv (je Seed, Ebene, Plan), erwartungen_sg7.csv (je Seed E1–E5),
ohne_erwartung.csv (Zerlegung Strang A, Luecken, Ecke Identitaet je Seed). Keine Findings, keine Wiki-Aenderung.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "TN_ksweep"))
import run_tn as tn  # noqa: E402
import eval_sigmoid as ev  # noqa: E402

PUB = ROOT / "results" / "public"
CFG = ROOT / "experiments" / "TN_ksweep"
SEEDS = {
    3: dict(config=CFG / "config_sig_explore.yaml", run=PUB / "2026-10-20_SG_explore_H0",
            refs=[PUB / "2026-10-15_TN_ksweep_H0" / "fits.csv", PUB / "2026-10-15_LR_explore_H0" / "fits.csv"],
            archives=[ROOT / "cache" / "hist_5dfd.gz"]),
    4: dict(config=CFG / "config_sig_confirm_s4.yaml", run=PUB / "2026-10-20_SG_confirm_H0s4", refs=[], archives=[]),
    **{s: dict(config=CFG / f"config_sig_s{s}.yaml", run=PUB / f"2026-10-20_SG_seed{s}_H0", refs=[], archives=[])
       for s in range(5, 16)},
}
PLAN_ID = {"const_0.003": "e0.003_const", "const_0.01": "e0.01_const", "decay_0.003_t133": "e0.003_t133"}
REFS = ["frozen", "frozen_oracle_G", "frozen_oracle_CW"]


def seed_tables(seed, spec):
    """Gepaarte Tabelle (xi 10) und Referenz-Fits (frozen*) eines Seeds."""
    cfg = tn.load_config(spec["config"])
    f = ev.load_fits(spec["run"], spec["refs"])
    f = f[f.K.isin([2, 6, 12])]
    keep = f.variant.str.contains("_sig10_") | f.variant.isin(
        [ev.ident_name(l, p) for l in ("G", "CW") for p in PLAN_ID.values()] + REFS)
    f = f[keep]
    snaps = ev.Snapshots(ROOT / "cache" / "toy_H0" / tn.cache_signature(cfg), spec["archives"])
    need = {("hist", x.variant, int(x.K), int(x.rep)) for _, x in f.iterrows()
            if pd.isna(x.get("ba_train_end", np.nan)) or pd.isna(x.get("winkel_netto_c0", np.nan))}
    missing = snaps.preload(need)
    D = tn.build_data(cfg) if need else None
    pf = ev.per_fit(f, D, snaps)
    pp = ev.paired(pf)
    pp = pp[pp.xi == 10].copy()
    pp["fe"] = pp.ba_end_sig - pp.ba_best_val_id                   # Fruehstopp-Vergleich (E3)
    return pp, pf, len(missing)


def cells(pp, seed):
    """Mittel je Ebene und Plan ueber K {2,6,12} x 5 reps."""
    g = pp.groupby(["level", "plan"]).agg(
        n=("rep", "size"), d_end=("d_end", "mean"), fe=("fe", "mean"), bv_minus_end=("bv_minus_end_sig", "mean"),
        bv_minus_end_id=("bv_minus_end_id", "mean"), ecke_echt=("ecke_echt_sig", "mean"),
        ecke_echt_id=("ecke_echt_id", "mean"), luecke=("luecke_sig", "mean"), luecke_id=("luecke_id", "mean"),
        ba_end_sig=("ba_end_sig", "mean"), ba_end_id=("ba_end_id", "mean"),
        ba_best_val_id=("ba_best_val_id", "mean"), d_best_val=("d_best_val", "mean")).reset_index()
    g.insert(0, "seed", seed)
    return g


def check(v, cond, tol):
    m = re.fullmatch(r"\s*(>=|<=|>)\s*([+-]?[0-9.]+)\s*", cond)
    op, thr = m.group(1), float(m.group(2))
    if np.isnan(v):
        return False
    return v >= thr - tol if op == ">=" else (v <= thr + tol if op == "<=" else v > thr + tol)


METRIC = {"E1": "d_end", "E2": "d_end", "E3": "fe", "E4": "bv_minus_end", "E5": "ecke_echt"}


def erwartungen(cg, pred):
    """Je Seed und Erwartung: Werte aller Zellen und erfuellt (alle Zellen)."""
    tol = float(pred["bewertung"]["toleranz"])
    rows = []
    for seed, g in cg.groupby("seed"):
        r = dict(seed=seed)
        for name, e in pred["erwartungen"].items():
            key = name.split("_")[0]
            lv = e["ebene"]
            conds = ({lv: e["erfuellt_wenn"]} if isinstance(lv, str) else
                     dict(lv) if isinstance(lv, dict) else {l: e["erfuellt_wenn"] for l in lv})
            vals, ok = [], True
            for l, cond in conds.items():
                for p in e["plaene"]:
                    v = float(g[(g.level == l) & (g.plan == PLAN_ID[p])][METRIC[key]].mean())
                    vals.append(f"{l} {p}: {v:+.3f}")
                    ok &= check(v, cond, tol)
            r[key] = ok
            r[f"{key}_werte"] = "; ".join(vals)
        rows.append(r)
    return pd.DataFrame(rows)


def ohne_erwartung(pp_all, pf_all):
    """Zerlegung Strang A (const .003), Luecken (Sigmoid, Identitaet, frozen), Ecke Identitaet je Seed."""
    rows = []
    for seed, pp in pp_all.items():
        pf = pf_all[seed]
        fz = pf[pf.variant == "frozen"]
        for lvl in ("G", "CW"):
            q = pp[(pp.level == lvl) & (pp.plan == "e0.003_const")]
            fo = pf[pf.variant == f"frozen_oracle_{lvl}"]
            rows.append(dict(seed=seed, level=lvl, frozen_end=fz.ba_end.mean(), id_end=q.ba_end_id.mean(),
                             id_best_val=q.ba_best_val_id.mean(), sig_end=q.ba_end_sig.mean(),
                             frozen_oracle_end=fo.ba_end.mean(), plus_gewichte_id=q.ba_end_id.mean() - fz.ba_end.mean(),
                             luecke_sig=q.luecke_sig.mean(), luecke_id=q.luecke_id.mean(), luecke_frozen=fz.luecke.mean(),
                             ecke_echt_id_const003=q.ecke_echt_id.mean(),
                             ecke_echt_id_const01=pp[(pp.level == lvl) & (pp.plan == "e0.01_const")].ecke_echt_id.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PUB / "2026-10-20_SG7_seeds3-15"))
    ap.add_argument("--predictions", default=str(CFG / "predictions_sg7.yaml"))
    args = ap.parse_args()
    pred = yaml.safe_load(Path(args.predictions).read_text(encoding="utf-8"))
    pp_all, pf_all, cg, miss = {}, {}, [], {}
    for seed, spec in SEEDS.items():
        pp, pf, n_miss = seed_tables(seed, spec)
        pp_all[seed], pf_all[seed], miss[seed] = pp, pf, n_miss
        cg.append(cells(pp, seed))
        print(f"Seed {seed:2d}: {len(pp)} Paare (xi 10), fehlende Snapshots {n_miss}", flush=True)
    cg = pd.concat(cg, ignore_index=True)
    er = erwartungen(cg, pred)
    oe = ohne_erwartung(pp_all, pf_all)
    neu = set(pred["bewertung"]["seeds"])
    bew = []
    for key in METRIC:
        n = int(er[er.seed.isin(neu)][key].sum())
        bew.append(dict(erwartung=key, erfuellt_seeds_9_15=f"{n}/{len(neu)}",
                        urteil="gestützt" if n >= 6 else "nicht gestützt",
                        erfuellt_seeds_3_8=f"{int(er[~er.seed.isin(neu)][key].sum())}/{int((~er.seed.isin(neu)).sum())}"))
    bew = pd.DataFrame(bew)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cg.to_csv(out / "kennzahlen_seeds.csv", index=False, float_format="%.5f")
    er.to_csv(out / "erwartungen_sg7.csv", index=False)
    oe.to_csv(out / "ohne_erwartung.csv", index=False, float_format="%.5f")
    bew.to_csv(out / "bewertung_sg7.csv", index=False)
    mark = lambda s: f"{s} (Herkunft)" if s not in neu else str(s)
    erm = er.copy()
    erm["seed"] = erm.seed.map(mark)
    for k in METRIC:
        erm[k] = erm[k].map({True: "erfüllt", False: "nicht erfüllt"})
    L = ["# SG7 · Seeds 3–15: Sigmoid ξ 10 gegen Identität, Erwartungen E1–E5\n",
         f"Erwartungen und Regel: `{Path(args.predictions).relative_to(ROOT).as_posix()}` (festgelegt vor dem Lauf). "
         "Je Seed Mittel über K {2, 6, 12} × 5 reps, gepaart je (K, rep), Test-BA. Bewertet werden nur die Seeds 9–15; "
         "Seeds 3–8 (Herkunft der Erwartungen) sind getrennt markiert und nicht eingerechnet. Toleranz 1e-9.\n",
         "Quellen: Seed 3 Lauf 1 + TN/LR_explore (Train-BA der Identität aus Hub-Snapshots), Seed 4 Lauf 2, Seeds 5–15 SG6/SG7. "
         "Fehlende Snapshots je Seed: " + ", ".join(f"{s}: {n}" for s, n in miss.items()) + ".\n",
         "## Bewertung E1–E5 (Seeds 9–15)\n", tn._md(bew),
         "\n## Je Seed\n", tn._md(erm[["seed"] + list(METRIC)]),
         "\n## Werte je Seed und Zelle\n", tn._md(erm[["seed"] + [f"{k}_werte" for k in METRIC]]),
         "\n## Ohne Erwartung (nur beschreibend): Zerlegung Strang A (const .003), Lücken, Ecke Identität\n",
         tn._md(oe),
         "\n## Kennzahlen je Seed, Ebene, Plan\n", tn._md(cg)]
    (out / "uebersicht.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(bew.to_string(index=False))
    print(f"geschrieben: {out}")


if __name__ == "__main__":
    main()
