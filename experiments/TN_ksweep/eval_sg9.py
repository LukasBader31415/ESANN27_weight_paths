"""Versuchsplan SG9, Teil 4: Tabelle 1 über 15 Datensätze, E1–E5-Kennzahlen Seeds 16/17 (beschreibend), K-Verlauf Seeds 3–5.

Aufruf (Repo-Root): python experiments/TN_ksweep/eval_sg9.py [--out results/public/2026-10-20_SG9_seeds3-17]
Nutzt eval_seeds.py (seed_tables, cells, erwartungen, ohne_erwartung) und eval_sigmoid.py unverändert; Seeds 16/17
werden hier ergänzt. Status: beschreibend. Das E1–E5-Urteil bleibt auf Seeds 9–15 (predictions_sg7.yaml,
bewertung_sg7.csv); Seeds 16/17 bekommen Kennzahlen, aber kein neues Urteil.

Ausgabe (<out>/):
- tabelle1_seeds.csv: je Seed und Ebene (Mittel über K {2, 6, 12} × 5 reps, const .003, Test-BA): Start, frozen,
  Identität Ende, Identität Best_val, Sigmoid ν 10 Ende, Orakel (frozen_oracle_*), dazu „+ Gewichte“ = X − frozen (Ende).
- tabelle1.csv: über die Seeds 3–17 (n = 15) und zur Kontrolle 3–15 (n = 13): Mittel, min, max, Anzahl Seeds mit
  X > frozen und X ≥ Orakel (Toleranz 1e-9).
- e1e5_seeds9-17.csv: E1–E5-Kennzahlen und „erfüllt“ je Seed 9–17; Spalte rolle (bewertet SG7 / beschreibend).
- kverlauf_s3-5.csv: je Seed 3/4/5, Ebene, K 2–20 (Mittel über 5 reps): Start, + Prototypen (frozen − Start),
  + Gewichte (X − frozen am Ende) für Identität Ende, Identität Best_val, Sigmoid Ende, Orakel; dazu absolute BA.
- kverlauf_mittel.csv: Mittel über die drei Seeds je Ebene und K, mit min/max über die Seeds.
- uebersicht.md. Keine Findings, keine Wiki-Änderung.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "TN_ksweep"))
import run_tn as tn  # noqa: E402
import eval_seeds as es  # noqa: E402

PUB = ROOT / "results" / "public"
CFG = ROOT / "experiments" / "TN_ksweep"
OUT = PUB / "2026-10-20_SG9_seeds3-17"
SEEDS = {**es.SEEDS,
         **{s: dict(config=CFG / f"config_sig_s{s}.yaml", run=PUB / f"2026-10-20_SG_seed{s}_H0", refs=[], archives=[])
            for s in (16, 17)}}
BEWERTET = set(range(9, 16))
TOL = 1e-9
# K-Verlauf (Teil A): Seed 3 Identitaet/Referenzen aus TN_ksweep, Sigmoid aus SG9_k_H0s3; Seeds 4/5 alles aus SG9
KV = {3: [PUB / "2026-10-20_SG9_k_H0s3" / "fits.csv", PUB / "2026-10-15_TN_ksweep_H0" / "fits.csv"],
      4: [PUB / "2026-10-20_SG9_k_H0s4" / "fits.csv"],
      5: [PUB / "2026-10-20_SG9_k_H0s5" / "fits.csv"]}
# Werte aus dem Folienentwurf 02b (13 Seeds, Chat 20.10.) zur Gegenprobe der Rechnung
KONTROLLE_13 = {("–", "start"): 0.700, ("–", "plus_prototypen"): 0.057,
                ("G", "plus_id_end"): -0.010, ("G", "plus_id_best_val"): 0.036, ("G", "plus_sig_end"): 0.052,
                ("G", "plus_orakel"): 0.031, ("CW", "plus_id_end"): 0.048, ("CW", "plus_id_best_val"): 0.068,
                ("CW", "plus_sig_end"): 0.075, ("CW", "plus_orakel"): 0.068}


def start_ba(spec):
    """Start = Test-BA bei Epoche 0 (gemeinsame Startmenge, gleiche Gewichte), aus frozen, K {2, 6, 12} × 5 reps."""
    from eval_sigmoid import load_fits
    f = load_fits(spec["run"], spec["refs"])
    f = f[(f.variant == "frozen") & f.K.isin([2, 6, 12])]
    if len(f) != 15:
        raise SystemExit(f"frozen K 2/6/12: {len(f)} Fits statt 15 in {spec['run']}")
    return float(f.ba_init.mean())


def tabelle1_seeds(oe, starts):
    rows = []
    for _, r in oe.iterrows():
        rows.append(dict(seed=int(r.seed), level=r.level, start=starts[int(r.seed)], frozen=r.frozen_end,
                         id_end=r.id_end, id_best_val=r.id_best_val, sig_end=r.sig_end, orakel=r.frozen_oracle_end,
                         plus_prototypen=r.frozen_end - starts[int(r.seed)],
                         plus_id_end=r.id_end - r.frozen_end, plus_id_best_val=r.id_best_val - r.frozen_end,
                         plus_sig_end=r.sig_end - r.frozen_end, plus_orakel=r.frozen_oracle_end - r.frozen_end))
    return pd.DataFrame(rows)


def tabelle1_agg(ts, seeds, label):
    t = ts[ts.seed.isin(seeds)]
    rows = []
    g0 = t[t.level == "G"]
    for k in ("start", "plus_prototypen"):
        rows.append(dict(seeds=label, n=len(g0), level="–", groesse=k, mittel=g0[k].mean(), min=g0[k].min(),
                         max=g0[k].max(), n_ueber_frozen=np.nan, n_ge_orakel=np.nan))
    for lvl in ("G", "CW"):
        q = t[t.level == lvl]
        for k, ab in (("plus_id_end", "id_end"), ("plus_id_best_val", "id_best_val"), ("plus_sig_end", "sig_end"),
                      ("plus_orakel", "orakel")):
            rows.append(dict(seeds=label, n=len(q), level=lvl, groesse=k, mittel=q[k].mean(), min=q[k].min(),
                             max=q[k].max(), n_ueber_frozen=int((q[k] > TOL).sum()),
                             n_ge_orakel=int((q[ab] >= q.orakel - TOL).sum()) if ab != "orakel" else np.nan,
                             ba_mittel=q[ab].mean()))
    return pd.DataFrame(rows)


def kverlauf():
    rows = []
    for seed, files in KV.items():
        f = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        f = f.drop_duplicates(["variant", "K", "rep"], keep="first")          # SG9 hat Vorrang (gleicher Cache)
        for lvl in ("G", "CW"):
            need = {"frozen", f"{lvl}_const", f"{lvl}_sig10_e0.003_const", f"frozen_oracle_{lvl}"}
            for K in range(2, 21):
                s = f[(f.K == K) & f.variant.isin(need)].set_index("variant")
                n = s.groupby(level=0).size()
                if set(n.index) != need or (n != 5).any():
                    raise SystemExit(f"Seed {seed} {lvl} K {K}: unvollständig {n.to_dict()}")
                m = s.groupby(level=0)[["ba_init", "ba_end", "ba_best_val"]].mean()
                fz, start = m.loc["frozen", "ba_end"], m.loc["frozen", "ba_init"]
                idn, sig, ora = m.loc[f"{lvl}_const"], m.loc[f"{lvl}_sig10_e0.003_const"], m.loc[f"frozen_oracle_{lvl}"]
                rows.append(dict(seed=seed, level=lvl, K=K, start=start, frozen=fz, id_end=idn.ba_end,
                                 id_best_val=idn.ba_best_val, sig_end=sig.ba_end, orakel=ora.ba_end,
                                 plus_prototypen=fz - start, plus_id_end=idn.ba_end - fz,
                                 plus_id_best_val=idn.ba_best_val - fz, plus_sig_end=sig.ba_end - fz,
                                 plus_orakel=ora.ba_end - fz))
    kv = pd.DataFrame(rows)
    cols = [c for c in kv.columns if c not in ("seed", "level", "K")]
    g = kv.groupby(["level", "K"])[cols]
    km = g.mean().add_suffix("").join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()
    return kv, km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out = Path(ap.parse_args().out)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pred = yaml.safe_load((CFG / "predictions_sg7.yaml").read_text(encoding="utf-8"))

    pp_all, pf_all, cg, starts, miss = {}, {}, [], {}, {}
    for seed, spec in SEEDS.items():
        pp, pf, n_miss = es.seed_tables(seed, spec)
        pp_all[seed], pf_all[seed], miss[seed] = pp, pf, n_miss
        cg.append(es.cells(pp, seed))
        starts[seed] = start_ba(spec)
        print(f"Seed {seed:2d}: {len(pp)} Paare (xi 10), fehlende Snapshots {n_miss}", flush=True)
    cg = pd.concat(cg, ignore_index=True)
    oe = es.ohne_erwartung(pp_all, pf_all)
    ts = tabelle1_seeds(oe, starts)
    t1 = pd.concat([tabelle1_agg(ts, list(range(3, 18)), "3–17"), tabelle1_agg(ts, list(range(3, 16)), "3–15")],
                   ignore_index=True)

    # Gegenprobe gegen den Folienentwurf 02b (13 Seeds, Werte auf 3 Stellen)
    kontrolle = []
    for (lvl, k), soll in KONTROLLE_13.items():
        ist = float(t1[(t1.seeds == "3–15") & (t1.level == lvl) & (t1.groesse == k)].mittel.iloc[0])
        kontrolle.append(dict(level=lvl, groesse=k, folie_02b=soll, rechnung=round(ist, 4),
                              gleich_auf_3_stellen=abs(round(ist, 3) - soll) < 1e-9))
    kontrolle = pd.DataFrame(kontrolle)

    er = es.erwartungen(cg[cg.seed.isin(range(9, 18))], pred)
    er.insert(1, "rolle", er.seed.map(lambda s: "bewertet (SG7)" if s in BEWERTET else "beschreibend, kein Urteil"))
    kv, km = kverlauf()

    out.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out / "tabelle1_seeds.csv", index=False, float_format="%.5f")
    t1.to_csv(out / "tabelle1.csv", index=False, float_format="%.5f")
    er.to_csv(out / "e1e5_seeds9-17.csv", index=False)
    kv.to_csv(out / "kverlauf_s3-5.csv", index=False, float_format="%.5f")
    km.to_csv(out / "kverlauf_mittel.csv", index=False, float_format="%.5f")

    t1m = t1.copy()
    for c in ("n_ueber_frozen", "n_ge_orakel"):
        t1m[c] = t1m.apply(lambda r: "–" if pd.isna(r[c]) else f"{int(r[c])}/{int(r.n)}", axis=1)
    erm = er.copy()
    for k in es.METRIC:
        erm[k] = erm[k].map({True: "erfüllt", False: "nicht erfüllt"})
    kv6 = kv[kv.K.isin([2, 6, 12, 20])][["seed", "level", "K", "start", "plus_prototypen", "plus_id_end",
                                          "plus_id_best_val", "plus_sig_end", "plus_orakel"]]
    L = ["# SG9 · Tabelle 1 über 15 Datensätze, Seeds 16/17, K-Verlauf Seeds 3–5 (beschreibend)\n",
         "Erzeugt von `experiments/TN_ksweep/eval_sg9.py`. Status: beschreibende Erweiterung (arbeitspakete.md „SG9“). "
         "Das E1–E5-Urteil bleibt auf Seeds 9–15 (`predictions_sg7.yaml`, `2026-10-20_SG7_seeds3-15/bewertung_sg7.csv`); "
         "Seeds 16/17 nur beschreibend. Kein Seed ausgeschlossen.\n",
         "Je Seed Mittel über K {2, 6, 12} × 5 reps, Lernrate const .003, Test-BA. „+ Gewichte“ = X − frozen am Ende; "
         "Orakel = frozen_oracle_* (Gewichte fest auf Orakel-Werten). Fehlende Snapshots je Seed: "
         + ", ".join(f"{s}: {n}" for s, n in miss.items()) + ".\n",
         "## Tabelle 1 (Seeds 3–17, n = 15; zum Vergleich 3–15, n = 13)\n", tn._md(t1m),
         "\n## Gegenprobe gegen den Folienentwurf 02b (13 Seeds)\n", tn._md(kontrolle),
         "\n## Je Seed\n", tn._md(ts),
         "\n## E1–E5-Kennzahlen Seeds 9–17 (Urteil nur 9–15, unverändert)\n",
         tn._md(erm[["seed", "rolle"] + list(es.METRIC)]),
         "\n### Werte je Seed und Zelle\n", tn._md(erm[["seed"] + [f"{k}_werte" for k in es.METRIC]]),
         "\n## K-Verlauf Seeds 3–5 (Auszug K 2, 6, 12, 20; alle K in `kverlauf_s3-5.csv`)\n", tn._md(kv6)]
    (out / "uebersicht.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(kontrolle.to_string(index=False))
    print(t1m[t1m.seeds == "3–17"].to_string(index=False))
    print(f"geschrieben: {out}")


if __name__ == "__main__":
    main()
