"""Versuchsplan SG, Teil D: Auswertung Sigmoid gegen Identitaet (lokal, fuer Lauf 1 und Lauf 2 gleich).

Aufruf (Repo-Root), Lauf 1 (Seed 3, Identitaet aus TN/LR_explore):
    python experiments/TN_ksweep/eval_sigmoid.py --config experiments/TN_ksweep/config_sig_explore.yaml \
        --run results/public/<lauf-1-id> \
        --ref results/public/2026-10-15_TN_ksweep_H0/fits.csv results/public/2026-10-15_LR_explore_H0/fits.csv \
        --archive cache/hist_5dfd.gz cache/traj_5dfd.gz [--archive <hist/traj-Archive von Lauf 1>]
Lauf 2 (Seed 4, Identitaet im selben Lauf), mit Kriterien:
    python experiments/TN_ksweep/eval_sigmoid.py --config experiments/TN_ksweep/config_sig_confirm_s4.yaml \
        --run results/public/<lauf-2-id> --criteria experiments/TN_ksweep/criteria_sg.yaml [--archive ...]

Je Sigmoid-Fit wird die Identitaet mit gleicher Ebene, gleichem Plan und gleichem (K, rep) gepaart (gleiche Startmenge):
  <lvl>_sig<xi>_e0.003_const -> <lvl>_const · <lvl>_sig<xi>_e0.003_t133 -> <lvl>_decay · sonst <lvl>_e<eta0>_<plan>.
Kennzahlen (ohne Schwellen): Anteil Fits in Ecke (K1: max alpha^2 > .99 in mindestens einer Klasse; dazu alt: min alpha^2 < 1e-3), Best_val - End,
BA_End(Sig) - BA_End(Id), Train-BA - Val-BA am Ende, BA_Best_val(Sig) - BA_Best_val(Id); Diagnosen: Epoche des
Minimums von E (ep_best_train) und mu_sum, gcos letzte 100 Epochen, Netto-Winkel, Kosinus alpha^2 zum Orakel am Ende.
Fehlt Train-BA bzw. Netto-Winkel in fits.csv (alte Laeufe), wird er aus den Snapshots (hist) nachgerechnet.
Kriterien: liest standardmaessig experiments/TN_ksweep/criteria_sg.yaml (Auftrag SG3; --ohne-kriterien schaltet ab).
Pruefung je xi und Ebene (Mittel ueber K und reps, je Plan der Datei). status vorlaeufig -> Vermerk „vorlaeufig –
Exploration“, keine Entscheidung (Lauf 1); status festgelegt -> Entscheidung H-S fuer das gewaehlte xi (Lauf 2).
Ausgabe: <out>/eval_sigmoid.md, eval_sigmoid_fits.csv, eval_sigmoid_agg.csv (Standard: --run-Ordner).
SG5 (Seed 4, Lauf 2 + SG_xi, neben Seed 3): --vergleich <Lauf-1-Ordner>/eval_sigmoid_fits.csv [--vergleich-seed 3]
    python experiments/TN_ksweep/eval_sigmoid.py --config experiments/TN_ksweep/config_sig_xi_s4.yaml \
        --run results/public/<sg5-id> --archive cache/hist_s4.gz \
        --vergleich results/public/<lauf-1-id>/eval_sigmoid_fits.csv
  -> zusaetzlich eval_sigmoid_vergleich.csv (je xi inkl. Identitaet, Ebene, Plan) und eval_sigmoid_vergleich_K.csv.
"""
import argparse
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "TN_ksweep"))
import run_tn as tn  # noqa: E402

SIG = re.compile(r"^(G|CW)_sig([0-9.]+)_(e[0-9.]+_(?:const|t[0-9.]+))$")
ECKE = 1e-3                 # alte Kennzahl: min alpha^2 < 1e-3 (zaehlt auch das Abschalten von N)
ECKE_ECHT = 0.99            # K1 (SG4): max alpha^2 > .99 in mindestens einer Klasse = Kollaps auf eine View
VIEWS = ["A", "B", "C", "N"]


def ident_name(level, plan):
    if plan == "e0.003_const":
        return f"{level}_const"
    if plan == "e0.003_t133":
        return f"{level}_decay"
    return f"{level}_{plan}"


class Snapshots:
    """hist/traj je Fit aus dem Cache-Ordner, sonst aus tar.gz-Archiven (ein Durchlauf je Archiv)."""

    def __init__(self, cache_dir, archives):
        self.cache, self.archives, self.mem = Path(cache_dir), [Path(a) for a in archives], {}

    def preload(self, keys):
        """keys: Menge von (art, variant, K, rep) mit art in {hist, traj}; laedt, was im Cache-Ordner fehlt."""
        need = {}
        for art, v, K, r in keys:
            stem = f"K={K:02d}_rep={r}"
            p = self.cache / art / v / (stem + (".npz" if art == "hist" else ".parquet"))
            if p.exists():
                self.mem[(art, v, K, r)] = p.read_bytes()
            else:
                need[f"{art}/{v}/{stem}" + (".npz" if art == "hist" else ".parquet")] = (art, v, K, r)
        for a in self.archives:
            if not need or not a.exists():
                continue
            with tarfile.open(a, "r:gz") as tf:
                for m in tf:
                    key = m.name.lstrip("./")
                    if key in need:
                        self.mem[need.pop(key)] = tf.extractfile(m).read()
        return sorted(need)                                   # fehlend

    def hist(self, v, K, r):
        b = self.mem.get(("hist", v, K, r))
        return None if b is None else dict(np.load(io.BytesIO(b)))

    def traj(self, v, K, r):
        b = self.mem.get(("traj", v, K, r))
        return None if b is None else pd.read_parquet(io.BytesIO(b))


def load_fits(run_dir, refs):
    f = pd.read_csv(Path(run_dir) / "fits.csv")
    for r in refs:                                            # Referenzen nur ergaenzen, Lauf hat Vorrang
        g = pd.read_csv(r)
        have = set(zip(f.variant, f.K, f.rep))
        f = pd.concat([f, g[[(v, K, rp) not in have for v, K, rp in zip(g.variant, g.K, g.rep)]]], ignore_index=True)
    return f


def per_fit(f, D, snaps):
    """Kennzahlen je Fit; fehlende Train-BA / Netto-Winkel / mu_sum-Minimum aus Snapshots."""
    rows = []
    for _, x in f.iterrows():
        a2 = np.array([[x.get(f"a2_end_c{c}_{v}", np.nan) for v in VIEWS] for c in (0, 1)])
        r = dict(variant=x.variant, level=x.level, K=int(x.K), rep=int(x.rep),
                 ecke=bool(np.nanmin(a2) < ECKE), ecke_echt=bool((np.nanmax(a2, axis=1) > ECKE_ECHT).any()), ba_end=x.ba_end, ba_best_val=x.ba_best_val,
                 bv_minus_end=x.ba_best_val - x.ba_end, val_end=x.val_end,
                 ba_train_end=x.get("ba_train_end", np.nan), ep_best_val=x.ep_best_val, ep_min_E=x.ep_best_train,
                 gcos0_last100=x.gcos0_last100, gcos1_last100=x.get("gcos1_last100", np.nan),
                 netto_c0=x.get("winkel_netto_c0", np.nan), netto_c1=x.get("winkel_netto_c1", np.nan),
                 cos_wtrue_c0_end=x.cos_wtrue_c0_end, cos_wtrue_c1_end=x.cos_wtrue_c1_end,
                 ep_min_mu_sum=np.nan, quelle_train="fits.csv")
        h = None
        if pd.isna(r["ba_train_end"]) or pd.isna(r["netto_c0"]):
            h = snaps.hist(x.variant, int(x.K), int(x.rep))
        if h is not None:
            W, V, pl = h["w_history"], h["v_history"], np.asarray(h["proto_lab"], int)
            if pd.isna(r["ba_train_end"]):
                r["ba_train_end"] = tn._ba(D["y_tr"], tn._predict(D["M_tr"], W[-1], pl, tn._a2_rows(V[-1], x.level)))
                r["quelle_train"] = "hist"
            if pd.isna(r["netto_c0"]):
                A = V[:, None, :].repeat(2, axis=1) if V.ndim == 2 else V
                r["netto_c0"], r["netto_c1"] = [float(np.arccos(np.clip(A[-1, c] @ A[0, c], -1, 1))) for c in (0, 1)]
        elif pd.isna(r["ba_train_end"]):
            r["quelle_train"] = "fehlt"
        r["luecke"] = r["ba_train_end"] - r["val_end"]
        if SIG.match(x.variant):
            t = snaps.traj(x.variant, int(x.K), int(x.rep))
            if t is not None and "mu_sum" in t:
                r["ep_min_mu_sum"] = int(t.epoch[t.mu_sum.idxmin()])
        else:
            r["ep_min_mu_sum"] = r["ep_min_E"]                # f = Identitaet: mu_sum = E
        rows.append(r)
    return pd.DataFrame(rows)


def paired(pf):
    """Sigmoid-Fits mit gepaarter Identitaet (gleiche Ebene, Plan, K, rep)."""
    pf = pf.set_index(["variant", "K", "rep"])
    out = []
    for (v, K, rep), s in pf.iterrows():
        m = SIG.match(v)
        if not m:
            continue
        lvl, xi, plan = m.group(1), float(m.group(2)), m.group(3)
        idn = ident_name(lvl, plan)
        if (idn, K, rep) not in pf.index:
            continue
        i = pf.loc[(idn, K, rep)]
        r = dict(level=lvl, xi=xi, plan=plan, K=K, rep=rep, ident=idn)
        for k in ("ecke", "ecke_echt", "bv_minus_end", "luecke", "val_end", "ba_end", "ba_best_val", "ep_best_val", "ep_min_E",
                  "ep_min_mu_sum", "gcos0_last100", "gcos1_last100", "netto_c0", "netto_c1",
                  "cos_wtrue_c0_end", "cos_wtrue_c1_end"):
            r[f"{k}_sig"], r[f"{k}_id"] = s[k], i[k]
        r["d_end"] = s.ba_end - i.ba_end
        r["d_best_val"] = s.ba_best_val - i.ba_best_val
        out.append(r)
    return pd.DataFrame(out)


AGG = dict(n=("rep", "size"), ecke_echt=("ecke_echt_sig", "mean"), ecke_echt_id=("ecke_echt_id", "mean"),
           ecke_anteil=("ecke_sig", "mean"), ecke_anteil_id=("ecke_id", "mean"),
           bv_minus_end=("bv_minus_end_sig", "mean"), bv_minus_end_id=("bv_minus_end_id", "mean"),
           d_end=("d_end", "mean"), luecke=("luecke_sig", "mean"), luecke_id=("luecke_id", "mean"),
           d_best_val=("d_best_val", "mean"), ep_best_val=("ep_best_val_sig", "mean"),
           ep_min_E=("ep_min_E_sig", "mean"), ep_min_mu_sum=("ep_min_mu_sum_sig", "mean"),
           gcos0_last100=("gcos0_last100_sig", "mean"), netto_c0=("netto_c0_sig", "mean"),
           netto_c0_id=("netto_c0_id", "mean"), cos_wtrue_c0_end=("cos_wtrue_c0_end_sig", "mean"),
           cos_wtrue_c1_end=("cos_wtrue_c1_end_sig", "mean"),
           val_end=("val_end_sig", "mean"), val_end_id=("val_end_id", "mean"))
# Vergleich zweier Laeufe (SG5): Kennzahlen je xi (Identitaet als eigene Zeile), Ebene, Plan – Seed a neben Seed b
VGL = {"ecke_echt": ("ecke_echt_sig", "ecke_echt_id"), "bv_minus_end": ("bv_minus_end_sig", "bv_minus_end_id"),
       "d_end": ("d_end", None), "d_best_val": ("d_best_val", None), "luecke": ("luecke_sig", "luecke_id"),
       "val_end": ("val_end_sig", "val_end_id")}


def vergleich_tab(pp, by):
    """Mittel je (xi, Ebene, Plan[, K]); xi 'Id' = Identitaet (je Paar-Schluessel einmal), d_end/d_best_val dort 0."""
    rows = []
    for key, g in pp.groupby(["xi", "level", "plan"] + by):
        rows.append(dict(zip(["xi", "level", "plan"] + by, key), **{m: g[s].astype(float).mean() for m, (s, _) in VGL.items()}))
    ident = pp.drop_duplicates(["level", "plan", "K", "rep"])
    for key, g in ident.groupby(["level", "plan"] + by):
        rows.append(dict(zip(["level", "plan"] + by, key), xi="Id",
                         **{m: (g[i].astype(float).mean() if i else 0.0) for m, (_, i) in VGL.items()}))
    t = pd.DataFrame(rows)
    t["xi"] = t.xi.map(lambda x: x if x == "Id" else f"{float(x):g}")
    return t


def vergleich(pp_a, pp_b, seed_a, seed_b):
    """Seed a (dieser Lauf) neben Seed b (--vergleich): je Kennzahl zwei Spalten S<a>, S<b>."""
    out = {}
    for name, by in (("plan", []), ("K", ["K"])):
        ta, tb = vergleich_tab(pp_a, by), vergleich_tab(pp_b, by)
        keys = ["xi", "level", "plan"] + by
        t = tb.merge(ta, on=keys, how="outer", suffixes=(f"_S{seed_b}", f"_S{seed_a}"))
        order = {"Id": 0, "5": 1, "10": 2, "20": 3}
        t = t.sort_values(keys, key=lambda c: c.map(order) if c.name == "xi" else c).reset_index(drop=True)
        cols = keys + [f"{m}_S{s}" for m in VGL for s in (seed_b, seed_a)]
        out[name] = t[cols]
    return out
# Kriterien-Datei criteria_sg.yaml (Auftrag SG3): Kriterium K<n>_... -> Kennzahl, Plan-Namen -> Plan-Kennung
KRIT_METRIC = {"K1": "ecke_echt_sig", "K2": "bv_minus_end_sig", "K3": "d_end", "K4": "luecke_sig", "K5": "d_best_val"}
PLAN_ID = {"const_0.003": "e0.003_const", "const_0.01": "e0.01_const", "decay_0.003_t133": "e0.003_t133"}


def check_criteria(pp, crit_path):
    """criteria_sg.yaml (status, xi_bestaetigung, kriterien: {K<n>_...: {kennzahl, plaene, schwelle}}).
    Je xi und Ebene: Mittel ueber K und reps je Plan, alle genannten Plaene muessen die Schwelle erfuellen.
    status vorlaeufig -> nur Pruefung je xi mit Vermerk, keine Entscheidung. Sonst Entscheidung fuer das gewaehlte xi
    (xi_bestaetigung, oder das einzige xi im Lauf): bestaetigt = alle Kriterien auf G und CW, teilweise = auf einer Ebene."""
    spec = yaml.safe_load(Path(crit_path).read_text(encoding="utf-8"))
    krit = {}
    for name, c in spec["kriterien"].items():
        m = re.fullmatch(r"\s*(<=|>=)\s*([+-]?[0-9.]+)\s*", str(c.get("schwelle", "")))
        key = name.split("_")[0]
        if not m or key not in KRIT_METRIC or any(p not in PLAN_ID for p in c.get("plaene", [])):
            raise SystemExit(f"Kriterium {name} nicht lesbar (schwelle/plaene): {c}")
        krit[name] = dict(metric=KRIT_METRIC[key], op=m.group(1), value=float(m.group(2)),
                          plans=[PLAN_ID[p] for p in c["plaene"]])
    rows = []
    for (xi, lvl), g in pp.groupby(["xi", "level"]):
        r = dict(xi=xi, level=lvl)
        for name, c in krit.items():
            vals = [g[g.plan == p][c["metric"]].astype(float).mean() for p in c["plans"]]
            ok = not any(np.isnan(vals)) and all((v <= c["value"]) if c["op"] == "<=" else (v >= c["value"]) for v in vals)
            r[name] = "erfüllt" if ok else "nicht erfüllt"
            r[f"{name}_werte"] = ", ".join(f"{p}: {v:.3f}" for p, v in zip(c["plans"], vals))
        r["alle"] = all(r[n] == "erfüllt" for n in krit)
        rows.append(r)
    res = pd.DataFrame(rows)
    status = str(spec.get("status", "")).strip()
    if status not in ("festgelegt", "endgueltig"):
        return res, f"Kriterien-Status „{status}“: vorläufig – Exploration, keine Entscheidung."
    xis = sorted(res.xi.unique())
    xi_sel = spec.get("xi_bestaetigung")
    if len(xis) != 1:                                    # Lauf 1 (mehrere xi): Exploration, nie eine Entscheidung
        return res, f"Mehrere ξ im Lauf ({', '.join(f'{x:g}' for x in xis)}): Exploration, keine Entscheidung."
    if xi_sel is None or float(xi_sel) != xis[0]:
        return res, "Keine Entscheidung: xi_bestaetigung fehlt oder passt nicht zum ξ des Laufs."
    xi_sel = float(xi_sel)
    n_ok = int(res[res.xi == xi_sel].alle.sum())
    verdict = "bestätigt" if n_ok == 2 else ("teilweise" if n_ok == 1 else "verworfen")
    return res, f"H-S {verdict} (ξ = {xi_sel:g}; Kriterien erfüllt auf {n_ok} von 2 Ebenen)."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="results/public/<lauf-id>")
    ap.add_argument("--ref", nargs="*", default=[], help="weitere fits.csv mit Identitaets-Referenzen (Seed 3)")
    ap.add_argument("--archive", nargs="*", default=[], help="tar.gz mit hist/ und traj/ (Hub-Snapshots)")
    ap.add_argument("--criteria", default=str(ROOT / "experiments" / "TN_ksweep" / "criteria_sg.yaml"),
                    help="Kriterien-Datei (Standard criteria_sg.yaml); --ohne-kriterien schaltet die Pruefung ab")
    ap.add_argument("--ohne-kriterien", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vergleich", default=None,
                    help="eval_sigmoid_fits.csv eines anderen Laufs (z. B. Seed 3, Lauf 1): Tabelle nebeneinander (SG5)")
    ap.add_argument("--seed", type=int, default=None, help="Datensatz-Seed dieses Laufs (Standard: data_seed der Config)")
    ap.add_argument("--vergleich-seed", type=int, default=3)
    args = ap.parse_args()

    cfg = tn.load_config(args.config)
    sig = tn.cache_signature(cfg)
    D = tn.build_data(cfg)
    f = load_fits(args.run, args.ref)
    f = f[f.K.isin(cfg["K"])]
    snaps = Snapshots(ROOT / "cache" / "toy_H0" / sig, args.archive)
    keys = set()
    for _, x in f.iterrows():
        if SIG.match(x.variant):
            keys.add(("traj", x.variant, int(x.K), int(x.rep)))
        if pd.isna(x.get("ba_train_end", np.nan)) or pd.isna(x.get("winkel_netto_c0", np.nan)):
            keys.add(("hist", x.variant, int(x.K), int(x.rep)))
    missing = snaps.preload(keys)
    pf = per_fit(f, D, snaps)
    pp = paired(pf)
    if pp.empty:
        raise SystemExit("keine gepaarten Sigmoid-/Identitaets-Fits gefunden")
    agg = pp.groupby(["xi", "level", "plan", "K"]).agg(**AGG).reset_index()

    out = Path(args.out or args.run)
    out.mkdir(parents=True, exist_ok=True)
    pp.to_csv(out / "eval_sigmoid_fits.csv", index=False, float_format="%.5f")
    agg.to_csv(out / "eval_sigmoid_agg.csv", index=False, float_format="%.4f")
    L = [f"# Auswertung Sigmoid · {Path(args.run).name}\n",
         f"Config `{Path(args.config).as_posix()}`, Cache-Signatur `{sig}`, gepaarte Fits: {len(pp)}, "
         f"Train-BA aus hist nachgerechnet: {int((pf.quelle_train == 'hist').sum())}, fehlend: {int((pf.quelle_train == 'fehlt').sum())}, "
         f"fehlende Snapshot-Dateien: {len(missing)}.\n",
         "Kennzahlen gepaart je (K, rep) gegen Identität gleicher Ebene und gleichen Plans. Ecke (K1) = max α² > .99 in mindestens einer Klasse am Ende; "
         "ecke_anteil = alte Kennzahl min α² < 1e-3 (über Klassen und Views). Lücke = Train-BA − Val-BA am Ende. d_end = BA_End(Sig) − BA_End(Id), d_best_val analog.\n"]
    if not args.ohne_kriterien and Path(args.criteria).exists():
        res, urteil = check_criteria(pp, args.criteria)
        res.to_csv(out / "eval_sigmoid_kriterien.csv", index=False)
        L += [f"## Kriterien (`{Path(args.criteria).name}`)\n", f"**{urteil}**\n", tn._md(res)]
    else:
        L.append("**Ohne Kriterien: nur Kennzahlen (Exploration), keine Akzeptanz, kein Finding.**\n")
    L += ["## Mittel je ξ, Ebene, Plan (über K und reps)\n",
          tn._md(pp.groupby(["xi", "level", "plan"]).agg(**AGG).reset_index()),
          "\n## Je ξ, Ebene, Plan, K\n", tn._md(agg)]
    if args.vergleich:
        sa, sb = args.seed if args.seed is not None else int(cfg["data_seed"]), args.vergleich_seed
        pb = pd.read_csv(args.vergleich)
        if "val_end_sig" not in pb:
            raise SystemExit(f"{args.vergleich} ohne val_end-Spalten: Lauf erst mit dieser Skriptversion neu auswerten")
        vg = vergleich(pp, pb, sa, sb)
        vg["plan"].to_csv(out / "eval_sigmoid_vergleich.csv", index=False, float_format="%.6f")
        vg["K"].to_csv(out / "eval_sigmoid_vergleich_K.csv", index=False, float_format="%.6f")
        kk = vg["K"][["xi", "level", "plan", "K", f"d_end_S{sb}", f"d_end_S{sa}", f"luecke_S{sb}", f"luecke_S{sa}"]]
        L += [f"\n## Vergleich Seed {sb} (`{Path(args.vergleich).parent.name}`) neben Seed {sa} (dieser Lauf)\n",
              "Mittel je ξ, Ebene, Plan über K und reps, gepaart gegen Identität gleicher Plan; Zeile ξ = Id ist die "
              "Identität selbst (End − Identität und Best_val − Identität dort 0). Kennzahlen: Anteil echte Ecke, "
              "Best_val − End, End − Identität, Best_val − Identität, Lücke Train − Val (End), Val-BA am Ende. "
              "Kriterien oben nur als Kennzahlen-Ampel zur Orientierung, **ohne Urteil**.\n",
              tn._md(vg["plan"]), "\n### Je K: End − Identität und Lücke\n", tn._md(kk)]
    if missing:
        L.append("\nFehlende Snapshot-Dateien (Spalten dann leer): " + ", ".join(missing[:20]) + (" …" if len(missing) > 20 else ""))
    (out / "eval_sigmoid.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"geschrieben: {out / 'eval_sigmoid.md'} ({len(pp)} Paare, {len(missing)} Snapshots fehlen)")


if __name__ == "__main__":
    main()
