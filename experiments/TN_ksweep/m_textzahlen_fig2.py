"""Paper M: Textzahlen zu Fig. 2 (Pfad Datensatz 3, K 6, rep 0) und Kosten/BA über alle SG9-Identitäts-Läufe.

Aufruf (Repo-Root): python experiments/TN_ksweep/m_textzahlen_fig2.py [--out results/public/2026-10-29_M_textzahlen_fig2]
Nur Auswertung vorhandener Logs/Snapshots (kein Fit). Lokal: Snapshots aus cache/hub/* (FALLBACK in src/paper_figs/data.py).

Ausgabe (<out>/): zahlen.csv (punkt, ebene, lauf, groesse, wert, kriterium, quelle), sg9_kosten_ba_fits.csv
(je Fit Punkt 4), summary.md wird von Hand aus zahlen.csv geschrieben.
Punkte (Auftrag 29.10.2026):
1 Winkel zum Orakel (Start, Minimum, Best_val, Ende), α² Heimat-View am Ende, Halt-R der Sigmoide.
2 Größte Winkelabweichung zwischen den Lernraten-Pfaden über R (alle 16 Identitäts-Pläne von TN + LR_explore).
3 Identität: Σμ, Train-/Val-BA bei Best_val und Ende, Grenzwechsel, Kostenbeitrag nach Lage.
4 SG9-Identitäts-Läufe (Seeds 3–17 × K {2, 6, 12} × 5 reps): Anteil Σμ(Ende) < Σμ(Best_val) und Train-BA(Ende) <
  Train-BA(Best_val).
5 Orakel-Vektoren α² (W_TRUE) und Ecke am Ende (Identität, Datensatz 3, K 6, rep 0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "TN_ksweep"))
import run_tn as tn  # noqa: E402
import eval_sg9 as e9  # noqa: E402
from src.paper_figs import data as dt, figures as fg  # noqa: E402

OUT = ROOT / "results" / "public" / "2026-10-29_M_textzahlen_fig2"
RUN = dt.RUN_S3
K, REP = 6, 0
ID, SIG = "e0.003_const", "sig10_e0.003_const"
# 14 Pläne aus config_lr_explore.yaml + e0.003_const und e0.003_t133 aus TN (gleiche Signatur 5dfd…)
LR_PLANS = [f"e{e}_{t}" for e in ("0.001", "0.003", "0.01", "0.03") for t in ("const", "t50", "t133", "t400")]
HALT_TOL = (0.1, 0.5)                 # Grad, Winkel zum Orakel gegen den Endwert
ROT_TOL = (0.5, 1.0)                  # Grad, Drehung gegen den Endvektor (Schrittwinkel je Epoche ≈ .17°)
SRC_TRAJ = "cache/hub/5dfd/traj bzw. cache/hub/sg/traj (Signatur 5dfd9972f654650e), Spalte a2_c*_*, eta"
SRC_HIST = "cache/hub/5dfd/hist bzw. cache/hub/sg/hist (Snapshots w_history, v_history), Kontrolle gegen Log ≤ 1e-10"

rows = []


def add(punkt, ebene, lauf, groesse, wert, kriterium="", quelle=SRC_TRAJ):
    rows.append(dict(punkt=punkt, ebene=ebene, lauf=lauf, groesse=groesse, wert=wert, kriterium=kriterium,
                     quelle=quelle))


def a_unit(t, c):
    a = np.sqrt(t[[f"a2_c{c}_{x}" for x in dt.VIEWS]].to_numpy(float))
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def halt_index(x, tol):
    """Erste Epoche t*, ab der x[t] ≤ tol für alle t ≥ t* gilt (x[-1] = 0 vorausgesetzt)."""
    bad = np.where(x > tol)[0]
    return 0 if len(bad) == 0 else int(bad[-1]) + 1


def punkt1():
    for lvl in ("G", "CW"):
        for p in (ID, SIG):
            v = dt.verlauf(RUN, lvl, p, K, REP, 0)
            t = dt.traj(RUN, dt.variant_name(lvl, p), K, REP)
            lauf = f"DS 3, {v.attrs['variant']}, K {K}, rep {REP}"
            w, R = v.winkel.to_numpy(), v.R.to_numpy()
            eb, im, n = v.attrs["ep_best_val"], int(w.argmin()), len(w) - 1
            ref = "G: Orakel global (W_TRUE['glob']); CW: Orakel Klasse 0 (W_TRUE[0]); Winkel über alle 4 Views " \
                  "zwischen a = sqrt(α²) (normiert) und sqrt(W_TRUE)"
            add(1, lvl, lauf, "Winkel zum Orakel am Start (Epoche 0), Grad", f"{w[0]:.1f}", ref)
            add(1, lvl, lauf, "Winkel-Minimum, Grad (Epoche, R)", f"{w[im]:.1f} ({im}, R {R[im]:.3f})",
                ref + "; Minimum über alle 401 Epochen")
            add(1, lvl, lauf, "Winkel bei Best_val, Grad (Epoche, R)", f"{w[eb]:.1f} ({eb}, R {R[eb]:.3f})",
                ref + "; Best_val = erste Epoche mit maximaler Val-BA (argmax ba_val)")
            add(1, lvl, lauf, f"Winkel am Ende, Grad (Epoche {n}, R {R[n]:.3f})", f"{w[n]:.1f}", ref)
            for c, home in ((0, "A"), (1, "B")):
                cc = 0 if lvl == "G" else c
                add(1, lvl, lauf, f"α²_{home} am Ende (Heimat-View Klasse {c})", f"{t[f'a2_c{cc}_{home}'].iloc[-1]:.3f}",
                    "G: ein Vektor für beide Klassen" if lvl == "G" else f"CW: Gewichtsvektor Klasse {c}")
            # Halt: (i) Drehung des eigenen Vektors gegen das Ende, (ii) Winkel zum Orakel gegen den Endwert
            for c in ((0,) if lvl == "G" else (0, 1)):
                a = a_unit(t, c)
                rot = np.degrees(np.arccos(np.clip(a @ a[n], -1, 1)))
                step = np.degrees(np.arccos(np.clip((a[1:] * a[:-1]).sum(1), -1, 1)))
                for tol in ROT_TOL:
                    i = halt_index(rot, tol)
                    add(1, lvl, lauf, f"Halt-R: Drehung ∠(a(t), a(T)) ≤ {tol}° für alle t ≥ t* (Klasse {c})",
                        f"R {R[i]:.3f} (Epoche {i})",
                        "a = normierter Gewichtsfaktor über 4 Views; t* = erste Epoche, ab der die Drehung zum "
                        "Endvektor dauerhaft unter der Toleranz bleibt (Toleranz gesetzt, nicht vorab festgelegt; "
                        "≥ Schrittwinkel je Epoche, sonst zählt die Oszillation um den Haltepunkt als Bewegung)")
                add(1, lvl, lauf, f"Schrittwinkel je Epoche ∠(a(t−1), a(t)), Mittel letzte 100 Epochen (Klasse {c})",
                    f"{step[-100:].mean():.3f}°", "normierter Schritt: Länge je Epoche fest, Richtung frei")
                add(1, lvl, lauf, f"Netto-Drehung ∠(a(T−50), a(T)) und ∠(a(T−100), a(T)) (Klasse {c})",
                    f"{rot[n - 50]:.2f}° / {rot[n - 100]:.2f}°", "wie pfad2_rows (dreh_win 50)")
            for tol in HALT_TOL:
                i = halt_index(np.abs(w - w[n]), tol)
                add(1, lvl, lauf, f"Halt-R: |Winkel(t) − Winkel(T)| ≤ {tol}° für alle t ≥ t*",
                    f"R {R[i]:.3f} (Epoche {i})", ref + "; Toleranz gesetzt, nicht vorab festgelegt")


def punkt2():
    for lvl in ("G", "CW"):
        for name, plans in (("16 Identitäts-Pläne (LR_explore 14 + TN 2)", LR_PLANS), ("5 Pläne PLANS (F56)", fg.PLANS)):
            vs = [dt.verlauf(RUN, lvl, p, K, REP, 0) for p in plans]
            rend = np.array([float(v.R.iloc[-1]) for v in vs])
            grid = np.round(np.arange(0, rend.max() + 1e-9, 0.001), 6)
            Y = np.stack([np.where(grid <= v.R.iloc[-1] + 1e-12, np.interp(grid, v.R, v.winkel), np.nan) for v in vs])
            cnt = np.isfinite(Y).sum(0)
            w = np.where(cnt >= 2, np.nanmax(Y, 0) - np.nanmin(Y, 0), np.nan)
            lauf = f"DS 3, {lvl}, K {K}, rep {REP}, {name}"
            krit = ("Winkel zum Orakel je Plan linear über R interpoliert (Raster 0.001), jeder Plan nur bis zu seinem R "
                    "am Ende; Breite = max − min über die Pläne, nur wo ≥ 2 Pläne vorliegen")
            for rmax in (1.25, None):
                m = np.isfinite(w) & ((grid <= rmax) if rmax else True)
                i = int(np.nanargmax(np.where(m, w, -1)))
                add(2, lvl, lauf, f"größte Winkelabweichung zwischen den Pfaden, R ≤ {rmax if rmax else 'max'} (Grad)",
                    f"{w[i]:.2f} bei R {grid[i]:.3f} ({int(cnt[i])} Pläne)", krit)
            add(2, lvl, lauf, "Spanne R am Ende über die Pläne", f"{rend.min():.3f} – {rend.max():.3f}", "Σ η_t bis T = 400")


def punkt3():
    for lvl in ("G", "CW"):
        s = dt.verlauf_sets(RUN, lvl, ID, K, REP, ("train", "val"))
        mu, eb, n = s["mu_train"], s["eb"], s["n"]
        lauf = f"DS 3, {s['variant']}, K {K}, rep {REP}"
        q = SRC_HIST + f" (max. Abw. {', '.join(f'{k} {x:.0e}' for k, x in s['check'].items())})"
        add(3, lvl, lauf, f"Best_val: Epoche, R", f"{eb}, {s['R'][eb]:.3f}", "erste Epoche mit max. Val-BA", q)
        for nm, e in (("Best_val", eb), ("Ende", n)):
            add(3, lvl, lauf, f"Σμ (Train, 600 Punkte) bei {nm}", f"{s['E_mu'][e]:.1f}",
                "μ_i = (d+ − d−)/(d+ + d− + 1e-5), aus Snapshots; = loss im Log", q)
            add(3, lvl, lauf, f"Train-BA / Val-BA bei {nm}", f"{s['ba_train'][e]:.3f} / {s['ba_val'][e]:.3f}",
                "nächster Prototyp in der Prototyp-Metrik", q)
        add(3, lvl, lauf, "ΔΣμ Ende − Best_val", f"{s['E_mu'][n] - s['E_mu'][eb]:+.1f}", "", q)
        c0, c1 = mu[eb] < 0, mu[n] < 0
        add(3, lvl, lauf, "Trainingspunkte richtig → falsch (Best_val → Ende)", str(int((c0 & ~c1).sum())),
            "richtig = μ < 0; Vergleich nur der beiden Epochen", q)
        add(3, lvl, lauf, "Trainingspunkte falsch → richtig (Best_val → Ende)", str(int((~c0 & c1).sum())),
            "richtig = μ < 0", q)
        add(3, lvl, lauf, "Trainingspunkte richtig und „richtiger“ (μ_Ende < μ_Best_val)",
            str(int((c0 & c1 & (mu[n] < mu[eb])).sum())), "an beiden Epochen μ < 0 und μ gesunken", q)
        kb = dt.kostenbeitrag(mu, dt.squash(ID)[0], 0.1)
        lab = dict(sicher="sicher richtig (μ < −.1)", grenze="Grenzbereich (|μ| ≤ .1)", falsch="falsch (μ > .1)",
                   summe="Summe")
        for k in ("sicher", "grenze", "falsch", "summe"):
            add(3, lvl, lauf, f"Kostenbeitrag ΔΣμ vor / nach Best_val: {lab[k]}",
                f"{kb[k][eb] - kb[k][0]:+.1f} / {kb[k][n] - kb[k][eb]:+.1f}",
                "je Epoche Δμ_i = μ_i(t) − μ_i(t−1), Lage nach μ_i(t−1), summiert über Epochen 1…Best_val bzw. "
                "Best_val+1…T (data.kostenbeitrag, margin .1; Code: |μ| ≤ .1)", q)
        nb = int((np.abs(mu[:-1]) == 0.1).sum())
        add(3, lvl, lauf, "Anzahl (Epoche, Punkt) mit |μ| genau .1", str(nb), "Unterschied < .1 / ≤ .1", q)


def _fit_values(run, lvl, Kk, rep, D):
    """eb, Σμ und Train-BA bei Best_val und Ende; Quelle traj (+ hist für Train-BA, falls nicht geloggt) oder nur hist."""
    v = dt.variant_name(lvl, ID)
    try:
        t = dt.traj(run, v, Kk, rep)
    except FileNotFoundError:
        t = None
    h = dt.hist(run, v, Kk, rep)
    W, V, pl = h["w_history"], h["v_history"], np.asarray(h["proto_lab"], int)

    def at(e, M, y):
        a2 = tn._a2_rows(V[e], lvl)
        dp, dm = tn._assign(M, y, W[e], pl, a2)[:2]
        return (dp - dm) / (dp + dm + 1e-5), tn._ba(y, tn._predict(M, W[e], pl, a2))

    if t is not None:
        vb, loss, n = t.ba_val.to_numpy(), t.loss.to_numpy(), len(t) - 1
        eb = int(vb.argmax())
        if "ba_train" in t:
            tr = t.ba_train.to_numpy()
            src = "traj (loss, ba_val, ba_train)"
            trb, tre = tr[eb], tr[n]
        else:
            trb, tre = at(eb, D["M_tr"], D["y_tr"])[1], at(n, D["M_tr"], D["y_tr"])[1]
            src = "traj (loss, ba_val) + hist (Train-BA)"
        eb_, n_ = loss[eb], loss[n]
        chk = max(abs(at(e, D["M_tr"], D["y_tr"])[0].sum() - loss[e]) for e in (eb, n))
    else:
        vb = np.array([at(e, D["M_val"], D["y_val"])[1] for e in range(len(W))])
        n = len(W) - 1
        eb = int(vb.argmax())
        (mb, trb), (me, tre) = at(eb, D["M_tr"], D["y_tr"]), at(n, D["M_tr"], D["y_tr"])
        eb_, n_ = mb.sum(), me.sum()
        chk = np.nan
        src = "nur hist (Val-BA, Σμ, Train-BA aus Snapshots; kein Epochen-Log lokal)"
    return dict(ep_best_val=eb, E_bv=float(eb_), E_end=float(n_), tr_bv=float(trb), tr_end=float(tre),
                val_bv=float(vb[eb]), quelle=src, check_E=chk)


def punkt4(out):
    fr = []
    for seed, spec in sorted(e9.SEEDS.items()):
        run = dt.Run(seed, str(Path(spec["config"]).relative_to(ROOT)).replace("\\", "/"))
        D = dt.full_data(run)
        fits = e9.SEEDS[seed]["run"] / "fits.csv"
        ref = pd.concat([pd.read_csv(p) for p in [fits, *spec["refs"]] if Path(p).exists()])
        for lvl in ("G", "CW"):
            for Kk in (2, 6, 12):
                for rep in range(5):
                    r = _fit_values(run, lvl, Kk, rep, D)
                    f = ref[(ref.variant == dt.variant_name(lvl, ID)) & (ref.K == Kk) & (ref.rep == rep)]
                    r.update(seed=seed, sig=run.sig, level=lvl, K=Kk, rep=rep,
                             fits_ep_best_val=int(f.ep_best_val.iloc[0]) if len(f) else np.nan,
                             fits_val_best_val=float(f.val_best_val.iloc[0]) if len(f) else np.nan,
                             fits_tr_end=float(f.ba_train_end.iloc[0]) if len(f) and "ba_train_end" in f else np.nan,
                             fits_tr_bv=float(f.ba_train_best_val.iloc[0]) if len(f) and "ba_train_best_val" in f
                             else np.nan)
                    fr.append(r)
        print("Seed", seed, "fertig", flush=True)
    F = pd.DataFrame(fr)
    F["kosten_sinkt"] = F.E_end < F.E_bv
    F["trainba_sinkt"] = F.tr_end < F.tr_bv
    F["beides"] = F.kosten_sinkt & F.trainba_sinkt
    F.to_csv(out / "sg9_kosten_ba_fits.csv", index=False)
    krit = ("Best_val = erste Epoche mit max. Val-BA; strikt < ; Identität const .003, T 400; Σμ = loss im Log "
            "(Seed 4: aus Snapshots)")
    q = "sg9_kosten_ba_fits.csv; Quellen je Fit Spalte quelle; Seeds/Configs wie eval_sg9.SEEDS"
    for lvl in ("G", "CW"):
        g = F[F.level == lvl]
        N = len(g)
        for col, nm in (("beides", "Σμ(Ende) < Σμ(Best_val) UND Train-BA(Ende) < Train-BA(Best_val)"),
                        ("kosten_sinkt", "Σμ(Ende) < Σμ(Best_val)"), ("trainba_sinkt", "Train-BA(Ende) < Train-BA(Best_val)")):
            add(4, lvl, f"SG9 Identität, 15 DS × K {{2, 6, 12}} × 5 reps", f"Anteil {nm}",
                f"{int(g[col].sum())}/{N} ({g[col].mean():.1%})", krit, q)
        add(4, lvl, "SG9 Identität", "Fits mit Best_val = Ende (Epoche 400)", str(int((g.ep_best_val == 400).sum())),
            krit, q)
        d = g[np.isfinite(g.fits_ep_best_val)]
        add(4, lvl, "SG9 Identität", "Kontrolle: Best_val-Epoche = fits.csv ep_best_val",
            f"{int((d.ep_best_val == d.fits_ep_best_val).sum())}/{len(d)}", "", q)
        d = g[np.isfinite(g.fits_tr_end)]
        add(4, lvl, "SG9 Identität", "Kontrolle: max |Train-BA Ende − fits.csv ba_train_end|",
            f"{np.max(np.abs(d.tr_end - d.fits_tr_end)):.1e} (n {len(d)})", "", q)
        d = g[np.isfinite(g.check_E)]
        add(4, lvl, "SG9 Identität", "Kontrolle: max |Σμ(Snapshot) − loss(Log)| bei Best_val/Ende",
            f"{np.max(d.check_E):.1e} (n {len(d)})", "", q)
    return F


def punkt5():
    W = dt.train_data(RUN)["W_TRUE"]
    krit = ("run_tn.build_data: q[r, v] = mittlere quadrierte Distanz innerhalb des wahren Clusters r (Paare i ≠ j, "
            "Train, Distanz je View / Train-Maximum; toy_heimatview.within_q); W = (1/q̄) / Σ_v (1/q̄), "
            "q̄ = Mittel über die 6 Cluster der Klasse (CW) bzw. alle 12 (G); α²-Raum")
    for key, nm in (("glob", "G (global)"), (0, "CW Klasse 0"), (1, "CW Klasse 1")):
        add(5, nm.split()[0], "DS 3 (Train, 600 Punkte)", f"Orakel α² {nm} (A, B, C, N)",
            ", ".join(f"{x:.3f}" for x in np.asarray(W[key], float)), krit, "run_tn.build_data(config_sig_explore.yaml)")
    for lvl in ("G", "CW"):
        t = dt.traj(RUN, dt.variant_name(lvl, ID), K, REP)
        for c in ((0,) if lvl == "G" else (0, 1)):
            a2 = t[[f"a2_c{c}_{x}" for x in dt.VIEWS]].iloc[-1].to_numpy(float)
            j = int(a2.argmax())
            add(5, lvl, f"DS 3, {dt.variant_name(lvl, ID)}, K {K}, rep {REP}",
                f"Ecke am Ende{'' if lvl == 'G' else f' Klasse {c}'}: View (max α²); α² (A, B, C, N)",
                f"{dt.VIEWS[j]} ({a2[j]:.3f}); " + ", ".join(f"{x:.3f}" for x in a2),
                "Ecke = max α² > .99 (Definition E5/K1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ohne4", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    punkt1(); punkt2(); punkt3(); punkt5()
    if not a.ohne4:
        punkt4(out)
    Z = pd.DataFrame(rows).sort_values("punkt", kind="stable")
    Z.to_csv(out / "zahlen.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 70, "display.width", 250):
        print(Z[["punkt", "ebene", "lauf", "groesse", "wert"]].to_string(index=False))


if __name__ == "__main__":
    main()
