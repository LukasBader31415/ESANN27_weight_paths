"""Bildgruppen der Paper-Abbildungen (Auftrag ABB). Jede fig_* gibt eine Figure in Endgröße zurück.

Gemeinsame Parameter: width_frac (Anteil der Textbreite), height_cm, lang ('en' | 'de'), legend ('top' | 'inside' |
'none'), panels (Auswahl/Reihenfolge der Felder), titles (False: nur Buchstaben a, b, …; True: Buchstabe + Titel;
None: nichts), runs (Liste von data.Run; derzeit genau ein Lauf).
Einzelfelder panel_*(ax, …) zeichnen in eine vorhandene Achse und geben (handles, labels) zurück; fig_kombi setzt
sie frei zusammen. Vorlagen: docs/folien/2026-10-21_betreuung/entwuerfe_chat/code/ (fa_zerlegung_k.py,
f7_lernrate.py, f7b_ueberschiessen.py, fb_trainloss.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import data as dt
from .layout import COL, ETA_COL, DECAY_COL, figsize_cm

# Lernraten-Pläne der Vorlage (Seed 3, LR_explore + TN): vier konstante η und ein abnehmender Plan
PLANS = ("e0.001_const", "e0.003_const", "e0.01_const", "e0.03_const", "e0.03_t50")

TXT = {
    "en": dict(epoch="Epoch", R=r"Path length $R=\Sigma_t\,\eta_t$", a2=r"$\alpha^2_{{{v}}}$ ({lvl}, class {c})",
               angle=r"Angle to oracle ($^\circ$)", cost=r"Training cost $E$", val="Validation BA",
               train="Training BA", ba="BA", test="Test BA", K=r"$K$ (prototypes per class)",
               start="start", proto="+ prototypes", gain_id="+ weights (identity)",
               harm="weights harm", id_end="identity, end", id_bv="identity, best val.",
               sig=r"sigmoid $\nu=10$", oracle="oracle weights", frozen="frozen",
               gain=r"BA $-$ BA(frozen)", best="best validation epoch", corner="corner",
               band="range of best validation epochs", after="after best validation epoch",
               level=dict(G="G (global weights)", CW="CW (class-wise weights)"),
               box=("Between best validation epoch and end ({n} training points):", "{r} correct points get more correct",
                    "{rf} flip correct $\\to$ wrong, {fr} wrong $\\to$ correct", "training BA {a:.3f} $\\to$ {b:.3f}"),
               t_alpha="weight of the home view", t_angle="distance to oracle weights", t_cost="training cost",
               t_val="validation BA", t_trainval="training and validation BA", t_zer="decomposition over $K$"),
    "de": dict(epoch="Epoche", R=r"Weg $R=\Sigma_t\,\eta_t$", a2=r"$\alpha^2_{{{v}}}$ ({lvl}, Klasse {c})",
               angle=r"Winkel zum Orakel ($^\circ$)", cost=r"Train-Kosten $E$", val="Val-BA",
               train="Train-BA", ba="BA", test="Test-BA", K=r"$K$ (Prototypen je Klasse)",
               start="Start", proto="+ Prototypen", gain_id="+ Gewichte (Identität)",
               harm="Gewichte schaden", id_end="Identität Ende", id_bv="Identität Best_val",
               sig=r"Sigmoid $\nu=10$", oracle="Orakel-Gewichte", frozen="frozen",
               gain=r"BA $-$ BA(frozen)", best="Best_val", corner="Ecke", band="Spanne der Best_val-Stellen",
               after="nach Best_val", level=dict(G="G (globale Gewichte)", CW="CW (klassenweise Gewichte)"),
               box=("Zwischen Best_val und Ende ({n} Trainingspunkte):", "{r} richtige Punkte werden richtiger",
                    "{rf} kippen richtig $\\to$ falsch, {fr} falsch $\\to$ richtig", "Train-BA {a:.3f} $\\to$ {b:.3f}"),
               t_alpha="Gewicht der Heimat-View", t_angle="Abstand zu den Orakel-Gewichten", t_cost="Train-Kosten",
               t_val="Val-BA", t_trainval="Train- und Val-BA", t_zer="Zerlegung über $K$"),
}


def _t(lang):
    return TXT[lang]


def plan_label(plan, lang="en"):
    """'e0.01_const' -> η = .01; 'e0.03_t50' -> η0 = .03, τ = 50 (decaying)."""
    if plan.startswith("sig"):                              # ABB2: 'sig10_e0.003_const'
        sig, rest_plan = plan.split("_", 1)
        return rf"sigmoid $\nu$ = {sig[3:]}, " + plan_label(rest_plan, lang)
    e, rest = plan.split("_", 1)
    eta = e[1:].lstrip("0") or "0"
    if rest == "const":
        return rf"$\eta$ = {eta}"
    return rf"$\eta_0$ = {eta}, $\tau$ = {rest[1:]}"


def plan_style(level, plans):
    """Farbe/Linienstil je Plan: konstante η als Abstufung der Ebenenfarbe, abnehmende Pläne gestrichelt in Tinte."""
    const = [p for p in plans if p.endswith("_const")]
    cols = ETA_COL[level]
    idx = np.linspace(0, len(cols) - 1, max(len(const), 1)).round().astype(int) if len(const) <= len(cols) else None
    out = {}
    for i, p in enumerate(const):
        out[p] = dict(color=cols[idx[i]] if idx is not None else plt.cm.Oranges(0.3 + 0.6 * i / len(const)), ls="-")
    for p in plans:
        if p not in out:
            out[p] = dict(color=DECAY_COL, ls=(0, (4, 1.5, 1, 1.5)))
    return out


def _letter(ax, i, title=None, titles=False):
    if titles is None:
        return
    s = f"({chr(97 + i)})" + (f" {title}" if titles and title else "")
    ax.set_title(s, loc="left")


def _look(h):
    """Aussehen eines Legenden-Handles (Farbe, Linie, Marker, Schraffur) zum Vergleich gleich benannter Einträge."""
    if isinstance(h, Line2D):
        return ("line", to_hex(h.get_color(), keep_alpha=True), str(h.get_linestyle()), h.get_marker(), h.get_alpha())
    if isinstance(h, Patch):
        return ("patch", to_hex(h.get_facecolor(), keep_alpha=True), to_hex(h.get_edgecolor(), keep_alpha=True),
                h.get_hatch(), h.get_alpha())
    return ("other", id(h))


def _legend(fig, axes, handles, labels, legend, ncol=None):
    """Gemeinsame Legende. Gleiche Labels werden zusammengefasst; sieht ein Eintrag je Ebene anders aus (z. B. G blau,
    CW orange), wird er als zweigeteiltes Symbol (Tupel, HandlerTuple) gezeigt – links G, rechts CW."""
    if legend == "none" or not handles:
        return
    groups = {}
    for h, lab in zip(handles, labels):
        g = groups.setdefault(lab, [])
        if all(_look(h) != _look(x) for x in g):
            g.append(h)
    L = list(groups)
    H = [g[0] if len(g) == 1 else tuple(g) for g in groups.values()]
    hm = {tuple: HandlerTuple(ndivide=None, pad=0)}
    if legend == "inside":
        axes[0].legend(H, L, loc="best", handler_map=hm)
        return
    r = fig.canvas.get_renderer()
    for nc in range(ncol or min(len(L), 3), 0, -1):          # Spalten verkleinern, bis die Legende in die Breite passt
        leg = fig.legend(H, L, loc="outside upper center", ncol=nc, handler_map=hm)
        if leg.get_window_extent(r).width <= 0.98 * fig.bbox.width or nc == 1:
            rows = int(np.ceil(len(L) / nc))
            nb = int(np.ceil(len(L) / rows))                       # gleichmaessig auf die Zeilen verteilen
            if nb != nc:
                leg.remove()
                fig.legend(H, L, loc="outside upper center", ncol=nb, handler_map=hm)
            return
        leg.remove()


def _new(width_frac, height_cm, nrows, ncols, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize_cm(width_frac, height_cm), layout="constrained",
                           squeeze=False, **kw)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    return fig, ax


# --------------------------------------------------------------------------- #
# Einzelfelder
# --------------------------------------------------------------------------- #
def _xcol(x):
    return "R" if x == "R" else "epoch"


def panel_alpha(ax, level="CW", K=6, rep=0, plans=PLANS, x="R", rmax=1.6, cls=0, oracle=True, runs=None, lang="en",
                best=True, endmarks=False):
    """α² der Heimat-View über Epoche oder Weg R, je Plan; Orakel-Wert als waagerechte Linie."""
    run, T = dt.one_run(runs), _t(lang)
    st = plan_style(level, plans)
    H, L = [], []
    for p in plans:
        v = dt.verlauf(run, level, p, K, rep, cls)
        xv = v[_xcol(x)].to_numpy()
        h, = ax.plot(xv, v.a2_home, **st[p])
        if best:
            eb = v.attrs["ep_best_val"]
            ax.plot(xv[eb], v.a2_home[eb], "o", color=st[p]["color"], mec="white", mew=0.5, zorder=5)
        if endmarks:
            _endmark(ax, xv, v.a2_home.to_numpy(), st[p]["color"], x, rmax)
        H.append(h); L.append(plan_label(p, lang))
    if oracle:
        o = dt.oracle_a(run, level, cls)[dt.VIEWS.index(dt.HOME[cls])] ** 2
        h = ax.axhline(o, color=COL["oracle"], ls=(0, (4, 2)), lw=0.8)
        H.append(h); L.append(T["oracle"])
    ax.set_ylabel(T["a2"].format(v=dt.HOME[cls], lvl=level, c=cls))
    ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


def panel_winkel(ax, level="CW", K=6, rep=0, plans=PLANS, x="R", rmax=1.6, cls=0, runs=None, lang="en", best=True,
                 endmarks=False, sigmoid=None, corner_label=False, oracle_min_label=False):
    """Winkel zwischen a(t) und den Orakel-Gewichten (alle 4 Views), in Grad.
    ABB2: endmarks (Ende nach T als Quadrat, jenseits rmax Pfeil mit R-Wert), sigmoid (zusätzlicher Lauf, schwarz),
    corner_label (Beschriftung am Plateau), oracle_min_label (kleinster Winkel als Zahl)."""
    run, T = dt.one_run(runs), _t(lang)
    st = plan_style(level, plans)
    lines = [(p, st[p]) for p in plans]
    if sigmoid:
        lines.append((sigmoid, dict(color=COL["ink"], ls="-", lw=1.6)))
    H, L, mins = [], [], []
    for p, sty in lines:
        v = dt.verlauf(run, level, p, K, rep, cls)
        xv = v[_xcol(x)].to_numpy()
        h, = ax.plot(xv, v.winkel, **sty)
        if best:
            eb = v.attrs["ep_best_val"]
            ax.plot(xv[eb], v.winkel[eb], "o", color=sty["color"], mec="white", mew=0.5, zorder=5)
        if endmarks:
            _endmark(ax, xv, v.winkel.to_numpy(), sty["color"], x, rmax)
        im = int(v.winkel.to_numpy().argmin())
        mins.append((float(v.winkel[im]), xv[im]))
        H.append(h); L.append(plan_label(p, lang))
    ax.set_ylabel(T["angle"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    if oracle_min_label:
        m, xm = min(mins)
        ax.annotate(rf"{m:.1f}$^\circ$", (xm, m), xytext=(5, 0), textcoords="offset points", ha="left", va="center")
    if corner_label:
        a_or = dt.oracle_a(run, level, cls)
        ecke = float(np.degrees(np.arccos(a_or[dt.VIEWS.index(dt.HOME[cls])])))
        ax.annotate(T["corner"], (0.70, ecke), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", color=COL["mute"])
    return H, L


def panel_kosten(ax, level="CW", K=6, rep=0, plans=PLANS, x="R", rmax=1.6, runs=None, lang="en", best=True,
                 endmarks=False):
    """Train-Kosten (Log, Identität: E = Σ μ_i) je Plan."""
    run, T = dt.one_run(runs), _t(lang)
    st = plan_style(level, plans)
    H, L = [], []
    for p in plans:
        t = dt.traj(run, dt.variant_name(level, p), K, rep)
        xv = t[_xcol(x)].to_numpy()
        h, = ax.plot(xv, t.loss, **st[p])
        if best:
            eb = int(t.ba_val.to_numpy().argmax())
            ax.plot(xv[eb], t.loss.iloc[eb], "o", color=st[p]["color"], mec="white", mew=0.5, zorder=5)
        if endmarks:
            _endmark(ax, xv, t.loss.to_numpy(), st[p]["color"], x, rmax)
        H.append(h); L.append(plan_label(p, lang))
    ax.set_ylabel(T["cost"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


def panel_valba(ax, level="CW", K=6, rep=0, plans=PLANS, x="R", rmax=1.6, runs=None, lang="en", best=True,
                endmarks=False):
    """Val-BA je Plan; Punkt = Best_val."""
    run, T = dt.one_run(runs), _t(lang)
    st = plan_style(level, plans)
    H, L = [], []
    for p in plans:
        t = dt.traj(run, dt.variant_name(level, p), K, rep)
        xv = t[_xcol(x)].to_numpy()
        h, = ax.plot(xv, t.ba_val, **{**st[p], "lw": 0.9})
        if best:
            eb = int(t.ba_val.to_numpy().argmax())
            ax.plot(xv[eb], t.ba_val.iloc[eb], "o", color=st[p]["color"], mec="white", mew=0.5, zorder=5)
        if endmarks:
            _endmark(ax, xv, t.ba_val.to_numpy(), st[p]["color"], x, rmax)
        H.append(h); L.append(plan_label(p, lang))
    ax.set_ylabel(T["val"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


def panel_band(ax, level="CW", K=6, rep=0, plans=PLANS, runs=None, x="R"):
    """Band = Spanne der Best_val-Stellen über die Pläne (auf der x-Achse R oder Epoche)."""
    run = dt.one_run(runs)
    xs = []
    for p in plans:
        t = dt.traj(run, dt.variant_name(level, p), K, rep)
        xs.append(float(t[_xcol(x)].iloc[int(t.ba_val.to_numpy().argmax())]))
    ax.axvspan(min(xs), max(xs), color=COL["band"], alpha=0.12, lw=0, zorder=0)
    return min(xs), max(xs)


def panel_trainval(ax, level="G", K=6, rep=0, plan="e0.003_const", x="epoch", runs=None, lang="en", shade=True,
                   rmax=None, ba=None):
    """Ein Lauf: Train-BA (aus den Snapshots, μ < 0) und Val-BA; Bereich nach Best_val hinterlegt."""
    run, T = dt.one_run(runs), _t(lang)
    if ba is not None and tuple(ba) != ("train", "val"):         # ABB2: freie Auswahl inkl. Test-BA
        return _ba_lines(ax, run, level, plan, K, rep, x, rmax, ba, T, shade)
    k = dt.kosten_ba(run, level, plan, K, rep)
    xv = k["R"] if x == "R" else k["epoch"]
    c = COL[level]
    h1, = ax.plot(xv, k["ba_train"], color=c, lw=1.1)
    h2, = ax.plot(xv, k["ba_val"], color=c, lw=0.8, alpha=0.5)
    ax.plot(xv[k["eb"]], k["ba_val"][k["eb"]], "o", color=c, mec="white", mew=0.5, zorder=5)
    if shade:
        ax.axvspan(xv[k["eb"]], xv[k["n"]], color=COL["harm"], alpha=0.06, lw=0, zorder=0)
        ax.axvline(xv[k["eb"]], color=COL["ink"], ls=":", lw=0.7)
    ax.set_ylabel(T["ba"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return [h1, h2], [T["train"], T["val"]]


def panel_kosten_einzel(ax, level="G", K=6, rep=0, plan="e0.003_const", x="epoch", runs=None, lang="en", box=False,
                        shade=True, rmax=None):
    """Ein Lauf: Train-Kosten (Log); optional Textkasten mit den Kipp-Zahlen zwischen Best_val und Ende."""
    run, T = dt.one_run(runs), _t(lang)
    k = dt.kosten_ba(run, level, plan, K, rep)
    xv = k["R"] if x == "R" else k["epoch"]
    h, = ax.plot(xv, k["loss"], color=COL["ink"], lw=1.1)
    if shade:
        ax.axvspan(xv[k["eb"]], xv[k["n"]], color=COL["harm"], alpha=0.06, lw=0, zorder=0)
        ax.axvline(xv[k["eb"]], color=COL["ink"], ls=":", lw=0.7)
    if box:
        b = T["box"]
        s = "\n".join([b[0].format(n=len(dt.train_data(run)["y_tr"])), b[1].format(r=k["richtiger"]),
                       b[2].format(rf=k["kippt_rf"], fr=k["kippt_fr"]),
                       b[3].format(a=k["ba_train"][k["eb"]], b=k["ba_train"][k["n"]])])
        ax.text(0.98, 0.95, s, transform=ax.transAxes, ha="right", va="top", multialignment="left",
                fontsize=plt.rcParams["font.size"] - 1, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL["start"], lw=0.5))
    ax.set_ylabel(T["cost"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return [h], [T["cost"]]


def panel_zerlegung(ax, level="G", seeds=(3,), style="bars", show=("id_end", "id_best_val", "sig_end", "oracle"),
                    ylim=None, lang="en", spanne=True, spanne_for=("sig_end",)):
    """Zerlegung über K (kverlauf_s3-5.csv). bars: Start / + Prototypen / + Gewichte Identität Ende (schraffiert bei
    Schaden), Linie Best_val, Orakel gestrichelt, Sigmoid als Raute. lines: Gewinn gegenüber frozen je K.
    Mehrere Seeds: Mittel; Spanne (min–max über die Seeds) als Strich (bars) bzw. Band (lines)."""
    T = _t(lang)
    km = dt.kverlauf(seeds)
    d = km[km.level == level].sort_values("K").reset_index(drop=True)
    K = d.K.to_numpy()
    many = len(seeds) > 1 and spanne
    c = COL[level]
    H, L = [], []
    if style == "bars":
        lo = ylim[0] if ylim else min(0.6, float(d.start.min()) - 0.03)
        ax.bar(K, d.start - lo, bottom=lo, width=0.72, color=COL["start"], lw=0)
        ax.bar(K, d.frozen - d.start, bottom=d.start, width=0.72, color=COL["proto"], lw=0)
        H += [Patch(color=COL["start"]), Patch(color=COL["proto"])]; L += [T["start"], T["proto"]]
        if "id_end" in show:
            g = (d.id_end - d.frozen).to_numpy()
            pos = g >= 0
            ax.bar(K[pos], g[pos], bottom=d.frozen[pos], width=0.72, color=c, alpha=0.35, lw=0)
            ax.bar(K[~pos], -g[~pos], bottom=d.id_end[~pos], width=0.72, color="white", edgecolor=COL["harm"],
                   hatch="/////", lw=0.5)
            H += [Patch(color=c, alpha=0.35), Patch(facecolor="white", edgecolor=COL["harm"], hatch="/////")]
            L += [T["gain_id"], T["harm"]]
            if many:
                ax.vlines(K - 0.2, d.id_end_min, d.id_end_max, color=COL["ink"], lw=0.5)
        if "id_best_val" in show:
            h, = ax.plot(K, d.id_best_val, color=c, lw=1.1)
            H.append(h); L.append(T["id_bv"])
        if "oracle" in show:
            h, = ax.plot(K, d.orakel, color=COL["oracle"], ls=(0, (4, 2)), lw=0.8)
            H.append(h); L.append(T["oracle"])
        if "sig_end" in show:
            xs = K + (0.2 if many else 0)
            if many:
                ax.vlines(xs, d.sig_end_min, d.sig_end_max, color=c, lw=0.6)
            h, = ax.plot(xs, d.sig_end, "D", color=c, ms=3, mec="white", mew=0.4, ls="", zorder=5)
            H.append(h); L.append(T["sig"])
        ax.set_ylabel(T["test"])
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(lo, None)
    else:
        ax.axhline(0, color=COL["mute"], lw=0.6)
        spec = {"id_end": ("plus_id_end", dict(color=COL["ident"], ls="-", marker="o", ms=2), T["id_end"]),
                "id_best_val": ("plus_id_best_val", dict(color=COL["ident"], ls=(0, (1, 1))), T["id_bv"]),
                "sig_end": ("plus_sig_end", dict(color=c, ls="-", marker="D", ms=2.5), T["sig"]),
                "oracle": ("plus_orakel", dict(color=COL["oracle"], ls=(0, (4, 2)), lw=0.8), T["oracle"])}
        for key in show:
            col, kw, lab = spec[key]
            h, = ax.plot(K, d[col], **kw)
            if many and key in spanne_for:
                ax.fill_between(K, d[f"{col}_min"], d[f"{col}_max"], color=kw["color"], alpha=0.15, lw=0)
            H.append(h); L.append(lab)
        ax.set_ylabel(T["gain"])
        if ylim:
            ax.set_ylim(*ylim)
    ax.set_xlim(1.4, K.max() + 0.6)
    ax.set_xticks([k for k in K if k % 2 == 0])
    ax.set_xlabel(T["K"])
    return H, L


PANELS = dict(alpha=panel_alpha, winkel=panel_winkel, kosten=panel_kosten, valba=panel_valba, trainval=panel_trainval,
              kosten_einzel=panel_kosten_einzel, zerlegung=panel_zerlegung)
TITLE_KEY = dict(alpha="t_alpha", winkel="t_angle", kosten="t_cost", valba="t_val", trainval="t_trainval",
                 kosten_einzel="t_cost", zerlegung="t_zer")


# --------------------------------------------------------------------------- #
# Bildgruppen
# --------------------------------------------------------------------------- #
def fig_zerlegung_k(seeds=(3, 4, 5), style="bars", show=("id_end", "id_best_val", "sig_end", "oracle"), ylim=None,
                    width_frac=1.0, height_cm=5.0, lang="en", legend="top", panels=("G", "CW"), titles=False,
                    spanne_for=("sig_end",), base="start", spanne_on="total", k_show=None, look="paper"):
    """Zerlegung über K für G und CW (Vorlage fa_zerlegung_k.py); seeds=[3,4,5] → Mittel + Spanne (SG9).
    ABB4: base='start' → Balken relativ zum Start (panel_zerlegung_start; spanne_on, k_show); base='abs' wie bisher.
    ABB5: Voreinstellung = Paper-Kandidat Abb. 2 (base='start', Seeds 3–5, look='paper': Beschriftungen nach
    Rückmeldung 23.10., Feldtitel '(a) G', '(b) CW' in Ebenenfarbe); look='v1' = Darstellung vor ABB5. Für die
    Darstellung vor ABB4: base='abs', seeds=(3,), look='v1'."""
    fig, ax = _new(width_frac, height_cm, 1, len(panels), sharey=True)
    H, L = [], []
    for i, lvl in enumerate(panels):
        if base == "start":
            h, l = panel_zerlegung_start(ax[0, i], lvl, seeds, show, ylim, lang, spanne_on=spanne_on, k_show=k_show,
                                         look=look)
        else:
            h, l = panel_zerlegung(ax[0, i], lvl, seeds, style, show, ylim, lang, spanne_for=spanne_for)
        H += h; L += l
        if look == "paper" and titles is not None:
            name = _t(lang)["level"][lvl] if titles else _t(lang)["level2"][lvl]
            ax[0, i].set_title(f"({chr(97 + i)}) {name}", loc="left", color=COL[lvl])
        else:
            _letter(ax[0, i], i, _t(lang)["level"][lvl], titles)
        if i:
            ax[0, i].set_ylabel("")
    if base == "start" and not ylim:                           # gemeinsame Achse ab 0, oben nach allen Feldern
        y0, y1 = ax[0, 0].get_ylim()
        ax[0, 0].set_ylim(min(0.0, y0) if y0 < -1e-3 else 0.0, y1)
    _legend(fig, ax[0], H, L, legend, ncol=4)
    return fig


def fig_lernrate(level="CW", K=6, rep=0, plans=PLANS, x=("epoch", "R"), rmax=1.6, cls=0, runs=None, width_frac=1.0,
                 height_cm=6.0, lang="en", legend="top", panels=("alpha", "valba"), titles=False, endmarks=False):
    """α² der Heimat-View (oben) und Val-BA (unten) über Epoche und über Weg R (Vorlage f7_lernrate.py)."""
    x = (x,) if isinstance(x, str) else tuple(x)
    fig, ax = _new(width_frac, height_cm, len(panels), len(x), sharex="col", sharey="row")
    H, L, i = [], [], 0
    for r, pn in enumerate(panels):
        for j, xx in enumerate(x):
            kw = dict(level=level, K=K, rep=rep, plans=plans, x=xx, rmax=rmax, runs=runs, lang=lang)
            if pn in ("alpha", "winkel"):
                kw["cls"] = cls
            if pn == "alpha":
                kw["oracle"] = False
            if endmarks:
                kw["endmarks"] = True
            h, l = PANELS[pn](ax[r, j], **kw)
            H += h; L += l
            if r < len(panels) - 1:
                ax[r, j].set_xlabel("")
            if j:
                ax[r, j].set_ylabel("")
            _letter(ax[r, j], i, _t(lang)[TITLE_KEY[pn]], titles); i += 1
    _legend(fig, ax.flatten(), H, L, legend, ncol=len(plans))
    return fig


def fig_ueberschiessen(level="CW", K=6, rep=0, plans=PLANS, rmax=1.6, cls=0, runs=None, width_frac=1.0, height_cm=7.0,
                       lang="en", legend="top", panels=("alpha", "winkel", "kosten", "valba"), ncols=2, titles=False,
                       band=True):
    """Über R: α² + Orakel, Winkel zum Orakel, Train-Kosten, Val-BA; Band = Spanne der Best_val-Stellen
    (Vorlage f7b_ueberschiessen.py). Orakel aus build_data (W_TRUE), nicht hart codiert."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, ax = _new(width_frac, height_cm, nrows, ncols, sharex=True)
    H, L = [], []
    for i, pn in enumerate(panels):
        a = ax.flat[i]
        kw = dict(level=level, K=K, rep=rep, plans=plans, x="R", rmax=rmax, runs=runs, lang=lang)
        if pn in ("alpha", "winkel"):
            kw["cls"] = cls
        h, l = PANELS[pn](a, **kw)
        H += h; L += l
        if band:
            panel_band(a, level, K, rep, plans, runs)
        if i < len(panels) - ncols:
            a.set_xlabel("")
        _letter(a, i, _t(lang)[TITLE_KEY[pn]], titles)
    for a in ax.flat[len(panels):]:
        a.set_visible(False)
    _legend(fig, ax.flatten(), H, L, legend, ncol=len(plans) + 1)
    return fig


def fig_kosten_ba(level="G", K=6, rep=0, plan="e0.003_const", x="epoch", box=True, runs=None, width_frac=1.0,
                  height_cm=5.0, lang="en", legend="top", panels=("kosten_einzel", "trainval"), titles=False, ba=None,
                  rmax=None):
    """Ein Lauf: Train-Kosten oben, Train-/Val-BA unten, Textkasten (Vorlage fb_trainloss.py)."""
    fig, ax = _new(width_frac, height_cm, len(panels), 1, sharex=True)
    H, L = [], []
    for i, pn in enumerate(panels):
        kw = dict(level=level, K=K, rep=rep, plan=plan, x=x, runs=runs, lang=lang)
        if pn == "kosten_einzel":
            kw["box"] = box
        if pn == "trainval" and ba is not None:
            kw["ba"] = ba
        if rmax is not None:
            kw["rmax"] = rmax
        h, l = PANELS[pn](ax[i, 0], **kw)
        if pn != "kosten_einzel":
            H += h; L += l
        if i < len(panels) - 1:
            ax[i, 0].set_xlabel("")
        _letter(ax[i, 0], i, _t(lang)[TITLE_KEY[pn]], titles)
    fig.align_ylabels(ax[:, 0])
    _legend(fig, ax[:, 0], H, L, legend, ncol=2)
    return fig


def fig_kombi(panels, nrows=1, ncols=None, sharex=False, width_frac=1.0, height_cm=4.5, lang="en", legend="top",
              titles=False, band=None, legend_ncol=None):
    """Freie Kombination von Einzelfeldern: panels = [(name, kwargs), …] mit name in PANELS
    ('alpha', 'winkel', 'kosten', 'valba', 'trainval', 'kosten_einzel', 'zerlegung'); band = dict(level, K, rep, plans)
    zeichnet die Best_val-Spanne in alle Felder mit x = R."""
    ncols = ncols or int(np.ceil(len(panels) / nrows))
    fig, ax = _new(width_frac, height_cm, nrows, ncols, sharex=sharex)
    H, L = [], []
    for i, (pn, kw) in enumerate(panels):
        a = ax.flat[i]
        h, l = PANELS[pn](a, **{"lang": lang, **kw})
        H += h; L += l
        a = getattr(a, "_ersatz", a)
        if band and kw.get("x", "R") == "R" and pn != "zerlegung":
            panel_band(a, **band)
        _letter(a, i, _t(lang)[TITLE_KEY[pn]], titles)
    for a in ax.flat[len(panels):]:
        a.set_visible(False)
    _legend(fig, ax.flatten(), H, L, legend, ncol=legend_ncol)
    return fig


# --------------------------------------------------------------------------- #
# Zahlen für den Text (dieselben Daten wie die Bilder)
# --------------------------------------------------------------------------- #
def text_numbers(level="CW", K=6, rep=0, plans=PLANS, cls=0, runs=None, kosten_levels=("G", "CW"),
                 kosten_plan="e0.003_const", kostenbeitrag=False, margin=0.1, sigmoid=None, pfad2=False, pfad2_kw=None,
                 start_ba=False, start_seeds=(3, 4, 5)):
    """Kennzahlen, die die Bilder tragen: je Plan Best_val-Epoche und -R, Winkel-Minimum (Grad, R); je Ebene für einen
    Lauf (Identität): Kosten nach Best_val, richtiger / kippen r→f / f→r, Train-BA Best_val → Ende, Kontrollen."""
    run = dt.one_run(runs)
    rows = []
    for p in plans:
        v = dt.verlauf(run, level, p, K, rep, cls)
        eb, im = v.attrs["ep_best_val"], int(v.winkel.to_numpy().argmin())
        rows.append(dict(bild="lernrate/ueberschiessen", lauf=f"{v.attrs['variant']} K{K} rep{rep}",
                         groesse="Best_val: Epoche, R, Val-BA", wert=f"{eb}, {v.R[eb]:.3f}, {v.ba_val[eb]:.3f}"))
        rows.append(dict(bild="ueberschiessen", lauf=f"{v.attrs['variant']} K{K} rep{rep}",
                         groesse="Winkel zum Orakel: Minimum (Grad) bei R; am Best_val; am Ende",
                         wert=f"{v.winkel[im]:.1f} bei R {v.R[im]:.3f}; {v.winkel[eb]:.1f}; {v.winkel.iloc[-1]:.1f}"))
    rb = [dt.verlauf(run, level, p, K, rep, cls) for p in plans]
    rs = [x.R[x.attrs["ep_best_val"]] for x in rb]
    rows.append(dict(bild="ueberschiessen", lauf=f"{level} K{K} rep{rep}, {len(plans)} Pläne",
                     groesse="Spanne der Best_val-Stellen R (Band)", wert=f"{min(rs):.3f} – {max(rs):.3f}"))
    for lvl in kosten_levels:
        k = dt.kosten_ba(run, lvl, kosten_plan, K, rep)
        lauf = f"{dt.variant_name(lvl, kosten_plan)} K{K} rep{rep}"
        rows += [dict(bild="kosten_ba", lauf=lauf, groesse="Best_val-Epoche; Kosten nach Best_val (E_Ende − E_Best_val)",
                      wert=f"{k['eb']}; {k['dE']:+.1f}"),
                 dict(bild="kosten_ba", lauf=lauf, groesse="richtiger / kippen r→f / f→r",
                      wert=f"{k['richtiger']} / {k['kippt_rf']} / {k['kippt_fr']}"),
                 dict(bild="kosten_ba", lauf=lauf, groesse="Train-BA Best_val → Ende; Val-BA Best_val → Ende",
                      wert=f"{k['ba_train'][k['eb']]:.3f} → {k['ba_train'][k['n']]:.3f}; "
                           f"{k['ba_val'][k['eb']]:.3f} → {k['ba_val'][k['n']]:.3f}"),
                 dict(bild="kosten_ba", lauf=lauf, groesse="Kontrolle max|Σμ − loss|",
                      wert=f"{k['check_E_loss']:.1e}")]
    if kostenbeitrag:                                         # ABB2: zusätzliche Zeilen, bestehende unverändert
        pl = [kosten_plan] + ([sigmoid] if sigmoid else [])
        for lvl in kosten_levels:
            for p in pl:
                rows += kostenbeitrag_rows(run, lvl, p, K, rep, margin)
    if pfad2:                                                 # ABB3: Zahlen zu fig_pfad2, bestehende unverändert
        rows += pfad2_rows(run, K=K, rep=rep, **(pfad2_kw or {}))
    if start_ba:                                              # ABB4: Start-BA für die Bildunterschrift der Zerlegung
        rows += start_rows(tuple(start_seeds))
        rows += start_rows_abb5(tuple(start_seeds))           # ABB5: Mittel und Spanne, G = CW geprüft
    return pd.DataFrame(rows)


# Werte aus den Folien-Entwürfen (Chat 20.10.) für die Prüfung in Abschnitt 4 des Auftrags ABB
ENTWURF = {"best_val_R_cw": (0.37, 0.56), "winkel_min_cw": 5.0,
           "G": dict(richtiger=336, kippt_rf=70, kippt_fr=34, tr_bv=0.817, tr_end=0.757),
           "CW": dict(richtiger=360, kippt_rf=60, kippt_fr=36, tr_bv=0.855, tr_end=0.815)}


def pruefung_entwuerfe(runs=None, K=6, rep=0, plans=PLANS):
    """Vergleicht die Kennzahlen mit den Entwürfen (Seed 3). Meldet Abweichungen, passt nichts an."""
    run = dt.one_run(runs)
    rows = []
    rs = [dt.verlauf(run, "CW", p, K, rep).pipe(lambda v: v.R[v.attrs["ep_best_val"]]) for p in plans]
    lo, hi = ENTWURF["best_val_R_cw"]
    rows.append(dict(groesse="CW Best_val-R Spanne", entwurf=f"{lo:.2f}–{hi:.2f}", rechnung=f"{min(rs):.3f}–{max(rs):.3f}",
                     gleich=round(min(rs), 2) == lo and round(max(rs), 2) == hi))
    wm = min(float(dt.verlauf(run, "CW", p, K, rep).winkel.min()) for p in plans)
    rows.append(dict(groesse="CW Winkel-Minimum (Grad)", entwurf=f"≈ {ENTWURF['winkel_min_cw']:.0f}",
                     rechnung=f"{wm:.2f}", gleich=abs(wm - ENTWURF["winkel_min_cw"]) < 1.0))
    for lvl in ("G", "CW"):
        k = dt.kosten_ba(run, lvl, "e0.003_const", K, rep)
        e = ENTWURF[lvl]
        ist = (k["richtiger"], k["kippt_rf"], k["kippt_fr"])
        rows.append(dict(groesse=f"{lvl} K{K} rep{rep}: richtiger / r→f / f→r",
                         entwurf=f"{e['richtiger']} / {e['kippt_rf']} / {e['kippt_fr']}", rechnung=" / ".join(map(str, ist)),
                         gleich=ist == (e["richtiger"], e["kippt_rf"], e["kippt_fr"])))
        a, b = k["ba_train"][k["eb"]], k["ba_train"][k["n"]]
        rows.append(dict(groesse=f"{lvl} K{K} rep{rep}: Train-BA Best_val → Ende", entwurf=f"{e['tr_bv']:.3f} → {e['tr_end']:.3f}",
                         rechnung=f"{a:.3f} → {b:.3f}", gleich=round(a, 3) == e["tr_bv"] and round(b, 3) == e["tr_end"]))
    return pd.DataFrame(rows)


# =========================================================================== #
# Auftrag ABB2 (21.10.2026): Pfad über R, Diskrepanz Kosten ↔ BA, Grenzsprünge und Kostenbeitrag
# Alles zusätzlich; bestehende Funktionen behalten ihr Verhalten (neue Parameter mit Voreinstellung = bisher).
# =========================================================================== #
TXT["en"].update(
    cost_mu=r"Training cost $E=\Sigma_i\,\mu_i$", ba_set=dict(train="training BA", val="validation BA", test="test BA"),
    normalized="normalized to [0, 1]", sig_suffix=" (sigmoid)", mu=r"$\mu$", median=r"median $\mu$",
    margin=r"$|\mu| \leq {m}$", fr=r"wrong $\to$ correct", rf=r"correct $\to$ wrong", bilanz="balance",
    kum="cumulative balance", count="points / {w} ep.", kum_y="cum. balance",
    kb=dict(sicher=r"safely correct ($\mu<-{m}$)", grenze=r"border ($|\mu|\leq {m}$)", falsch=r"wrong ($\mu>{m}$)",
            summe="total"), kb_y=r"cum. $\Delta E$", t_diskrepanz="cost vs. BA", t_mu=r"distribution of $\mu$",
    t_grenz="boundary crossings", t_kb="cost contribution", set=dict(train="train", val="val"))
TXT["de"].update(
    cost_mu=r"Train-Kosten $E=\Sigma_i\,\mu_i$", ba_set=dict(train="Train-BA", val="Val-BA", test="Test-BA"),
    normalized="auf [0, 1] skaliert", sig_suffix=" (Sigmoid)", mu=r"$\mu$", median=r"Median $\mu$",
    margin=r"$|\mu| \leq {m}$", fr=r"falsch $\to$ richtig", rf=r"richtig $\to$ falsch", bilanz="Bilanz",
    kum="Bilanz kumuliert", count="Punkte / {w} Ep.", kum_y="Bilanz kum.",
    kb=dict(sicher=r"sicher richtig ($\mu<-{m}$)", grenze=r"Grenzbereich ($|\mu|\leq {m}$)", falsch=r"falsch ($\mu>{m}$)",
            summe="Summe"), kb_y=r"$\Delta E$ kum.", t_diskrepanz="Kosten vs. BA", t_mu=r"Verteilung von $\mu$",
    t_grenz="Grenzsprünge", t_kb="Kostenbeitrag", set=dict(train="Train", val="Val"))


def _endmark(ax, xv, yv, color, x="R", rmax=None):
    """Ende des Laufs (Epoche T): Quadrat; Enden jenseits rmax (x = R) werden zu einem Pfeil am rechten Rand mit den
    R-Werten aus den Daten zusammengefasst (z. B. 'R = 4, 12'; eine Farbe, wenn es nur ein Lauf ist)."""
    if x == "R" and rmax and xv[-1] > rmax:
        alt = getattr(ax, "_endpfeil", None)
        werte = (alt["werte"] if alt else []) + [(float(xv[-1]), float(yv[-1]), color)]
        if alt:
            alt["ann"].remove()
        werte.sort()
        y = float(np.mean([w[1] for w in werte]))
        c = werte[0][2] if len(werte) == 1 else COL["ink"]
        txt = r"$R$ = " + ", ".join(f"{round(w[0], 1):g}" for w in werte)
        ann = ax.annotate(txt, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-3, -6),
                          textcoords="offset points", ha="right", va="top", color=c,
                          fontsize=plt.rcParams["font.size"] - 1,
                          arrowprops=dict(arrowstyle="-|>", color=c, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
        ax._endpfeil = dict(ann=ann, werte=werte)
    else:
        ax.plot(xv[-1], yv[-1], "s", color=color, ms=3, mec="white", mew=0.4, zorder=6, clip_on=False)


def _shade_after(ax, xv, eb, n):
    ax.axvspan(xv[eb], xv[n], color=COL["harm"], alpha=0.06, lw=0, zorder=0)
    ax.axvline(xv[eb], color=COL["ink"], ls=":", lw=0.7)


BA_STYLE = dict(train=dict(lw=1.1, alpha=1.0, ls="-"), val=dict(lw=0.8, alpha=0.55, ls="-"),
                test=dict(lw=0.8, alpha=0.9, ls=(0, (1, 1))))


def _ba_lines(ax, run, level, plan, K, rep, x, rmax, ba, T, shade, ls_override=None, suffix="", color=None):
    """BA-Kurven (train/val/test) aus den Snapshots (data.verlauf_sets, mit Kontrollen)."""
    v = dt.verlauf_sets(run, level, plan, K, rep, tuple(ba))
    xv = v["R"] if x == "R" else v["epoch"]
    c = color or COL[level]
    H, L = [], []
    for s in ba:
        sty = dict(BA_STYLE[s])
        if ls_override:
            sty["ls"] = ls_override
        h, = ax.plot(xv, v[f"ba_{s}"], color=c, **sty)
        H.append(h); L.append(T["ba_set"][s] + suffix)
    if "val" in ba:
        ax.plot(xv[v["eb"]], v["ba_val"][v["eb"]], "o", color=c, mec="white", mew=0.5, zorder=5)
    if shade:
        _shade_after(ax, xv, v["eb"], v["n"])
    ax.set_ylabel(T["ba"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


def panel_diskrepanz(ax, level="CW", K=6, rep=0, plan="e0.003_const", x="R", rmax=1.6, mode="twin",
                     ba=("train", "val"), sigmoid=None, shade_after_best=True, runs=None, lang="en", band=None):
    """Ein Lauf: Train-Kosten E = Σ μ (aus den Snapshots, = loss bei Identität) und BA je Epoche.
    mode 'twin': Kosten links (schwarz), BA rechts (Ebenenfarbe); 'stacked': zwei flache Felder (Kosten oben, BA unten);
    'normalized': beide auf [0, 1] in einer Achse (nur zum Ausprobieren, im Bild gekennzeichnet).
    sigmoid: derselbe Satz Kurven für den Sigmoid-Lauf gestrichelt (Kosten ebenfalls als Σ μ, ungequetscht).
    band: dict(level, K, rep, plans) → Band der Best_val-Stellen (x = R) in allen erzeugten Achsen."""
    run, T = dt.one_run(runs), _t(lang)
    runs_ = [(plan, "-", "")] + ([(sigmoid, "--", T["sig_suffix"])] if sigmoid else [])
    data = {p: dt.verlauf_sets(run, level, p, K, rep, tuple(ba)) for p, _, _ in runs_}
    c = COL[level]
    if mode == "stacked":
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(2, 1, hspace=0.0)
        ax.remove()
        ax_c = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1], sharex=ax_c)
        ax._ersatz = ax_c                                    # Aufrufer beschriften die neue obere Achse
        ax_c.tick_params(labelbottom=False)
    elif mode == "twin":
        ax_c, ax_b = ax, ax.twinx()
        ax_b.spines["right"].set_visible(True)
        ax_b.tick_params(axis="y", colors=c)
    else:
        ax_c = ax_b = ax
    H, L = [], []
    for p, ls, suf in runs_:
        v = data[p]
        xv = v["R"] if x == "R" else v["epoch"]
        E = v["E_mu"]
        if mode == "normalized":
            E = (E - E.min()) / (E.max() - E.min())
        h, = ax_c.plot(xv, E, color=COL["ink"], lw=1.1, ls=ls)
        H.append(h); L.append(T["cost_mu"] + suf)
        for s in ba:
            y = v[f"ba_{s}"]
            if mode == "normalized":
                y = (y - y.min()) / (y.max() - y.min())
            sty = {**BA_STYLE[s], "ls": ls if s != "test" else BA_STYLE["test"]["ls"]}
            h, = ax_b.plot(xv, y, color=c, **sty)
            H.append(h); L.append(T["ba_set"][s] + suf)
        if "val" in ba:
            yb = v["ba_val"] if mode != "normalized" else (v["ba_val"] - v["ba_val"].min()) / np.ptp(v["ba_val"])
            ax_b.plot(xv[v["eb"]], yb[v["eb"]], "o", color=c, mec="white", mew=0.5, zorder=5)
    v0 = data[plan]
    xv0 = v0["R"] if x == "R" else v0["epoch"]
    axes = [ax_c] if ax_c is ax_b else [ax_c, ax_b]
    if shade_after_best:
        for a in ([ax_c, ax_b] if mode == "stacked" else [ax_c]):
            _shade_after(a, xv0, v0["eb"], v0["n"])
    if band and x == "R":
        for a in ([ax_c, ax_b] if mode == "stacked" else [ax_c]):
            panel_band(a, runs=runs, **band)
    if mode == "normalized":
        ax_c.set_ylabel(T["normalized"])
        ax_c.text(0.98, 0.03, T["normalized"], transform=ax_c.transAxes, ha="right", va="bottom", color=COL["mute"])
    else:
        ax_c.set_ylabel(T["cost"]); ax_b.set_ylabel(T["ba"])
        if mode == "twin":
            ax_b.yaxis.label.set_color(c)
    for a in axes:
        if x == "R" and rmax:
            a.set_xlim(0, rmax)
    (ax_b if mode == "stacked" else ax_c).set_xlabel(T["R"] if x == "R" else T["epoch"])
    if mode == "stacked":
        ax_c.set_xlabel("")
        ax_c.figure.align_ylabels([ax_c, ax_b])
    return H, L


def panel_mu_dichte(ax, level="CW", K=6, rep=0, plan="e0.003_const", set="train", x="epoch", rmax=None, margin=0.1,
                    bins=80, runs=None, lang="en"):
    """Dichte von μ (Train oder Val) über die Zeit als Graustufenbild (log), Linie μ = 0, Band |μ| < margin, Median."""
    from matplotlib.colors import LogNorm
    run, T = dt.one_run(runs), _t(lang)
    v = dt.verlauf_sets(run, level, plan, K, rep, (set,))
    mu = v[f"mu_{set}"]
    xv = v["R"] if x == "R" else v["epoch"].astype(float)
    yedges = np.linspace(-1, 1, bins + 1)
    Hm = np.stack([np.histogram(m, bins=yedges)[0] for m in mu])        # (Epochen, bins)
    xe = np.r_[xv[0] - (xv[1] - xv[0]) / 2, (xv[1:] + xv[:-1]) / 2, xv[-1] + (xv[-1] - xv[-2]) / 2]
    Hm = np.where(Hm > 0, Hm, np.nan)
    ax.pcolormesh(xe, yedges, Hm.T, cmap="Greys", norm=LogNorm(vmin=1, vmax=np.nanmax(Hm)), shading="flat",
                  rasterized=True)
    ax.axhspan(-margin, margin, color=COL["G" if level == "G" else "CW"], alpha=0.12, lw=0)
    ax.axhline(0, color=COL["ink"], lw=0.6)
    h, = ax.plot(xv, np.median(mu, axis=1), color=COL[level], lw=1.0)
    ax.plot(xv[v["eb"]], np.median(mu[v["eb"]]), "o", color=COL[level], mec="white", mew=0.5, zorder=5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(T["mu"] + f" ({T['set'][set]})"); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return [h, Patch(color=COL[level], alpha=0.12)], [T["median"], T["margin"].format(m=margin)]


def panel_grenzspruenge(ax, level="CW", K=6, rep=0, plan="e0.003_const", sets=("train", "val"), win=10, x="epoch",
                        rmax=None, cumulative=False, runs=None, lang="en", shade_after_best=True):
    """Je Fenster (win Epochen) Anzahl falsch → richtig (nach oben) und richtig → falsch (nach unten) als Balken,
    Bilanz als Linie; cumulative=True: Bilanz seit Start als Linie statt Balken. richtig = μ < 0."""
    run, T = dt.one_run(runs), _t(lang)
    v = dt.verlauf_sets(run, level, plan, K, rep, tuple(sets))
    xv = v["R"] if x == "R" else v["epoch"].astype(float)
    c = COL[level]
    shades = {"train": 1.0, "val": 0.45}
    H, L = [], []
    n = len(xv)
    starts = np.arange(1, n, win)
    for j, s in enumerate(sets):
        fr, rf = dt.grenzwechsel(v[f"mu_{s}"])
        a = shades.get(s, 0.7)
        if cumulative:
            h, = ax.plot(xv, np.cumsum(fr - rf), color=c, alpha=a, lw=1.1)
            H.append(h); L.append(f"{T['kum']} ({T['set'][s]})")
            continue
        up = np.array([fr[i:i + win].sum() for i in starts])
        dn = np.array([rf[i:i + win].sum() for i in starts])
        xm = np.array([xv[min(i + win // 2, n - 1)] for i in starts])
        width = (xv[min(win, n - 1)] - xv[0]) / (len(sets) + 0.5)
        off = (j - (len(sets) - 1) / 2) * width
        b1 = ax.bar(xm + off, up, width=width, color=c, alpha=a, lw=0)
        ax.bar(xm + off, -dn, width=width, color=COL["harm"], alpha=a, lw=0)
        h, = ax.plot(xm, up - dn, color=COL["ink"], lw=0.8, alpha=a, marker="." if j else None, ms=2)
        H += [b1, h]; L += [f"{T['fr']} ({T['set'][s]})", f"{T['bilanz']} ({T['set'][s]})"]
    if not cumulative:
        H.append(Patch(color=COL["harm"])); L.append(T["rf"])
    ax.axhline(0, color=COL["mute"], lw=0.5)
    if shade_after_best:
        _shade_after(ax, xv, v["eb"], v["n"])
    ax.set_ylabel(T["kum_y"] if cumulative else T["count"].format(w=win))
    ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


def panel_kostenbeitrag(ax, level="CW", K=6, rep=0, plan="e0.003_const", margin=0.1, f=None, x="epoch", rmax=None,
                        runs=None, lang="en", shade_after_best=True):
    """Kumulierte Änderung der Train-Kosten, aufgeteilt nach Lage des Punkts in der Epoche davor (sicher richtig /
    Grenzbereich / falsch) und Summe (= Kosten − Start). f: None = Kosten des Laufs (Identität bzw. Sigmoid),
    'identity' oder 'sigmoid' (β = 10) erzwingen."""
    from scipy.special import expit
    run, T = dt.one_run(runs), _t(lang)
    v = dt.verlauf_sets(run, level, plan, K, rep, ("train",))
    fn = dt.squash(plan)[0] if f is None else ((lambda m: m) if f == "identity" else (lambda m: expit(10.0 * m)))
    kb = dt.kostenbeitrag(v["mu_train"], fn, margin)
    xv = v["R"] if x == "R" else v["epoch"]
    sty = dict(sicher=dict(color=COL[level], ls="-"), grenze=dict(color=COL[level], ls=(0, (3, 1.5)), alpha=0.7),
               falsch=dict(color=COL["harm"], ls="-"), summe=dict(color=COL["ink"], ls="-", lw=1.4))
    H, L = [], []
    for k in ("sicher", "grenze", "falsch", "summe"):
        h, = ax.plot(xv, kb[k], **{"lw": 1.0, **sty[k]})
        H.append(h); L.append(T["kb"][k].format(m=margin))
    ax.axhline(0, color=COL["mute"], lw=0.5)
    if shade_after_best:
        _shade_after(ax, xv, v["eb"], v["n"])
    ax.set_ylabel(T["kb_y"]); ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)
    return H, L


PANELS.update(diskrepanz=panel_diskrepanz, mu_dichte=panel_mu_dichte, grenzspruenge=panel_grenzspruenge,
              kostenbeitrag=panel_kostenbeitrag)
TITLE_KEY.update(diskrepanz="t_diskrepanz", mu_dichte="t_mu", grenzspruenge="t_grenz", kostenbeitrag="t_kb")


def fig_pfad(level="CW", K=6, rep=0, plans=PLANS, rmax=1.6, panels=("winkel", "diskrepanz"), ncols=2, endmarks=True,
             sigmoid=None, diskrepanz_plan="e0.003_const", diskrepanz_mode="twin", ba=("train", "val"), width_frac=1.0,
             height_cm=4.8, lang="en", legend="top", titles=False, band=True, layout="row", corner_label=False,
             oracle_min_label=False, cls=0, runs=None, legend_ncol=None):
    """Abb. 3 (Kandidat): Felder über den Weg R mit gemeinsamer x-Achse (bis rmax) und gleichem Best_val-Band.
    panels frei aus PANELS ('winkel', 'diskrepanz', 'alpha', 'valba', 'kosten', …).
    layout 'row': Felder nebeneinander (ncols); 'left_big': erstes Feld links über volle Höhe, die übrigen rechts
    gestapelt (mit diskrepanz_mode='stacked' wird die Diskrepanz rechts in Kosten oben / BA unten geteilt)."""
    T = _t(lang)
    bd = dict(level=level, K=K, rep=rep, plans=plans) if band else None
    if layout == "left_big":
        fig = plt.figure(figsize=figsize_cm(width_frac, height_cm), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.0)
        gs = fig.add_gridspec(1, 2)
        right = gs[1].subgridspec(max(len(panels) - 1, 1), 1, hspace=0.0)
        axes = [fig.add_subplot(gs[0])] + [fig.add_subplot(right[i]) for i in range(len(panels) - 1)]
    else:
        nrows = int(np.ceil(len(panels) / ncols))
        fig, ax = _new(width_frac, height_cm, nrows, ncols)
        axes = list(ax.flat)
    H, L = [], []
    for i, pn in enumerate(panels):
        a = axes[i]
        kw = dict(level=level, K=K, rep=rep, x="R", rmax=rmax, runs=runs, lang=lang)
        if pn == "diskrepanz":
            kw.update(plan=diskrepanz_plan, mode=diskrepanz_mode, ba=ba, sigmoid=sigmoid, band=bd)
        elif pn in ("trainval", "kosten_einzel", "mu_dichte", "grenzspruenge", "kostenbeitrag"):
            kw.update(plan=diskrepanz_plan)
        else:
            kw.update(plans=plans)
            if pn in ("alpha", "winkel"):
                kw["cls"] = cls
            if endmarks:
                kw["endmarks"] = True
            if pn == "winkel":
                kw.update(sigmoid=sigmoid, corner_label=corner_label, oracle_min_label=oracle_min_label)
        h, l = PANELS[pn](a, **kw)
        H += h; L += l
        a = axes[i] = getattr(a, "_ersatz", a)
        if band and pn != "diskrepanz":
            panel_band(a, level, K, rep, plans, runs)
        _letter(a, i, T[TITLE_KEY[pn]], titles)
    for a in axes[len(panels):]:
        a.set_visible(False)
    _legend(fig, axes, H, L, legend, ncol=legend_ncol)
    return fig


def fig_grenzspruenge(level="CW", K=6, rep=0, plan="e0.003_const", sigmoid=None,
                      panels=("mu_dichte", "grenzspruenge", "kostenbeitrag"), margin=0.1, win=10, x="epoch",
                      sets=("train", "val"), mu_set="train", cumulative=False, width_frac=1.0, height_cm=8.0, lang="en",
                      legend="top", titles=False, legend_ncol=None, runs=None):
    """Explorativ (Teil B): Felder untereinander für einen Lauf; mit sigmoid= zweite Spalte für den Sigmoid-Lauf
    (gleiche y-Achsen je Zeile)."""
    T = _t(lang)
    cols = [plan] + ([sigmoid] if sigmoid else [])
    fig, ax = _new(width_frac, height_cm, len(panels), len(cols), sharex="col", sharey="row")
    H, L, i = [], [], 0
    for r, pn in enumerate(panels):
        for j, p in enumerate(cols):
            kw = dict(level=level, K=K, rep=rep, plan=p, x=x, runs=runs, lang=lang)
            if pn == "mu_dichte":
                kw.update(set=mu_set, margin=margin)
            elif pn == "grenzspruenge":
                kw.update(sets=sets, win=win, cumulative=cumulative)
            elif pn == "kostenbeitrag":
                kw.update(margin=margin)
            h, l = PANELS[pn](ax[r, j], **kw)
            H += h; L += l
            if r < len(panels) - 1:
                ax[r, j].set_xlabel("")
            if j:
                ax[r, j].set_ylabel("")
            if r == 0 and len(cols) > 1:
                ax[r, j].set_title(plan_label(p, lang), loc="right")
            _letter(ax[r, j], i, T[TITLE_KEY[pn]], titles); i += 1
    fig.align_ylabels(ax[:, 0])
    _legend(fig, ax.flatten(), H, L, legend, ncol=legend_ncol)
    return fig


# --------------------------------------------------------------------------- #
# Zahlen und Prüfungen ABB2
# --------------------------------------------------------------------------- #
def kostenbeitrag_rows(run, level, plan, K, rep, margin=0.1):
    """Kostenbeitrag je Gruppe und Grenzsprünge vor/nach Best_val (Train; Kosten des Laufs)."""
    v = dt.verlauf_sets(run, level, plan, K, rep, ("train", "val"))
    kb = dt.kostenbeitrag(v["mu_train"], dt.squash(plan)[0], margin)
    eb, n = v["eb"], v["n"]
    lauf = f"{v['variant']} K{K} rep{rep}"
    vn = {k: (kb[k][eb] - kb[k][0], kb[k][n] - kb[k][eb]) for k in kb}
    rows = [dict(bild="grenzspruenge", lauf=lauf, groesse=f"Kostenbeitrag vor/nach Best_val (margin {margin}): {k}",
                 wert=f"{a:+.1f} / {b:+.1f}") for k, (a, b) in vn.items()]
    for s in ("train", "val"):
        fr, rf = dt.grenzwechsel(v[f"mu_{s}"])
        rows.append(dict(bild="grenzspruenge", lauf=lauf, groesse=f"Grenzsprünge {s} vor/nach Best_val: f→r, r→f",
                         wert=f"{fr[1:eb + 1].sum()} / {fr[eb + 1:].sum()}, {rf[1:eb + 1].sum()} / {rf[eb + 1:].sum()}"))
    return rows


KB_AUFTRAG = {"G": dict(sicher=(-12.6, -79.4), grenze=(-46.5, -4.4), falsch=(-15.0, 24.4)),
              "CW": dict(sicher=(-25.3, -62.4), grenze=(-63.3, -5.3), falsch=(-15.3, 1.6))}


def pruefung_kostenbeitrag(runs=None, K=6, rep=0, margin=0.1):
    """Abgleich mit den Zahlen aus dem Auftrag ABB2 (Seed 3, Identität, K 6, rep 0, margin 0.1). Meldet, passt nichts an."""
    run = dt.one_run(runs)
    rows = []
    for lvl, soll in KB_AUFTRAG.items():
        v = dt.verlauf_sets(run, lvl, "e0.003_const", K, rep, ("train",))
        kb = dt.kostenbeitrag(v["mu_train"], dt.squash("e0.003_const")[0], margin)
        eb, n = v["eb"], v["n"]
        for k, (a, b) in soll.items():
            ia, ib = kb[k][eb] - kb[k][0], kb[k][n] - kb[k][eb]
            rows.append(dict(groesse=f"{lvl} {k} vor/nach Best_val", auftrag=f"{a:+.1f} / {b:+.1f}",
                             rechnung=f"{ia:+.1f} / {ib:+.1f}", gleich=round(ia, 1) == a and round(ib, 1) == b))
    return pd.DataFrame(rows)


def kontrollen(level_plans, K=6, rep=0, runs=None, sets=("train", "val")):
    """Kontrollen gegen den Log je verwendetem Lauf: max |Abweichung| Val-BA, Σ f(μ) − loss (und Train/Test, falls geloggt)."""
    run = dt.one_run(runs)
    rows = []
    for lvl, p in level_plans:
        v = dt.verlauf_sets(run, lvl, p, K, rep, tuple(sets))
        rows.append(dict(lauf=f"{v['variant']} K{K} rep{rep}", **{f"max_abw_{k}": f"{x:.1e}" for k, x in v["check"].items()}))
    return pd.DataFrame(rows)


# =========================================================================== #
# Auftrag ABB3 (23.10.2026): vereinfachte Fassung des Pfad-Bildes (fig_pfad2)
# Zusätzlich; bestehende Funktionen, Parameter und Texte unverändert. Steilheit der Sigmoide heißt β (F01).
# =========================================================================== #
TXT["en"].update(angle2="Angle to oracle ($^\\circ$)\n(G: global, CW: class 0)", ident="identity",
                 sig2=r"sigmoid $\beta$ = {b:g}", end="end of training", lr_band=r"identity, other $\eta$ (range)",
                 cost2=r"$\Sigma_i\,\mu_i$", level2=dict(G="G", CW="CW"),
                 cost_leg=r"training cost $\Sigma_i\,\mu_i$", best2="best val. epoch", end2="end")
TXT["de"].update(angle2="Winkel zum Orakel ($^\\circ$)\n(G: global, CW: Klasse 0)", ident="Identität",
                 sig2=r"Sigmoid $\beta$ = {b:g}", end="Ende des Trainings", lr_band=r"Identität, andere $\eta$ (Spanne)",
                 cost2=r"$\Sigma_i\,\mu_i$", level2=dict(G="G", CW="CW"),
                 cost_leg=r"Train-Kosten $\Sigma_i\,\mu_i$", best2="Best_val", end2="Ende")

DATASETS = {3: dt.RUN_S3}                                     # Datensatz-Seed -> Lauf (derzeit nur Seed 3)
LS2 = dict(ident=(0, (3.5, 1.5)), sig="-")                    # Identität gestrichelt, Sigmoid durchgezogen
BA2 = dict(train=dict(lw=1.2, alpha=1.0), val=dict(lw=0.7, alpha=0.55))


def _run2(dataset, runs):
    if runs is not None:
        return dt.one_run(runs)
    if dataset not in DATASETS:
        raise ValueError(f"Datensatz {dataset} nicht hinterlegt (vorhanden: {sorted(DATASETS)}) – runs= übergeben")
    return DATASETS[dataset]


def _laeufe2(ident, sigmoid):
    return [(p, k) for p, k in ((ident, "ident"), (sigmoid, "sig")) if p]


def _xaxis2(ax, x, rmax, T):
    ax.set_xlabel(T["R"] if x == "R" else T["epoch"])
    if x == "R" and rmax:
        ax.set_xlim(0, rmax)


def _bv_span2(run, level, K, rep, plans, x="R"):
    """Spanne der Best_val-Stellen (x = R oder Epoche) über die Pläne einer Ebene."""
    xs = []
    for p in plans:
        t = dt.traj(run, dt.variant_name(level, p), K, rep)
        xs.append(float(t[_xcol(x)].iloc[int(t.ba_val.to_numpy().argmax())]))
    return min(xs), max(xs)


def eta_band(run, level, K=6, rep=0, plans=PLANS, x="R", r_bis=1.6, n=400):
    """η-Band: min und max des Winkels zum Orakel über die Identitäts-Pläne je Stelle (x = R: Raster 0 … r_bis,
    jeder Plan nur bis zu seinem R am Ende; x = Epoche: alle Epochen). Gibt (grid, lo, hi) zurück."""
    vs = [dt.verlauf(run, level, p, K, rep, 0) for p in plans]
    if x == "R":
        grid = np.linspace(0, r_bis or max(float(v.R.iloc[-1]) for v in vs), n)
        Y = np.stack([np.where(grid <= v.R.iloc[-1], np.interp(grid, v.R, v.winkel), np.nan) for v in vs])
    else:
        grid, Y = vs[0].epoch.to_numpy(float), np.stack([v.winkel.to_numpy() for v in vs])
    return grid, np.nanmin(Y, 0), np.nanmax(Y, 0)


def panel_winkel2(ax, levels=("G", "CW"), K=6, rep=0, ident="e0.003_const", sigmoid="sig10_e0.003_const", x="R",
                  rmax=1.6, run=None, lang="en", lr_band=False, lr_band_plans=PLANS, lr_band_levels=("CW",),
                  corner_label=True):
    """Winkel zu den Orakel-Gewichten der jeweiligen Ebene (G: W_TRUE['glob'], CW: W_TRUE[0], Klasse 0), je Ebene ein
    Identitäts- (gestrichelt) und ein Sigmoid-Lauf (durchgezogen); Punkt = Best_val, Quadrat = Ende.
    lr_band: Spanne (min–max) über die Identitäts-Pläne lr_band_plans als graues Band (je Ebene in lr_band_levels)."""
    T = _t(lang)
    run = run or dt.RUN_S3
    for lvl in levels:
        c = COL[lvl]
        if lr_band and lvl in lr_band_levels:
            grid, lo, hi = eta_band(run, lvl, K, rep, lr_band_plans, x, rmax)
            ax.fill_between(grid, lo, hi, color=COL["mute"], alpha=0.2, lw=0, zorder=1)
        for p, kind in _laeufe2(ident, sigmoid):
            v = dt.verlauf(run, lvl, p, K, rep, 0)
            xv, w = v[_xcol(x)].to_numpy(), v.winkel.to_numpy()
            ax.plot(xv, w, color=c, ls=LS2[kind], lw=1.1, zorder=3)
            eb = v.attrs["ep_best_val"]
            ax.plot(xv[eb], w[eb], "o", color=c, mec="white", mew=0.5, zorder=5)
            _endmark(ax, xv, w, c, x, rmax)
        if corner_label and ident:                             # rechtsbündig über dem Ende des Identitäts-Laufs
            ecke = float(np.degrees(np.arccos(dt.oracle_a(run, lvl, 0)[dt.VIEWS.index(dt.HOME[0])])))
            v = dt.verlauf(run, lvl, ident, K, rep, 0)
            xe = min(float(v[_xcol(x)].iloc[-1]), rmax) if x == "R" and rmax else float(v[_xcol(x)].iloc[-1])
            ax.annotate(T["corner"], (xe, ecke), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                        color=c, annotation_clip=False)
    ax.set_ylabel(T["angle2"])
    ax.set_ylim(0, None)
    if corner_label and ident:                                 # Platz für die Beschriftung über dem höchsten Ende
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    _xaxis2(ax, x, rmax, T)


def panel_ba2(ax, level="G", K=6, rep=0, ident="e0.003_const", sigmoid="sig10_e0.003_const", x="R", rmax=1.6,
              run=None, lang="en", sets=("train", "val"), sty=None):
    """Train-BA (dick) und Val-BA (dünn, heller) einer Ebene, Identität gestrichelt vs. Sigmoid durchgezogen; Train-BA
    aus den Snapshots (data.verlauf_sets, Kontrollen gegen den Log); Punkt = Best_val.
    sty: {'train': dict(lw, alpha), 'val': dict(lw, alpha[, ls])}; ls bei 'val' ersetzt den Linienstil des Laufs."""
    T = _t(lang)
    run = run or dt.RUN_S3
    c = COL[level]
    sty = sty or BA2
    for p, kind in _laeufe2(ident, sigmoid):
        v = dt.verlauf_sets(run, level, p, K, rep, ("train", "val"))
        xv = v["R"] if x == "R" else v["epoch"]
        for s in sets:
            ax.plot(xv, v[f"ba_{s}"], color=c, **{"ls": LS2[kind], **sty[s]})
        if "val" in sets:
            ax.plot(xv[v["eb"]], v["ba_val"][v["eb"]], "o", color=c, mec="white", mew=0.5, zorder=5)
    ax.set_ylabel(T["ba"])
    _xaxis2(ax, x, rmax, T)


def panel_kosten2(ax, level="G", K=6, rep=0, ident="e0.003_const", sigmoid="sig10_e0.003_const", x="R", rmax=1.6,
                  run=None, lang="en", color=None, kinds=("ident", "sig"), lw=1.0):
    """Flaches Kostenfeld: Σ μ_i (Train, ungequetscht wie panel_diskrepanz) für Identität und Sigmoid;
    color=None: Ebenenfarbe (eigene Zeile), sonst z. B. Tinte (twin-Achse im BA-Feld); kinds: welche Läufe."""
    T = _t(lang)
    run = run or dt.RUN_S3
    for p, kind in [(p, k) for p, k in _laeufe2(ident, sigmoid) if k in kinds]:
        v = dt.verlauf_sets(run, level, p, K, rep, ("train", "val"))
        ax.plot(v["R"] if x == "R" else v["epoch"], v["E_mu"], color=color or COL[level], ls=LS2[kind], lw=lw)
    ax.set_ylabel(T["cost2"])
    _xaxis2(ax, x, rmax, T)


def fig_pfad2(K=6, rep=0, dataset=3, rmax=1.25, levels=("G", "CW"), ident="e0.003_const", sigmoid="sig10_e0.003_const",
              layout="row", x="R", lr_band=False, lr_band_plans=PLANS, lr_band_levels=("CW",), band="none",
              kosten=True, kosten_mode="twin", kosten_levels=("identity", "sigmoid"), corner_label=True, legend="top",
              width_frac=1.0, height_cm=5.2, lang="en", titles=False, legend_ncol=None, runs=None,
              train_lw=None, val_lw=None, val_alpha=None, val_style=None, cost_lw=None, cost_color=None, look="paper",
              legend_style=None):
    """Abb. 3, vereinfachte Fassung (Auftrag ABB3; Voreinstellung = Paper-Kandidat nach Rückmeldung 23.10.):
    (a) Winkel zum Orakel, je Ebene Identität und Sigmoid; (b) G, (c) CW mit Train-/Val-BA (gemeinsame y-Achse);
    kosten=True mit kosten_mode 'twin': Σ μ auf rechter Achse in (b), (c) | 'row': Σ μ als flache Zeile (d), (e) darunter.
    kosten_levels: für welche Läufe Kosten gezeichnet werden ('identity', 'sigmoid').
    layout 'row' ((a) links, (b), (c) daneben) | '2x2' ((a) oben über volle Breite).
    band 'none' | 'je_ebene': Spanne der Best_val-Stellen der gezeichneten Läufe je Ebene (in (a) beide Ebenen in
    Ebenenfarbe; mit lr_band einschließlich der Band-Pläne). Punkte = Best_val, Quadrate = Ende bleiben immer.
    Legende: Linienstil = Identität/Sigmoid (zweigeteilt G | CW), Strichstärke = Train/Val; Ebene im Feldtitel.
    Linien (ABB4): train_lw, val_lw, val_alpha, val_style (None = Linienstil des Laufs, sonst z. B. ':' für alle
    Val-Linien – dann unterscheidet die Val-Linie Identität/Sigmoid nicht mehr), cost_lw, cost_color (None = Tinte bei
    twin, Ebenenfarbe bei row).
    ABB5: look 'paper' (Voreinstellung, Paper-Kandidat Abb. 3) | 'v1' (Darstellung bis ABB4). look setzt die
    Voreinstellungen von train_lw, val_lw, val_alpha, cost_lw, cost_color, legend_style (LOOK2); ausdrücklich
    übergebene Werte gelten vor. 'paper': Legende gruppiert (Farbe · Strichart · Strichstärke · Marker), Val-Linien
    hell und dünn hinter Train-BA, Kosten dunkelgrau und dünner, rechte Achse nur in der letzten Spalte beschriftet,
    Titel '(a) path to oracle'. legend_style 'grouped' | 'compact' (wie bis ABB4)."""
    T = _t(lang)
    run = _run2(dataset, runs)
    lk = LOOK2[look]
    train_lw = lk["train_lw"] if train_lw is None else train_lw
    val_lw = lk["val_lw"] if val_lw is None else val_lw
    val_alpha = lk["val_alpha"] if val_alpha is None else val_alpha
    cost_lw = lk["cost_lw"] if cost_lw is None else cost_lw
    cost_color = lk["cost_color"] if cost_color is None else cost_color
    legend_style = legend_style or lk["legend_style"]
    vsty = dict(lw=val_lw, alpha=val_alpha, **({"ls": val_style} if val_style else {}), **lk["val_z"])
    sty = dict(train=dict(lw=train_lw, alpha=BA2["train"]["alpha"], **lk["train_z"]), val=vsty)
    levels = tuple(levels)
    plans = [p for p, _ in _laeufe2(ident, sigmoid)]
    nb = len(levels)
    zeile = kosten and kosten_mode == "row"
    k_kinds = tuple({"identity": "ident", "sigmoid": "sig"}[k] for k in kosten_levels)
    fig = plt.figure(figsize=figsize_cm(width_frac, height_cm), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    if layout == "2x2":
        hr = (1, 1, 0.42) if zeile else (1, 1)
        gs = fig.add_gridspec(len(hr), nb, height_ratios=hr)
        ax_a = fig.add_subplot(gs[0, :])
        pos_b = [gs[1, j] for j in range(nb)]
        pos_k = [gs[2, j] for j in range(nb)] if zeile else []
    else:
        hr = (1, 0.38) if zeile else (1,)
        gs = fig.add_gridspec(len(hr), 1 + nb, height_ratios=hr)
        ax_a = fig.add_subplot(gs[:, 0])
        pos_b = [gs[0, 1 + j] for j in range(nb)]
        pos_k = [gs[1, 1 + j] for j in range(nb)] if zeile else []
    ax_b, ax_k = [], []
    for j in range(nb):
        a = fig.add_subplot(pos_b[j], sharey=ax_b[0] if ax_b else None)
        ax_b.append(a)
        if zeile:
            ax_k.append(fig.add_subplot(pos_k[j], sharex=a, sharey=ax_k[0] if ax_k else None))
        elif kosten:                                         # twin: Kosten rechts im BA-Feld
            k = a.twinx()
            if ax_k:
                k.sharey(ax_k[0])
            k.spines["right"].set_visible(True)
            ax_k.append(k)

    def span(lvl):
        ex = list(lr_band_plans) if lr_band and lvl in lr_band_levels else []
        return _bv_span2(run, lvl, K, rep, plans + ex, x)

    panel_winkel2(ax_a, levels, K, rep, ident, sigmoid, x, rmax, run, lang, lr_band, lr_band_plans, lr_band_levels,
                  corner_label)
    if band == "je_ebene":
        for lvl in levels:
            ax_a.axvspan(*span(lvl), color=COL[lvl], alpha=0.10, lw=0, zorder=0)
    if look == "paper" and titles is False:
        ax_a.set_title(f"(a) {T['p_paper']['title_a']}", loc="left")
    else:
        _letter(ax_a, 0, T["t_angle"], titles)
    for j, lvl in enumerate(levels):
        a = ax_b[j]
        panel_ba2(a, lvl, K, rep, ident, sigmoid, x, rmax, run, lang, sty=sty)
        if band == "je_ebene":
            for b in [a] + ([ax_k[j]] if zeile else []):
                b.axvspan(*span(lvl), color=COL[lvl], alpha=0.10, lw=0, zorder=0)
        if j:
            a.set_ylabel(""); a.tick_params(labelleft=False)
        if titles is not None:
            name = T["level"][lvl] if titles else T["level2"][lvl]
            a.set_title(f"({chr(98 + j)}) {name}", loc="left", color=COL[lvl])
        if kosten:
            k = ax_k[j]
            panel_kosten2(k, lvl, K, rep, ident, sigmoid, x, rmax, run, lang,
                          color=cost_color or (None if zeile else COL["ink"]), kinds=k_kinds, lw=cost_lw)
            if zeile:
                a.set_xlabel(""); a.tick_params(labelbottom=False)
                if j:
                    k.set_ylabel(""); k.tick_params(labelleft=False)
                if titles is not None:
                    k.set_title(f"({chr(98 + nb + j)})", loc="left", color=COL[lvl])
            else:
                k.set_xlabel("")
                if j < nb - 1:
                    k.set_ylabel(""); k.tick_params(labelright=False)
                elif look == "paper":
                    k.set_ylabel(T["p_paper"]["cost_axis"])
    if zeile:
        fig.align_ylabels([ax_b[0], ax_k[0]])
    if legend_style == "grouped" and legend != "none":
        _legend_pfad2_grouped(fig, levels, ident, sigmoid, lr_band, kosten and not zeile, k_kinds, sty, cost_color,
                              cost_lw, T, legend_ncol)
        return fig
    H, L = [], []                                            # Legende kompakt, nicht jede Kombination einzeln
    if ident:
        H.append(tuple(Line2D([], [], color=COL[l], ls=LS2["ident"], lw=1.1) for l in levels)); L.append(T["ident"])
    if sigmoid:
        H.append(tuple(Line2D([], [], color=COL[l], ls=LS2["sig"], lw=1.1) for l in levels))
        L.append(T["sig2"].format(b=dt.squash(sigmoid)[1]))
    if lr_band:
        H.append(Patch(color=COL["mute"], alpha=0.2, lw=0)); L.append(T["lr_band"])
    H += [Line2D([], [], color=COL["mute"], **sty["train"]), Line2D([], [], color=COL["mute"], **vsty)]
    L += [T["ba_set"]["train"], T["ba_set"]["val"]]
    if kosten and not zeile:                                  # nur Identität: Legendenstrich gestrichelt wie die Linie
        H.append(Line2D([], [], color=cost_color or COL["ink"], lw=cost_lw,
                        ls=LS2[k_kinds[0]] if len(k_kinds) == 1 else "-"))
        L.append(T["cost_leg"])
    H += [Line2D([], [], color=COL["mute"], ls="", marker="o", mec="white", mew=0.5),
          Line2D([], [], color=COL["mute"], ls="", marker="s", ms=3, mec="white", mew=0.4)]
    L += [T["best2"], T["end2"]]
    if legend == "inside":
        ax_b[0].legend(H, L, loc="lower left", handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    else:
        _legend(fig, [ax_a] + ax_b, H, L, legend, ncol=legend_ncol or (4 if len(L) > 6 else 3))
    return fig


def pfad2_rows(run, levels=("G", "CW"), K=6, rep=0, ident="e0.003_const", sigmoid="sig10_e0.003_const", dreh_win=50,
               plans=PLANS, band_r=1.2):
    """Zahlen zu fig_pfad2 je Lauf: Best_val (Epoche, R, Val-BA), Val-/Train-BA am Ende, Abfall Best_val → Ende,
    Winkel-Minimum (Wert, R), Winkel am Ende, Netto-Drehung am Ende = Winkel zwischen a(T − dreh_win) und a(T)
    (a = Gewichtsfaktor über alle 4 Views, normiert; CW: Klasse 0). Train-BA aus den Snapshots (Kontrollen).
    Je Ebene: Breite des η-Bandes (max − min des Winkels über die Identitäts-Pläne plans, Maximum über R ≤ band_r;
    jeder Plan zählt nur bis zu seinem R am Ende) und Best_val-R-Spanne über plans bzw. über die gezeichneten Läufe."""
    rows = []
    for lvl in levels:
        for p, _ in _laeufe2(ident, sigmoid):
            v = dt.verlauf(run, lvl, p, K, rep, 0)
            s = dt.verlauf_sets(run, lvl, p, K, rep, ("train", "val"))
            t = dt.traj(run, dt.variant_name(lvl, p), K, rep)
            eb, n = s["eb"], s["n"]
            im = int(v.winkel.to_numpy().argmin())
            a = np.sqrt(t[[f"a2_c0_{x}" for x in dt.VIEWS]].to_numpy(float))
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            dreh = float(np.degrees(np.arccos(np.clip(a[n] @ a[n - dreh_win], -1, 1))))
            lauf = f"{s['variant']} K{K} rep{rep}"
            vb, trb = s["ba_val"], s["ba_train"]
            rows += [dict(bild="pfad2", lauf=lauf, groesse="Best_val: Epoche, R, Val-BA",
                          wert=f"{eb}, {s['R'][eb]:.3f}, {vb[eb]:.3f}"),
                     dict(bild="pfad2", lauf=lauf, groesse="Ende (Epoche, R): Val-BA, Train-BA",
                          wert=f"({n}, {s['R'][n]:.3f}): {vb[n]:.3f}, {trb[n]:.3f}"),
                     dict(bild="pfad2", lauf=lauf, groesse="Abfall Best_val → Ende: Val-BA; Train-BA",
                          wert=f"{vb[n] - vb[eb]:+.3f} ({vb[eb]:.3f} → {vb[n]:.3f}); "
                               f"{trb[n] - trb[eb]:+.3f} ({trb[eb]:.3f} → {trb[n]:.3f})"),
                     dict(bild="pfad2", lauf=lauf, groesse="Winkel zum Orakel: Minimum (Grad) bei R; am Ende",
                          wert=f"{v.winkel[im]:.1f} bei R {v.R[im]:.3f}; {v.winkel.iloc[-1]:.1f}"),
                     dict(bild="pfad2", lauf=lauf, groesse=f"Netto-Drehung a(T−{dreh_win}) → a(T) (Grad)",
                          wert=f"{dreh:.2f}")]
    for lvl in levels:                                         # je Ebene, für die Bildunterschrift
        grid, lo, hi = eta_band(run, lvl, K, rep, plans, "R", band_r)
        w = hi - lo
        iw = int(np.nanargmax(w))
        a, b = _bv_span2(run, lvl, K, rep, plans)
        c, d = _bv_span2(run, lvl, K, rep, [p for p, _ in _laeufe2(ident, sigmoid)])
        lauf = f"{lvl} K{K} rep{rep}"
        rows += [dict(bild="pfad2", lauf=lauf, groesse=f"η-Band (Winkel, {len(plans)} Identitäts-Pläne): "
                                                        f"Breite max über R ≤ {band_r:g} (Grad) bei R",
                      wert=f"{w[iw]:.2f} bei R {grid[iw]:.3f}"),
                 dict(bild="pfad2", lauf=lauf, groesse=f"Best_val-R-Spanne über die {len(plans)} Identitäts-Pläne",
                      wert=f"{a:.3f} – {b:.3f}"),
                 dict(bild="pfad2", lauf=lauf, groesse="Best_val-R-Spanne über die gezeichneten Läufe (Identität, Sigmoid)",
                      wert=f"{c:.3f} – {d:.3f}")]
    return rows


# =========================================================================== #
# Auftrag ABB4 (23.10.2026): Zerlegung relativ zum Start (base='start'), Linienoptionen fig_pfad2, Start-BA
# Zusätzlich; bestehende Voreinstellungen unverändert.
# =========================================================================== #
TXT["en"].update(gain_start=r"BA $-$ BA(start)", sig_end_b=r"sigmoid $\beta{=}10$, end",
                 range_seeds="range over datasets")                # F05: "dataset" statt "seed" (29.10.)
TXT["de"].update(gain_start=r"BA $-$ BA(Start)", sig_end_b=r"Sigmoid $\beta{=}10$, Ende",
                 range_seeds="Spanne über die Seeds")
# ABB5: Beschriftungen Paper-Kandidat Abb. 2 (Rückmeldung Lukas 23.10.)
TXT["en"]["z_paper"] = dict(gain_id="+ weights (identity, end)", harm="identity end below frozen",
                            id_bv="identity, best val. epoch", oracle="oracle weights (fixed)",
                            sig=r"sigmoid $\beta$ = 10, end", range_total="range over datasets (identity, end)",
                            range_sig="range over datasets (sigmoid, end)", y=r"test BA $-$ start BA")
TXT["de"]["z_paper"] = dict(gain_id="+ Gewichte (Identität, Ende)", harm="Identität Ende unter frozen",
                            id_bv="Identität, Best_val", oracle="Orakel-Gewichte (fest)",
                            sig=r"Sigmoid $\beta$ = 10, Ende", range_total="Spanne über Datensätze (Identität, Ende)",
                            range_sig="Spanne über Datensätze (Sigmoid, Ende)", y=r"Test-BA $-$ Start-BA")


def panel_zerlegung_start(ax, level="G", seeds=(3, 4, 5), show=("id_end", "id_best_val", "sig_end", "oracle"),
                          ylim=None, lang="en", spanne_on="total", k_show=None, look="v1"):
    """Zerlegung über K relativ zum Start (data.kverlauf_start, Mittel über die Seeds): y = BA − Start, Achse ab 0.
    Gestapelt: + Prototypen (frozen − Start), darauf + Gewichte Identität Ende (id_end − frozen; schraffiert, wenn
    negativ). Best_val (Linie), Sigmoid β = 10 Ende (Raute), Orakel (gestrichelt) ebenfalls relativ zum Start.
    spanne_on 'total' (Fehlerstrich min–max über die Seeds an der Balkenspitze id_end − Start) | 'sig_end' (an der
    Raute) | 'none'. k_show: Liste der gezeigten K (z. B. jedes zweite); Balkenbreite folgt dem Abstand.
    look: 'v1' Beschriftungen wie ABB4 | 'paper' Beschriftungen nach Rückmeldung 23.10. (TXT[...]['z_paper'])."""
    T = _t(lang)
    Z = T["z_paper"] if look == "paper" else dict(
        gain_id=T["gain_id"], harm=T["harm"], id_bv=T["id_bv"], oracle=T["oracle"], sig=T["sig_end_b"],
        range_total=T["range_seeds"], range_sig=T["range_seeds"], y=T["gain_start"])
    d = dt.kverlauf_start(seeds)
    d = d[d.level == level].sort_values("K").reset_index(drop=True)
    if k_show is not None:
        d = d[d.K.isin(list(k_show))].reset_index(drop=True)
    K = d.K.to_numpy()
    step = float(np.min(np.diff(K))) if len(K) > 1 else 1.0
    w = 0.72 * step
    many = len(seeds) > 1
    c = COL[level]
    H, L = [], []
    P, g = d.r_frozen.to_numpy(), d.r_gain.to_numpy()
    ax.bar(K, P, width=w, color=COL["proto"], lw=0)
    H.append(Patch(color=COL["proto"])); L.append(T["proto"])
    if "id_end" in show:
        pos = g >= 0
        ax.bar(K[pos], g[pos], bottom=P[pos], width=w, color=c, alpha=0.35, lw=0)
        ax.bar(K[~pos], -g[~pos], bottom=(P + g)[~pos], width=w, color="white", edgecolor=COL["harm"],
               hatch="/////", lw=0.5)
        H += [Patch(color=c, alpha=0.35), Patch(facecolor="white", edgecolor=COL["harm"], hatch="/////")]
        L += [Z["gain_id"], Z["harm"]]
    if "id_best_val" in show:
        h, = ax.plot(K, d.r_id_best_val, color=c, lw=1.1)
        H.append(h); L.append(Z["id_bv"])
    if "oracle" in show:
        h, = ax.plot(K, d.r_orakel, color=COL["oracle"], ls=(0, (4, 2)), lw=0.8)
        H.append(h); L.append(Z["oracle"])
    xs = K + (0.22 * step if many and spanne_on == "total" else 0)
    if "sig_end" in show:
        h, = ax.plot(xs, d.r_sig_end, "D", color=c, ms=3, mec="white", mew=0.4, ls="", zorder=5)
        H.append(h); L.append(Z["sig"])
    if many and spanne_on in ("total", "sig_end"):
        col, xe, ec = (("r_id_end", K, COL["ink"]) if spanne_on == "total" else ("r_sig_end", xs, c))
        m = d[col].to_numpy()
        ax.errorbar(xe, m, yerr=[m - d[f"{col}_min"], d[f"{col}_max"] - m], fmt="none", ecolor=ec,
                    elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=6)
        H.append(Line2D([], [], color=ec, marker="|", ms=6, mew=0.6, ls=""))
        L.append(Z["range_total"] if spanne_on == "total" else Z["range_sig"])
    ax.set_ylabel(Z["y"])
    if ylim:                                                  # sonst Autoskalierung; Balken haften an 0 (Achse ab 0)
        ax.set_ylim(*ylim)
    ax.axhline(0, color=COL["mute"], lw=0.5)
    ax.set_xlim(K.min() - 0.6 * step, K.max() + 0.6 * step)
    ax.set_xticks(K if len(K) <= 10 else [k for k in K if k % 2 == 0])
    ax.set_xlabel(T["K"])
    return H, L


def start_rows(seeds=(3, 4, 5), levels=("G", "CW")):
    """Start-BA (Mittel über die Seeds) je Ebene für die Bildunterschrift: Bereich über K, Mittel über K, Werte je K."""
    d = dt.kverlauf_start(seeds)
    s = "–".join(map(str, (min(seeds), max(seeds)))) if len(seeds) > 1 else str(seeds[0])
    rows = []
    for lvl in levels:
        x = d[d.level == lvl].sort_values("K")
        i0, i1 = int(x.start.to_numpy().argmin()), int(x.start.to_numpy().argmax())
        lauf = f"{lvl}, Seeds {s}"
        rows += [dict(bild="zerlegung_start", lauf=lauf, groesse="Start-BA (Mittel über Seeds): min / max über K",
                      wert=f"{x.start.iloc[i0]:.3f} (K {x.K.iloc[i0]}) / {x.start.iloc[i1]:.3f} (K {x.K.iloc[i1]})"),
                 dict(bild="zerlegung_start", lauf=lauf, groesse="Start-BA (Mittel über Seeds), Mittel über K",
                      wert=f"{x.start.mean():.3f}"),
                 dict(bild="zerlegung_start", lauf=lauf, groesse="Start-BA (Mittel über Seeds) je K",
                      wert=", ".join(f"{k}: {v:.3f}" for k, v in zip(x.K, x.start)))]
    return rows


# =========================================================================== #
# Auftrag ABB5 (23.10.2026): Paper-Look für Abb. 2 (fig_zerlegung_k) und Abb. 3 (fig_pfad2), Zahlen
# Alte Darstellung bleibt über look='v1' erreichbar.
# =========================================================================== #
LOOK2 = {"paper": dict(train_lw=1.2, val_lw=0.5, val_alpha=0.35, cost_lw=0.8, cost_color="#555555",
                       legend_style="grouped", val_z=dict(zorder=1.5), train_z=dict(zorder=3)),
         "v1": dict(train_lw=BA2["train"]["lw"], val_lw=BA2["val"]["lw"], val_alpha=BA2["val"]["alpha"], cost_lw=1.0,
                    cost_color=None, legend_style="compact", val_z={}, train_z={})}
TXT["en"]["p_paper"] = dict(title_a="path to oracle", cost_leg="training cost (right axis)",
                            cost_axis=r"training cost $\Sigma_i\,\mu_i$" + "\n(lower = better)",
                            ident="identity", sig=r"sigmoid $\beta$ = {b:g}")
TXT["de"]["p_paper"] = dict(title_a="Pfad zum Orakel", cost_leg="Train-Kosten (rechte Achse)",
                            cost_axis=r"Train-Kosten $\Sigma_i\,\mu_i$" + "\n(kleiner = besser)",
                            ident="Identität", sig=r"Sigmoid $\beta$ = {b:g}")


def _legend_pfad2_grouped(fig, levels, ident, sigmoid, lr_band, cost_twin, k_kinds, sty, cost_color, cost_lw, T,
                          ncol=None):
    """Legende in Gruppen, je Gruppe eine Spalte: Farbe (Ebene) · Strichart neutral grau (Identität gestrichelt,
    Sigmoid durchgezogen) · Strichstärke (Train-BA dick, Val-BA dünn/hell als zweifarbige Symbole G | CW; Kosten
    grau, rechte Achse) · Marker (Best_val, Ende). Leere Plätze füllen die Spalten auf gleiche Höhe."""
    P = T["p_paper"]
    grey = COL["mute"]
    cols = [[(Patch(color=COL[l], lw=0), T["level2"][l]) for l in levels], []]
    if ident:
        cols[1].append((Line2D([], [], color=grey, ls=LS2["ident"], lw=1.1), P["ident"]))
    if sigmoid:
        cols[1].append((Line2D([], [], color=grey, ls=LS2["sig"], lw=1.1), P["sig"].format(b=dt.squash(sigmoid)[1])))
    if lr_band:
        cols[1].append((Patch(color=grey, alpha=0.2, lw=0), T["lr_band"]))
    tr = {k: v for k, v in sty["train"].items() if k != "zorder"}
    va = {k: v for k, v in sty["val"].items() if k not in ("zorder", "ls")}
    groesse = [(tuple(Line2D([], [], color=COL[l], ls="-", **tr) for l in levels), T["ba_set"]["train"]),
               (tuple(Line2D([], [], color=COL[l], ls="-", **va) for l in levels), T["ba_set"]["val"])]
    if cost_twin:
        groesse.append((Line2D([], [], color=cost_color or COL["ink"], lw=cost_lw,
                               ls=LS2[k_kinds[0]] if len(k_kinds) == 1 else "-"), P["cost_leg"]))
    cols.append(groesse)
    cols.append([(Line2D([], [], color=grey, ls="", marker="o", mec="white", mew=0.5), T["best2"]),
                 (Line2D([], [], color=grey, ls="", marker="s", ms=3, mec="white", mew=0.4), T["end2"])])
    cols = [c for c in cols if c]
    rows = max(len(c) for c in cols)
    blank = (Line2D([], [], alpha=0), "")
    H, L = [], []
    for c in cols:                                           # Spalten nacheinander (Legende füllt spaltenweise)
        for h, l in c + [blank] * (rows - len(c)):
            H.append(h); L.append(l)
    hm = {tuple: HandlerTuple(ndivide=None, pad=0)}
    r = fig.canvas.get_renderer()
    for cs in (1.2, 0.8, 0.5):                               # Spaltenabstand verkleinern, bis die Legende passt
        leg = fig.legend(H, L, loc="outside upper center", ncol=ncol or len(cols), handler_map=hm, columnspacing=cs)
        if leg.get_window_extent(r).width <= 0.98 * fig.bbox.width or cs == 0.5:
            return leg
        leg.remove()


def start_rows_abb5(seeds=(3, 4, 5), levels=("G", "CW")):
    """Start-BA für die Bildunterschrift Abb. 2: Mittel über alle Seeds und K, Spanne der Einzelwerte (Seed × K) und
    Spanne der Seed-Mittel über K; Prüfung, ob G und CW dieselbe Start-BA haben (gemeinsame Startmenge, D19)."""
    kv = dt.results(dt.SG9_EVAL, "kverlauf_s3-5.csv")
    kv = kv[kv.seed.isin(list(seeds))]
    s = "–".join(map(str, (min(seeds), max(seeds)))) if len(seeds) > 1 else str(seeds[0])
    p = kv.pivot_table(index=["seed", "K"], columns="level", values="start")
    gleich = bool(np.allclose(p[levels[0]], p[levels[1]])) if len(levels) == 2 else None
    rows = []
    for lvl in levels:
        x = kv[kv.level == lvl]
        m = x.groupby("K").start.mean()
        rows += [dict(bild="zerlegung_start", lauf=f"{lvl}, Seeds {s}",
                      groesse="Start-BA: Mittel über Seeds und K; Spanne Einzelwerte (Seed × K); Spanne Seed-Mittel über K",
                      wert=f"{x.start.mean():.3f}; {x.start.min():.3f} – {x.start.max():.3f}; {m.min():.3f} – {m.max():.3f}")]
    rows.append(dict(bild="zerlegung_start", lauf=f"G vs. CW, Seeds {s}", groesse="Start-BA je Seed und K gleich",
                     wert=str(gleich)))
    return rows


def pfad2_kompakt(K=6, rep=0, dataset=3, levels=("G", "CW"), ident="e0.003_const", sigmoid="sig10_e0.003_const",
                  plans=PLANS, band_r=1.2, runs=None):
    """Kompakte Tabelle zu Abb. 3 (Bildunterschrift): je Lauf Best_val (R, Val-BA), Winkel-Minimum (Wert, R) und Ende,
    Abfall Train-/Val-BA Best_val → Ende; je Ebene Breite des η-Bandes über R ≤ band_r. Gleiche Rechnung wie pfad2_rows."""
    run = _run2(dataset, runs)
    rows = []
    for lvl in levels:
        grid, lo, hi = eta_band(run, lvl, K, rep, plans, "R", band_r)
        for p, kind in _laeufe2(ident, sigmoid):
            v = dt.verlauf(run, lvl, p, K, rep, 0)
            s = dt.verlauf_sets(run, lvl, p, K, rep, ("train", "val"))
            eb, n = s["eb"], s["n"]
            im = int(v.winkel.to_numpy().argmin())
            rows.append(dict(ebene=lvl, lauf="identity" if kind == "ident" else "sigmoid",
                             best_val_R=round(float(s["R"][eb]), 3), best_val_valBA=round(float(s["ba_val"][eb]), 3),
                             winkel_min=round(float(v.winkel[im]), 1), winkel_min_R=round(float(v.R[im]), 3),
                             winkel_ende=round(float(v.winkel.iloc[-1]), 1),
                             abfall_trainBA=round(float(s["ba_train"][n] - s["ba_train"][eb]), 3),
                             abfall_valBA=round(float(s["ba_val"][n] - s["ba_val"][eb]), 3),
                             eta_band_max=round(float(np.nanmax(hi - lo)), 2)))
    return pd.DataFrame(rows)
