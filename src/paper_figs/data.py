"""Daten für die Paper-Abbildungen: Ergebnisse aus results/public/ und Epochen-Logs/Snapshots aus dem Cache.

Pfade relativ zu DATA_ROOT (Repo-Wurzel; Umgebungsvariable ESANN27_ROOT oder set_root()). Snapshots werden zuerst
unter cache/toy_H0/<Signatur>/ gesucht (so liegen sie auf dem Hub), lokal ersatzweise in den entpackten
Archiv-Kopien cache/hub/5dfd, cache/hub/sg, cache/hub/s4. Fehlt etwas, bricht das Laden mit Lauf, Datei und
gesuchten Pfaden ab.

Ein Lauf (Run) = Datensatz-Seed + Config; die Cache-Signatur kommt aus run_tn.cache_signature. Die Bild- und
Kennzahl-Funktionen nehmen `runs=[...]`; derzeit wird genau ein Lauf ausgewertet (mehrere: später, Auftrag ABB).
"""
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ESANN27_ROOT", ROOT))
SG9_EVAL = "2026-10-20_SG9_seeds3-17"
FALLBACK = {"5dfd9972f654650e": ["results/public/2026-10-20_fig2_example_run_ds3", "cache/hub/5dfd", "cache/hub/sg"],
            "e0df24cd0d4bcce0": ["cache/hub/s4"]}
# Öffentliches Repo: Epochen-Logs und Snapshots des Beispiellaufs (Datensatz 3, K 6, rep 0; G/CW, Identität und
# Sigmoid β 10, η .003) liegen unter results/public/2026-10-20_fig2_example_run_ds3/; nach einem vollen Neulauf
# (run_tn.py) wird zuerst cache/toy_H0/<Signatur>/ gelesen.
VIEWS = ["A", "B", "C", "N"]
HOME = {0: "A", 1: "B"}                     # Heimat-View je Klasse (H0)


def _tn():
    p = str(ROOT / "experiments" / "TN_ksweep")
    if p not in sys.path:
        sys.path.insert(0, p)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import run_tn
    return run_tn


def set_root(path):
    """Datenwurzel setzen (Repo-Wurzel mit results/ und cache/)."""
    global DATA_ROOT
    DATA_ROOT = Path(path)
    _traj.cache_clear(); _hist.cache_clear()
    return DATA_ROOT


def where():
    """'lokal' (Archiv-Kopien unter cache/hub/ vorhanden) oder 'Hub' (nur cache/toy_H0/)."""
    return "lokal" if (DATA_ROOT / "cache" / "hub").exists() else "Hub"


@dataclass(frozen=True)
class Run:
    """Ein Lauf: Datensatz-Seed und Config (relativ zur Repo-Wurzel)."""
    seed: int
    config: str
    name: str = ""

    @property
    def cfg(self):
        return _cfg(self.config)

    @property
    def sig(self):
        return _tn().cache_signature(self.cfg)


@lru_cache(maxsize=None)
def _cfg(config):
    return _tn().load_config(ROOT / config)


RUN_S3 = Run(3, "experiments/TN_ksweep/config_sig_explore.yaml", "Seed 3 · TN, LR_explore, SG_explore (5dfd…)")


def one_run(runs):
    """Derzeit genau ein Lauf; Liste mit mehreren Läufen ist für die spätere Absicherung vorgesehen."""
    runs = [RUN_S3] if runs is None else (list(runs) if isinstance(runs, (list, tuple)) else [runs])
    if len(runs) != 1:
        raise NotImplementedError("mehrere Läufe (runs=[…]) noch nicht umgesetzt – derzeit genau einen Lauf übergeben")
    return runs[0]


def variant_name(level, plan):
    """Plan 'e<eta0>_const' / 'e<eta0>_t<tau>' (+ optional 'sig<xi>_') -> Variantenname wie in run_tn."""
    if plan.startswith("sig"):
        return f"{level}_{plan}"
    if plan == "e0.003_const":
        return f"{level}_const"
    if plan == "e0.003_t133":
        return f"{level}_decay"
    return f"{level}_{plan}"


def _file(run, kind, variant, K, rep):
    ext = "parquet" if kind == "traj" else "npz"
    rel = Path(kind) / variant / f"K={K:02d}_rep={rep}.{ext}"
    cands = [DATA_ROOT / "cache" / "toy_H0" / run.sig / rel] + [DATA_ROOT / d / rel for d in FALLBACK.get(run.sig, [])]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"{kind} fehlt: Lauf Seed {run.seed} (Signatur {run.sig}), {variant} K {K} rep {rep}; "
                            f"gesucht: {[str(c) for c in cands]}")


@lru_cache(maxsize=256)
def _traj(run, variant, K, rep):
    t = pd.read_parquet(_file(run, "traj", variant, K, rep))
    t["R"] = t["eta"].fillna(0).cumsum()
    return t


@lru_cache(maxsize=64)
def _hist(run, variant, K, rep):
    h = np.load(_file(run, "hist", variant, K, rep))
    return {k: h[k] for k in h.files}


def traj(run, variant, K, rep):
    """Epochen-Log (Kopie) mit zusätzlicher Spalte R = Σ η_t (Weg)."""
    return _traj(run, variant, K, rep).copy()


def hist(run, variant, K, rep):
    return _hist(run, variant, K, rep)


@lru_cache(maxsize=8)
def train_data(run):
    """Trainingsmatrizen, Labels und Orakel-Gewichte W_TRUE (alpha²-Raum) aus dem Generator (run_tn.build_data)."""
    D = _tn().build_data(run.cfg)
    return dict(M_tr=D["M_tr"], y_tr=D["y_tr"], W_TRUE=D["W_TRUE"])


def oracle_a(run, level, cls=0):
    """Orakel-Gewichtsfaktor a = sqrt(W_TRUE) (Einheitsvektor): G global, CW je Klasse."""
    W = train_data(run)["W_TRUE"]
    w = np.asarray(W["glob"] if level == "G" else W[cls], float)
    return np.sqrt(w)


def results(run_id, name="fits.csv"):
    p = DATA_ROOT / "results" / "public" / run_id / name
    if not p.exists():
        raise FileNotFoundError(f"{name} fehlt für Lauf {run_id}: {p}")
    return pd.read_csv(p)


def kverlauf(seeds=(3,)):
    """K-Verlauf je Seed (kverlauf_s3-5.csv aus eval_sg9.py); bei mehreren Seeds Mittel, min, max je Ebene und K."""
    try:
        kv = results(SG9_EVAL, "kverlauf_s3-5.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e} – Auswertung SG9 (Commit da18bc7) im Repo?") from None
    seeds = list(seeds)
    miss = sorted(set(seeds) - set(kv.seed))
    if miss:
        raise ValueError(f"K-Verlauf nur für Seeds {sorted(kv.seed.unique())}, nicht {miss}")
    kv = kv[kv.seed.isin(seeds)]
    cols = [c for c in kv.columns if c not in ("seed", "level", "K")]
    g = kv.groupby(["level", "K"])[cols]
    return g.mean().join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()


def kverlauf_start(seeds=(3, 4, 5)):
    """Zerlegung relativ zum Start (ABB4): je Seed BA − Start für frozen, id_end, id_best_val, sig_end, orakel und der
    Gewinn der Gewichte id_end − frozen; dann Mittel, min, max über die Seeds je Ebene und K (Spannen je Seed
    gerechnet, nicht aus den Mitteln). Spalte start = Start-BA (absolut)."""
    try:
        kv = results(SG9_EVAL, "kverlauf_s3-5.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e} – Auswertung SG9 (Commit da18bc7) im Repo?") from None
    seeds = list(seeds)
    miss = sorted(set(seeds) - set(kv.seed))
    if miss:
        raise ValueError(f"K-Verlauf nur für Seeds {sorted(kv.seed.unique())}, nicht {miss}")
    kv = kv[kv.seed.isin(seeds)].copy()
    rel = {f"r_{c}": kv[c] - kv["start"] for c in ("frozen", "id_end", "id_best_val", "sig_end", "orakel")}
    kv = kv.assign(**rel, r_gain=kv["id_end"] - kv["frozen"])
    cols = ["start"] + list(rel) + ["r_gain"]
    g = kv.groupby(["level", "K"])[cols]
    return g.mean().join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()


def tabelle1():
    """Tabelle 1 (tabelle1.csv) und Werte je Seed (tabelle1_seeds.csv) aus eval_sg9.py."""
    try:
        return results(SG9_EVAL, "tabelle1.csv"), results(SG9_EVAL, "tabelle1_seeds.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e} – Auswertung SG9 (Commit da18bc7) im Repo?") from None


# --------------------------------------------------------------------------- #
# Abgeleitete Verläufe
# --------------------------------------------------------------------------- #
def verlauf(run, level, plan, K, rep, cls=0):
    """Verlauf je Epoche: epoch, R, a2_home (Heimat-View der Klasse cls), winkel (Grad zu den Orakel-Gewichten,
    alle 4 Views), loss, ba_val, ba_train (falls im Log), swap; attrs: ep_best_val, variant."""
    v = variant_name(level, plan)
    t = traj(run, v, K, rep)
    c = cls if level == "CW" else 0                      # G: beide Klassen = derselbe Vektor
    a = np.sqrt(t[[f"a2_c{c}_{x}" for x in VIEWS]].to_numpy(float))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    out = pd.DataFrame(dict(epoch=t.epoch, R=t.R, a2_home=t[f"a2_c{c}_{HOME[cls]}"], loss=t.loss, ba_val=t.ba_val,
                            ba_train=t["ba_train"] if "ba_train" in t else np.nan, swap=t.swap,
                            winkel=np.degrees(np.arccos(np.clip(a @ oracle_a(run, level, cls), -1, 1)))))
    out.attrs.update(ep_best_val=int(t.ba_val.to_numpy().argmax()), variant=v)
    return out


def mu_points(run, level, plan, K, rep):
    """μ_i je Epoche und Trainingspunkt aus den Snapshots (Zuordnung und Gewichte derselben Epoche, wie mu_sum im Log)."""
    tn = _tn()
    D = train_data(run)
    h = hist(run, variant_name(level, plan), K, rep)
    W, V, pl = h["w_history"], h["v_history"], np.asarray(h["proto_lab"], int)
    mu = np.empty((len(W), len(D["y_tr"])))
    for t in range(len(W)):
        dp, dm = tn._assign(D["M_tr"], D["y_tr"], W[t], pl, tn._a2_rows(V[t], level))[:2]
        mu[t] = (dp - dm) / (dp + dm + 1e-5)
    return mu


def kosten_ba(run, level, plan, K, rep):
    """Ein Lauf: Kosten (Log), Train-BA aus μ < 0, Val-BA; Kennzahlen zwischen Best_val und Ende."""
    t = traj(run, variant_name(level, plan), K, rep)
    D = train_data(run)
    mu = mu_points(run, level, plan, K, rep)
    y = D["y_tr"]
    tr_ba = np.array([np.mean([(m[y == k] < 0).mean() for k in (0, 1)]) for m in mu])
    eb, n = int(t.ba_val.to_numpy().argmax()), len(mu) - 1
    c0, c1 = mu[eb] < 0, mu[n] < 0
    out = dict(epoch=t.epoch.to_numpy(), R=t.R.to_numpy(), loss=t.loss.to_numpy(), E_mu=mu.sum(1), ba_train=tr_ba,
               ba_val=t.ba_val.to_numpy(), eb=eb, n=n,
               richtiger=int((c0 & c1 & (mu[n] - mu[eb] < 0)).sum()), kippt_rf=int((c0 & ~c1).sum()),
               kippt_fr=int((~c0 & c1).sum()), dE=float(t.loss.iloc[n] - t.loss.iloc[eb]),
               check_E_loss=float(np.nanmax(np.abs(mu.sum(1) - t.loss.to_numpy()))) if "_sig" not in plan else np.nan,
               check_ba_train=float(np.nanmax(np.abs(tr_ba - t.ba_train.to_numpy()))) if "ba_train" in t else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Auftrag ABB2: μ und BA je Epoche für Train/Val/Test aus den Snapshots (mit Kontrollen gegen den Log)
# --------------------------------------------------------------------------- #
TOL = 1e-10


@lru_cache(maxsize=8)
def full_data(run):
    """Alle Datenrollen aus run_tn.build_data (Train, Val Strom 10, Test Strom 11; Distanzen je View / Train-Maximum)."""
    return _tn().build_data(run.cfg)


def squash(plan):
    """Kostenfunktion f des Laufs: Identität oder expit(ν μ) bei Plänen 'sig<ν>_…'."""
    from scipy.special import expit
    if plan.startswith("sig"):
        xi = float(plan.split("_")[0][3:])
        return lambda m: expit(xi * m), xi
    return (lambda m: m), None


@lru_cache(maxsize=32)
def _verlauf_sets(run, level, plan, K, rep, sets):
    tn = _tn()
    D = full_data(run)
    v = variant_name(level, plan)
    t = traj(run, v, K, rep)
    h = hist(run, v, K, rep)
    W, V, pl = h["w_history"], h["v_history"], np.asarray(h["proto_lab"], int)
    key = {"train": ("M_tr", "y_tr"), "val": ("M_val", "y_val"), "test": ("M_test", "y_test")}
    need = tuple(sorted(set(sets) | {"train", "val"}))                # train/val immer (Kontrollen)
    out = dict(epoch=t.epoch.to_numpy(), R=t.R.to_numpy(), loss=t.loss.to_numpy(), eb=int(t.ba_val.to_numpy().argmax()),
               n=len(W) - 1, variant=v, plan=plan)
    for s in need:
        M, y = D[key[s][0]], D[key[s][1]]
        mu = np.empty((len(W), len(y)))
        ba = np.empty(len(W))
        for e in range(len(W)):
            a2 = tn._a2_rows(V[e], level)
            dp, dm = tn._assign(M, y, W[e], pl, a2)[:2]
            mu[e] = (dp - dm) / (dp + dm + 1e-5)
            ba[e] = tn._ba(y, tn._predict(M, W[e], pl, a2))          # wie im Log (naechster Prototyp)
        out[f"mu_{s}"], out[f"ba_{s}"] = mu, ba
    f, xi = squash(plan)
    out["E_mu"] = out["mu_train"].sum(1)                               # Σ μ (ungequetscht)
    chk = dict(val=float(np.max(np.abs(out["ba_val"] - t.ba_val.to_numpy()))),
               loss=float(np.max(np.abs(f(out["mu_train"]).sum(1) - t.loss.to_numpy()))))
    if "ba_train" in t:
        chk["train"] = float(np.max(np.abs(out["ba_train"] - t.ba_train.to_numpy())))
    if "ba_test" in t and "ba_test" in out:
        chk["test"] = float(np.max(np.abs(out["ba_test"] - t.ba_test.to_numpy())))
    bad = {k: x for k, x in chk.items() if not x <= TOL}
    if bad:
        raise RuntimeError(f"Kontrolle gegen den Log fehlgeschlagen ({v} K{K} rep{rep}): {bad} – abgebrochen")
    out["check"] = chk
    return out


def verlauf_sets(run, level, plan, K, rep, sets=("train", "val")):
    """μ je Punkt und BA je Epoche für die gewünschten Mengen (train/val/test), aus den Snapshots nachgerechnet.
    Kontrollen (Abbruch bei Abweichung > 1e-10): Val-BA = ba_val im Log, Σ f(μ_train) = loss im Log; falls im Log
    vorhanden auch Train-BA und Test-BA. Ergebnis enthält 'check' mit den maximalen Abweichungen."""
    return _verlauf_sets(run, level, plan, K, rep, tuple(sorted(sets)))


def grenzwechsel(mu):
    """Je Epoche t ≥ 1: Anzahl falsch → richtig und richtig → falsch (richtig = μ < 0), Länge n_epochs (Epoche 0 = 0)."""
    r = mu < 0
    fr = np.r_[0, (~r[:-1] & r[1:]).sum(1)]
    rf = np.r_[0, (r[:-1] & ~r[1:]).sum(1)]
    return fr, rf


def kostenbeitrag(mu, f, margin=0.1):
    """Änderung der Kosten Σ f(μ) je Epoche, aufgeteilt nach Lage des Punkts in der Epoche davor:
    sicher richtig (μ < −margin), Grenzbereich (|μ| ≤ margin), falsch (μ > margin). Kumuliert ab Epoche 0."""
    d = f(mu[1:]) - f(mu[:-1])
    prev = mu[:-1]
    grp = dict(sicher=prev < -margin, grenze=np.abs(prev) <= margin, falsch=prev > margin)
    out = {k: np.r_[0.0, np.cumsum((d * g).sum(1))] for k, g in grp.items()}
    out["summe"] = np.r_[0.0, np.cumsum(d.sum(1))]
    return out
