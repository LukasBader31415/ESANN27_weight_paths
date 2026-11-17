"""Versuchsplan TN · K-Sweep auf dem Heimat-View-Datensatz H0 (arbeitspakete.md, D16-D19).

Aufruf (lokal oder Hub):
    python experiments/TN_ksweep/run_tn.py --config experiments/TN_ksweep/config_full.yaml \
        --run-id 2026-10-XX_TN_ksweep_H0 [--jobs N]

Datenrollen (D18):
- Train  = die 600 Trainingspunkte von generate("H0", 2.5, .8, seed=3), Strom 2 (v1.2 unveraendert).
- Val    = neue Punkte, gleiche Zentren/sigma, eigener Strom (Config `val`), nur fuer die Epochenwahl Best_val.
- Test   = neue Punkte, eigener Strom (Config `test`), Berichtsmenge fuer Init/End/Best_val/Best_test und Orakel.
- Wiederholungen `rep` = Modell-Seeds base_seed + rep (alle auf derselben Trainingsmenge).
- Gemeinsame Init (D19): je (K, rep) EINE Startmenge (init_class_prototypes auf der Metrik mit gleichen
  Gewichten, view_tag "global"); alle Varianten starten von ihr. Die Init-Methode ist unveraendert.

Je Fit (variant, K, rep) entstehen im Cache `cache/toy_H0/<sig>/`:
- fits/<variant>/K=xx_rep=r.json   eine Ergebniszeile
- traj/<variant>/K=xx_rep=r.parquet Epochen-Log (A2), nachtraeglich aus den Snapshots berechnet
- hist/<variant>/K=xx_rep=r.npz     Snapshots (Prototypen, Gewichte) fuer spaetere Auswertungen ohne Neufit
Ein Neustart rechnet nur fehlende Fits. Am Ende: results/public/<run-id>/ (summary.md, fits.csv, orakel.csv, Abb.).

Versuchsplan SG: Config-Block sig_levels/sig_xis/sig_plans (Sigmoid f = expit(xi mu) in Kosten, Tauschsuche und
Gewichtsgradient); Pruefungslaeufe mit criteria_lock starten nur nach dem Kriterien-Commit (check_criteria_lock).
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import expit

from contextlib import contextmanager

from src.toy import toy_heimatview as hv
import src.models.M3GLVQ_global_decoupled as _mod_g
import src.models.M3GLVQ_label_coupled_WSOM as _mod_cw
from src.models.M3GLVQ_global_decoupled import M3GLVQ_Global, _normalize_l2
from src.models.M3GLVQ_label_coupled_WSOM import M3GLVQ_Label as M3GLVQ_CW
from src.common.proto_init import init_class_prototypes

ALL_VARIANTS = ["G_const", "G_decay", "CW_const", "CW_decay", "frozen", "frozen_oracle_G", "frozen_oracle_CW",
                "single_A", "single_B", "single_C"]
PENDANT = {"G_const": "frozen", "G_decay": "frozen", "CW_const": "frozen", "CW_decay": "frozen"}
ORACLE_PENDANT = {"G_const": "frozen_oracle_G", "G_decay": "frozen_oracle_G",
                  "CW_const": "frozen_oracle_CW", "CW_decay": "frozen_oracle_CW"}
INIT_TAG = "global"                    # Seed-Ableitung der gemeinsamen Startmenge (wie bisher bei G)
SCRIPT_VERSION = "1.2"                 # 1.1: gemeinsame Init, frozen_oracle_*; 1.2: Best_train, ohne Early Stop (D20)
# SG (19.10.): Sigmoid-Varianten (sig_*), Spalten ba_train/mu_sum/xi/ba_train_end/ba_train_best_val, Sperre fuer
# Pruefungslaeufe. SCRIPT_VERSION bewusst NICHT erhoeht: sie geht in die Cache-Signatur ein, die bestehenden
# TN/LR-Fits bleiben gueltig (phi_prime=None ist bitgleich, geprueft mit funktionstest_sg.py).


# --------------------------------------------------------------------------- #
# Konfiguration und Cache-Signatur
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Optionale Lernraten-Plaene (LR-Exploration): je Ebene und Plan eine Variante "<G|CW>_e<eta0>_<t<tau>|const>".
    # Die Basis-Parameter (eta, eta0, tau) und damit die Cache-Signatur bleiben wie in TN.
    cfg["lr_variants"] = {}
    for lvl in cfg.get("lr_levels", []):
        for p in cfg.get("lr_plans", []):
            name = f"{lvl}_e{p['eta0']:g}_" + (f"t{p['tau']:g}" if p.get("tau") else "const")
            cfg["lr_variants"][name] = dict(level=lvl, eta0=float(p["eta0"]), tau=p.get("tau"))
    # Optionale Sigmoid-Varianten (Versuchsplan SG): Quetschfunktion f(mu) = expit(xi * mu) in Kosten, Tauschsuche
    # und Gewichtsgradient (phi_prime). Name "<G|CW>_sig<xi>_e<eta0>_<t<tau>|const>". Basis-Parameter und
    # Cache-Signatur bleiben wie in TN (gleiche Startmengen).
    cfg["sig_variants"] = {}
    for xi in cfg.get("sig_xis", []):
        if xi is None:
            raise ValueError("sig_xis enthaelt einen Platzhalter (null): xi erst nach Lauf 1 festlegen (Versuchsplan SG)")
        for lvl in cfg.get("sig_levels", []):
            for p in cfg.get("sig_plans", []):
                name = f"{lvl}_sig{float(xi):g}_e{p['eta0']:g}_" + (f"t{p['tau']:g}" if p.get("tau") else "const")
                cfg["sig_variants"][name] = dict(level=lvl, eta0=float(p["eta0"]), tau=p.get("tau"), xi=float(xi))
    cfg["variants"] = list(cfg.get("variants", [])) + list(cfg["lr_variants"]) + list(cfg["sig_variants"])
    bad = [v for v in cfg["variants"]
           if v not in ALL_VARIANTS and v not in cfg["lr_variants"] and v not in cfg["sig_variants"]]
    if bad:
        raise ValueError(f"unbekannte Varianten {bad}; erlaubt: {ALL_VARIANTS}")
    if min(cfg["K"]) < 2:
        raise ValueError("K >= 2 (G bricht mit decoupled=True bei K=1 ab, F29)")
    return cfg


def cache_signature(cfg):
    """Hash ueber alles, was ein einzelnes Fit-Ergebnis bestimmt (nicht K/rep/Varianten/Run-ID)."""
    keys = ["plan", "s", "w_heim", "data_seed", "val", "test", "T", "eta", "eta0", "tau",
            "patience", "patience_decay", "eps_loss", "eps_weight", "base_seed"]
    payload = {k: cfg[k] for k in keys}
    payload["generator"] = hv.VERSION
    payload["script"] = SCRIPT_VERSION
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Daten
# --------------------------------------------------------------------------- #
def sample_like(data, stream, n_per_cluster):
    """Neue Punkte aus derselben Verteilung wie generate() (gleiche Zentren, gleiche sigma),
    eigener Zufallsstrom rng([seed, stream]). Bildet generate().sample() nach, ohne v1.2 zu aendern;
    die Nachbildung wird in build_data() an den Stroemen 2 (X_train) und 3 (X_test) bitgleich geprueft."""
    p = data["params"]
    centres, sigma = data["centres"], data["sigma"]
    R, n, D, NI = centres.shape[0], int(n_per_cluster), hv.DIMS, hv.N_INF
    rng = np.random.default_rng([p["seed"], stream])
    eps = rng.normal(size=(R, n, NI + 1, D))
    X = np.empty((R * n, (NI + 1) * D))
    for r in range(R):
        rows = slice(r * n, (r + 1) * n)
        for v in range(NI):
            X[rows, v * D:(v + 1) * D] = centres[r, v] + sigma[r, v] * eps[r, :, v]
        X[rows, NI * D:] = p["sigma_noise"] * eps[r, :, NI]
    region = np.repeat(np.arange(R), n)
    y = (region >= p["k_true"]).astype(int)
    return X, y, region


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def build_data(cfg):
    data = hv.generate(cfg["plan"], s=cfg["s"], w_heim=cfg["w_heim"], seed=cfg["data_seed"])
    # Nachbildung der Ziehung bitgleich? (sonst wuerden Val/Test nicht aus derselben Verteilung stammen)
    Xtr2, _, _ = sample_like(data, 2, data["params"]["n_per_cluster"])
    Xte3, _, _ = sample_like(data, 3, data["params"]["n_per_cluster"])
    if not (np.array_equal(Xtr2, data["X_train"]) and np.array_equal(Xte3, data["X_test"])):
        raise RuntimeError("sample_like bildet generate() nicht bitgleich nach - Abbruch")
    Xtr, ytr, reg = data["X_train"], data["y_train"], data["region_train"]
    V = len(hv.VIEWS)
    D_tr = hv.make_D_blocks(Xtr)                                   # je View / Train-Maximum
    scale = [squareform(pdist(hv._view(Xtr, v))).max() for v in range(V)]
    out = dict(data=data, M_tr=np.stack(D_tr), y_tr=ytr, region_tr=reg, scale=scale,
               hashes=dict(X_train=_sha(Xtr), X_test_alt=_sha(data["X_test"])))
    for role in ("val", "test"):
        X, y, r = sample_like(data, cfg[role]["stream"], cfg[role]["n_per_cluster"])
        out[f"M_{role}"] = np.stack([cdist(hv._view(X, v), hv._view(Xtr, v)) / scale[v] for v in range(V)])
        out[f"y_{role}"], out[f"X_{role}"], out[f"region_{role}"] = y, X, r
        out["hashes"][f"X_{role}"] = _sha(X)
    # W_TRUE: inverse Rauschvarianz wie oracle_fair (gepoolte q, alle 4 Views, alpha^2-Raum, ungerundet)
    q = hv.within_q(D_tr, reg)
    rcls = (np.arange(q.shape[0]) >= data["params"]["k_true"]).astype(int)
    out["W_TRUE"] = {c: hv._inv_weights(q[rcls == c].mean(0)) for c in (0, 1)}
    out["W_TRUE"]["glob"] = hv._inv_weights(q.mean(0))
    return out


def oracle_table(D):
    """Faires Orakel (oracle_fair, Medoide/q aus allen 600 Trainingspunkten) auf neuer Testmenge und alt."""
    data = D["data"]
    rows = []
    new = dict(data, X_test=D["X_test"], y_test=D["y_test"])
    for name, d in (("test_neu", new), ("x_test_alt", data)):
        o = hv.oracle_fair(d)
        rows.append(dict(menge=name, n=len(d["y_test"]),
                         **{k: o[k] for k in ("single_A", "single_B", "single_C", "single_N", "single_best",
                                              "equal", "global", "label", "proto")},
                         label_minus_global=o["label"] - o["global"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Modelle
# --------------------------------------------------------------------------- #
def variant_spec(name, cfg, W_TRUE):
    """W_TRUE: Orakel-Gewichte (alpha^2-Raum) aus Train, fuer frozen_oracle_* (v_init = sqrt)."""
    eta, eta0, tau = cfg["eta"], cfg["eta0"], cfg["tau"]
    decay = (lambda ep: eta0 * np.exp(-ep / tau))
    V = len(hv.VIEWS)
    base = dict(sched=None, decay=False, v_init=None, v_init_per_label=None, phi=None, phi_prime=None, xi=None)
    spec = {
        "G_const":   dict(base, cls=M3GLVQ_Global, eta=eta),
        "G_decay":   dict(base, cls=M3GLVQ_Global, eta=eta0, sched=decay, decay=True),
        "CW_const":  dict(base, cls=M3GLVQ_CW, eta=eta),
        "CW_decay":  dict(base, cls=M3GLVQ_CW, eta=eta0, sched=decay, decay=True),
        "frozen":    dict(base, cls=M3GLVQ_Global, eta=0.0),
        "frozen_oracle_G":  dict(base, cls=M3GLVQ_Global, eta=0.0, v_init=np.sqrt(W_TRUE["glob"])),
        "frozen_oracle_CW": dict(base, cls=M3GLVQ_CW, eta=0.0,
                                 v_init_per_label={c: np.sqrt(W_TRUE[c]) for c in (0, 1)}),
    }
    for v, view in enumerate(("A", "B", "C")):
        spec[f"single_{view}"] = dict(base, cls=M3GLVQ_Global, eta=0.0, v_init=np.eye(V)[v])
    for vname, p in cfg.get("lr_variants", {}).items():          # LR-Plaene: eigene eta0/tau je Variante
        e0, tau_p = p["eta0"], p["tau"]
        spec[vname] = dict(base, cls=M3GLVQ_CW if p["level"] == "CW" else M3GLVQ_Global, eta=e0,
                           sched=(lambda ep, e0=e0, tau_p=tau_p: e0 * np.exp(-ep / tau_p)) if tau_p else None,
                           decay=bool(tau_p))
    for vname, p in cfg.get("sig_variants", {}).items():         # Sigmoid: f = expit(xi mu), f' = xi f (1 - f)
        e0, tau_p, xi = p["eta0"], p["tau"], p["xi"]
        spec[vname] = dict(base, cls=M3GLVQ_CW if p["level"] == "CW" else M3GLVQ_Global, eta=e0,
                           sched=(lambda ep, e0=e0, tau_p=tau_p: e0 * np.exp(-ep / tau_p)) if tau_p else None,
                           decay=bool(tau_p), xi=xi,
                           phi=(lambda mus, xi=xi: expit(xi * mus)),
                           phi_prime=(lambda mus, xi=xi: xi * expit(xi * mus) * (1.0 - expit(xi * mus))))
    s = spec[name]
    lr = {**cfg.get("lr_variants", {}), **cfg.get("sig_variants", {})}
    s["eta0"] = s["eta"]
    s["tau"] = lr[name]["tau"] if name in lr else (tau if s["decay"] else None)
    s["level"] = "CW" if s["cls"] is M3GLVQ_CW else "G"
    s["patience"] = cfg["patience_decay"] if s["decay"] else cfg["patience"]
    return s


def common_init(D, K, rep, cfg):
    """Gemeinsame Startmenge je (K, rep): init_class_prototypes (unveraendert) auf der Metrik mit gleichen
    Gewichten (= _overall bei a_v = 1/sqrt(V)), Seed aus base_seed + rep, view_tag INIT_TAG."""
    M_tr, y = D["M_tr"], D["y_tr"]
    V = M_tr.shape[0]
    overall = np.tensordot(np.full(V, 1.0 / np.sqrt(V)) ** 2, M_tr, axes=(0, 0))
    return {lab: np.asarray(init_class_prototypes(overall, np.where(y == lab)[0], K,
                                                  base_seed=cfg["base_seed"] + rep, label=lab,
                                                  view_tag=INIT_TAG), int) for lab in (0, 1)}


@contextmanager
def preset_init(start):
    """Ersetzt waehrend des Fits die Init-Funktion in beiden Modellmodulen durch die gemeinsame Startmenge.
    Modellcode und Init-Methode bleiben unveraendert."""
    def _fixed(overall, inClass, k, *, base_seed=42, label=None, view_tag="global"):
        w = start[label]
        if len(w) != k or not np.isin(w, inClass).all():
            raise RuntimeError("gemeinsame Startmenge passt nicht zu Klasse/K")
        return w.copy()
    old = (_mod_g.init_class_prototypes, _mod_cw.init_class_prototypes)
    _mod_g.init_class_prototypes = _mod_cw.init_class_prototypes = _fixed
    try:
        yield
    finally:
        _mod_g.init_class_prototypes, _mod_cw.init_class_prototypes = old


def _a2_rows(v_snap, level):
    """Snapshot-Gewichte -> alpha^2 je Prototyp-Klasse, Form (2, V)."""
    a = np.asarray(v_snap, float)
    a2 = a ** 2
    return np.vstack([a2, a2]) if level == "G" else a2


def _predict(M, w, proto_lab, a2_rows):
    """Naechster Prototyp; Distanz zu Prototyp k in der Metrik seiner Klasse (G: dieselbe fuer alle).
    Gleichstand -> erster Prototyp (Klasse 0 zuerst), wie predict() der Modelle."""
    Dsel = M[:, :, w]                                              # (V, n, P)
    d = np.einsum("pv,vnp->np", a2_rows[proto_lab], Dsel)
    return proto_lab[np.argmin(d, axis=1)]


def _ba(y, pred):
    rec = [np.mean(pred[y == c] == c) for c in (0, 1)]
    return float(np.mean(rec))


def _assign(M, y, w, proto_lab, a2_rows):
    """d+, d- und Einzel-View-Anteile wie im Training (Prototyp-Metrik; bei G identisch)."""
    Dsel = M[:, :, w]
    d = np.einsum("pv,vnp->np", a2_rows[proto_lab], Dsel)
    own = proto_lab[None, :] == y[:, None]
    dp_all = np.where(own, d, np.inf)
    dm_all = np.where(~own, d, np.inf)
    cp, cm = np.argmin(dp_all, axis=1), np.argmin(dm_all, axis=1)
    rows = np.arange(len(y))
    dp, dm = d[rows, cp], d[rows, cm]
    return dp, dm, Dsel[:, rows, cp], Dsel[:, rows, cm], cm


def _grads(level, a_prev, dp, dm, dp_V, dm_V, y, cm, proto_lab, phi_prime=None):
    """Gewichtsgradient wie im Modell, je scope zerlegt in mu+- und mu--Anteil.
    G:  ein Gradient (Modell: _weights_update, Faktor 4), scopes 0/1 = Beitraege der Punkte je Klasse.
    CW: Gradient je Klassenvektor a^(c) (Modell: _weights_update_coupled), scopes 0/1 = a^(0), a^(1).
    phi_prime (Sigmoid-Varianten): jeder Punktbeitrag mit f'(mu_i) gewichtet, wie im Modell."""
    den = (dp + dm + 1e-5) ** 2
    w_mu = None if phi_prime is None else phi_prime((dp - dm) / (dp + dm + 1e-5))
    out = {}
    if level == "G":
        plus = 4.0 * a_prev[:, None] * dp_V * dm[None, :] / den[None, :]
        minus = -4.0 * a_prev[:, None] * dm_V * dp[None, :] / den[None, :]
        if w_mu is not None:
            plus, minus = plus * w_mu[None, :], minus * w_mu[None, :]
        for c in (0, 1):
            m = y == c
            out[c] = (plus[:, m].sum(1), minus[:, m].sum(1))
        out["tot"] = (plus.sum(1), minus.sum(1))
    else:
        cm_lab = proto_lab[cm]
        for c in (0, 1):
            a_c = a_prev[c]
            own, foreign = y == c, cm_lab == c
            if w_mu is None:
                p = (a_c[:, None] * dp_V[:, own] * dm[None, own] / den[None, own]).sum(1)
                mi = -(a_c[:, None] * dm_V[:, foreign] * dp[None, foreign] / den[None, foreign]).sum(1)
            else:
                p = (a_c[:, None] * dp_V[:, own] * dm[None, own] / den[None, own] * w_mu[None, own]).sum(1)
                mi = (-(a_c[:, None] * dm_V[:, foreign] * dp[None, foreign]) / den[None, foreign]
                      * w_mu[None, foreign]).sum(1)
            out[c] = (p, mi)
    return out


def _step(a, grad, eta):
    g_norm = np.linalg.norm(grad, 2) + 1e-12
    return _normalize_l2(a - eta * grad / g_norm)


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def run_fit(name, K, rep, cfg, D):
    s = variant_spec(name, cfg, D["W_TRUE"])
    clf = s["cls"](K={0: K, 1: K}, T=cfg["T"], eta=s["eta"], base_seed=cfg["base_seed"] + rep,
                   decoupled=True, patience=s["patience"], eps_loss=cfg["eps_loss"],
                   eps_weight=cfg["eps_weight"], eta_schedule=s["sched"], v_init=s["v_init"],
                   track_path=True, track_vweights=True, track_metrics=True,
                   phi=s["phi"], phi_prime=s["phi_prime"])
    if s["v_init_per_label"] is not None:
        clf.v_init_per_label = s["v_init_per_label"]
    M_tr, y_tr = D["M_tr"], D["y_tr"]
    start = common_init(D, K, rep, cfg)
    t0 = time.perf_counter()
    with preset_init(start):
        clf.fit(list(M_tr), y_tr)
    t_fit = time.perf_counter() - t0
    if not np.array_equal(np.sort(clf._w_history[0]), np.sort(np.concatenate([start[0], start[1]]))):
        raise RuntimeError("Fit ist nicht von der gemeinsamen Startmenge gestartet")

    t1 = time.perf_counter()
    level, proto_lab = s["level"], np.asarray(clf._y, int)
    W = [np.asarray(w, int) for w in clf._w_history]
    Vh = [np.asarray(v, float) for v in clf._v_history]
    loss = list(clf._log["loss"])
    n = len(W)
    ba_val = np.array([_ba(D["y_val"], _predict(D["M_val"], W[t], proto_lab, _a2_rows(Vh[t], level))) for t in range(n)])
    ba_test = np.array([_ba(D["y_test"], _predict(D["M_test"], W[t], proto_lab, _a2_rows(Vh[t], level))) for t in range(n)])
    ba_train = np.array([_ba(y_tr, _predict(M_tr, W[t], proto_lab, _a2_rows(Vh[t], level))) for t in range(n)])

    # Best_train = Zustand, den fit() wiederherstellt (erste Epoche mit kleinstem Train-Loss, strikt <).
    # Kontrolle: dieser Snapshot == von fit() wiederhergestellte Prototypen und Gewichte.
    e_btr = int(np.argmin(loss))
    v_restored = clf._vWeights if level == "G" else clf._vWeights_ls
    best_train_ok = bool(np.array_equal(np.asarray(clf._w, int), W[e_btr]) and np.array_equal(v_restored, Vh[e_btr]))

    # Kontrolle: eigene Vorhersage == model.predict am End-Snapshot
    if level == "G":
        clf._w, clf._vWeights = W[-1].copy(), Vh[-1].copy()
    else:
        clf._w, clf._vWeights_ls = W[-1].copy(), Vh[-1].copy()
    pred_model = np.asarray(clf.predict(list(D["M_test"])), int)
    pred_own = _predict(D["M_test"], W[-1], proto_lab, _a2_rows(Vh[-1], level))
    predict_ok = bool(np.array_equal(pred_model, pred_own))

    # Epochen-Log (A2): Zuordnung mit (w_t, alpha_{t-1}), Gradient, Selbstpruefung des Schritts
    WT = D["W_TRUE"]
    rows, prev_g = [], {}
    recon_max = 0.0
    for t in range(n):
        a2t = _a2_rows(Vh[t], level)
        r = dict(epoch=t, swap=bool(t > 0 and not np.array_equal(W[t], W[t - 1])),
                 loss=float(loss[t]) if t < len(loss) else np.nan,
                 ba_val=float(ba_val[t]), ba_test=float(ba_test[t]), ba_train=float(ba_train[t]))
        dp_t, dm_t = _assign(M_tr, y_tr, W[t], proto_lab, a2t)[:2]
        r["mu_sum"] = float(((dp_t - dm_t) / (dp_t + dm_t + 1e-5)).sum())   # Sigma mu ungequetscht (= loss bei f = Id)
        for c in (0, 1):
            for vi, vn in enumerate(hv.VIEWS):
                r[f"a2_c{c}_{vn}"] = float(a2t[c, vi])
            r[f"cos_wtrue_c{c}"] = _cos(a2t[c], WT[c])
        r["cos_wtrue_glob"] = _cos(a2t[0], WT["glob"]) if level == "G" else np.nan
        if t > 0:
            eta_t = float(s["sched"](t - 1)) if s["sched"] is not None else s["eta"]
            r["eta"] = eta_t
            r["step_len"] = float(np.linalg.norm(Vh[t] - Vh[t - 1]))
            a_prev = Vh[t - 1]
            dp, dm, dp_V, dm_V, cm = _assign(M_tr, y_tr, W[t], proto_lab, _a2_rows(a_prev, level))
            g = _grads(level, a_prev, dp, dm, dp_V, dm_V, y_tr, cm, proto_lab, s["phi_prime"])
            for sc, (gp, gm) in g.items():
                gt = gp + gm
                r[f"gnorm_{sc}"] = float(np.linalg.norm(gt))
                r[f"gplus_{sc}"] = float(np.linalg.norm(gp))
                r[f"gminus_{sc}"] = float(np.linalg.norm(gm))
                r[f"gcos_{sc}"] = _cos(gt, prev_g[sc]) if sc in prev_g else np.nan
                prev_g[sc] = gt
            if level == "G":
                recon = _step(a_prev, g["tot"][0] + g["tot"][1], eta_t)
            else:
                recon = np.vstack([_step(a_prev[c], g[c][0] + g[c][1], eta_t) for c in (0, 1)])
            err = float(np.max(np.abs(recon - Vh[t])))
            r["recon_err"] = err
            recon_max = max(recon_max, err)
        rows.append(r)
    traj = pd.DataFrame(rows)
    t_log = time.perf_counter() - t1

    e_end = n - 1
    e_bv = int(np.argmax(ba_val))
    e_bt = int(np.argmax(ba_test))
    a2_end = _a2_rows(Vh[-1], level)
    last = traj[traj.epoch > max(0, e_end - 100)]
    row = dict(variant=name, level=level, K=int(K), rep=int(rep), base_seed=cfg["base_seed"] + rep,
               ba_init=float(ba_test[0]), ba_end=float(ba_test[e_end]), ba_best_val=float(ba_test[e_bv]),
               ba_best_test=float(ba_test[e_bt]), ba_best_train=float(ba_test[e_btr]),
               val_init=float(ba_val[0]), val_end=float(ba_val[e_end]), val_best_val=float(ba_val[e_bv]),
               ep_end=e_end, ep_best_val=e_bv, ep_best_test=e_bt, ep_best_train=e_btr,
               end_ist_T=bool(e_end == cfg["T"]), best_train_ok=best_train_ok,
               n_swaps=int(traj.swap.sum()),
               n_swaps_last100=int(last.swap.sum()),
               gcos0_last100=float(last.get("gcos_0", pd.Series(dtype=float)).mean()),
               gcos1_last100=float(last.get("gcos_1", pd.Series(dtype=float)).mean()),
               step_last100=float(last.get("step_len", pd.Series(dtype=float)).mean()),
               cos_wtrue_c0_end=_cos(a2_end[0], WT[0]), cos_wtrue_c1_end=_cos(a2_end[1], WT[1]),
               recon_err_max=recon_max, predict_ok=predict_ok,
               init_protos=",".join(map(str, W[0])),
               t_fit_s=round(t_fit, 3), t_log_s=round(t_log, 3), t_per_epoch_s=round(t_fit / max(1, e_end), 5))
    for c in (0, 1):
        for vi, vn in enumerate(hv.VIEWS):
            row[f"a2_end_c{c}_{vn}"] = float(a2_end[c, vi])
    # Lernraten-Kenngroessen und Winkel der Gewichtsfaktoren a (mit Vorzeichen, Einheitskugel)
    etas = np.array([float(s["sched"](t)) if s["sched"] is not None else s["eta"] for t in range(e_end)])
    row.update(eta0=s["eta0"], tau=s["tau"] if s["tau"] else np.nan, R=float(etas.sum()),
               eta_T=float(etas[-1]) if len(etas) else np.nan)
    # Quetschfunktion und Train-BA (Versuchsplan SG); xi = NaN bei f = Identitaet
    row.update(xi=s["xi"] if s["xi"] is not None else np.nan,
               ba_train_end=float(ba_train[e_end]), ba_train_best_val=float(ba_train[e_bv]))
    A = np.asarray(Vh, float)
    A = A[:, None, :].repeat(2, axis=1) if A.ndim == 2 else A          # G: beide Klassen = derselbe Vektor
    for c in (0, 1):
        cos_step = np.clip((A[1:, c] * A[:-1, c]).sum(1), -1, 1)
        row[f"winkel_weg_c{c}"] = float(np.arccos(cos_step).sum())      # Summe der tatsaechlichen Drehwinkel
        row[f"winkel_netto_c{c}"] = float(np.arccos(np.clip(A[-1, c] @ A[0, c], -1, 1)))   # Start -> Ende
    hist = dict(w_history=np.asarray(W, np.int32), v_history=np.asarray(Vh, float), proto_lab=proto_lab)
    return row, traj, hist


# --------------------------------------------------------------------------- #
# Cache, Worker, Fortschritt
# --------------------------------------------------------------------------- #
def _paths(cache_dir, name, K, rep):
    stem = f"K={K:02d}_rep={rep}"
    return (cache_dir / "fits" / name / f"{stem}.json",
            cache_dir / "traj" / name / f"{stem}.parquet",
            cache_dir / "hist" / name / f"{stem}.npz")


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


_WORKER_DATA = {}


def _worker(name, K, rep, cfg, cache_dir):
    key = cache_signature(cfg)
    if key not in _WORKER_DATA:                       # Daten je Prozess einmal aufbauen (deterministisch)
        _WORKER_DATA[key] = build_data(cfg)
    row, traj, hist = run_fit(name, K, rep, cfg, _WORKER_DATA[key])
    pj, pt, ph = _paths(cache_dir, name, K, rep)
    _atomic_write(pt, lambda p: traj.to_parquet(p, index=False))
    _atomic_write(ph, lambda p: _savez(p, hist))
    _atomic_write(pj, lambda p: p.write_text(json.dumps(row), encoding="utf-8"))
    return row


def _savez(path, hist):
    with open(path, "wb") as f:
        np.savez_compressed(f, **hist)


def _eta_text(done_rows, todo, t_elapsed, n_done_new):
    """Grobe Restzeit: t(K) = a*K^b je Ebene aus den bisher gerechneten Fits, sonst Mittelwert."""
    if n_done_new == 0:
        return "?"
    df = pd.DataFrame(done_rows)
    rest = 0.0
    for name, K, _ in todo:
        sub = df[df.variant == name]
        if sub.K.nunique() >= 2:
            g = sub.groupby("K").t_fit_s.mean()
            b, loga = np.polyfit(np.log(g.index.values), np.log(g.values), 1)
            rest += float(np.exp(loga) * K ** b)
        elif len(sub):
            rest += float(sub.t_fit_s.mean())
        else:
            rest += float(df.t_fit_s.mean())
    return f"{rest / 60:.1f} min (Fit-Zeit, sequentiell)"


# --------------------------------------------------------------------------- #
# Auswertung -> results/public
# --------------------------------------------------------------------------- #
def headroom(fits, oracle_new, basis="end"):
    """Abb. (e), Test-BA, Mittel ueber reps (gepaart je rep):
    Init | + Prototypen = X(frozen) - Init | + Gewichte (realisiert) = X - X(frozen)
    | Referenz mit geschaetzten Gewichten = X(frozen_oracle_<Ebene>) - X(frozen) (Vergleichsgroesse, keine
    Obergrenze); Medoid-Referenz (oracle_fair) nur als Zusatzlinie."""
    col = "ba_end" if basis == "end" else "ba_best_val"
    medoid = {"G": float(oracle_new["global"]), "CW": float(oracle_new["label"])}
    out = []
    for name, fz in PENDANT.items():
        fo = ORACLE_PENDANT[name]
        if not {name, fz, fo} <= set(fits.variant):
            continue
        a = fits[fits.variant == name].set_index(["K", "rep"])
        b = fits[fits.variant == fz].set_index(["K", "rep"]).add_suffix("_fz")
        c = fits[fits.variant == fo].set_index(["K", "rep"]).add_suffix("_fo")
        j = a.join(b, how="inner").join(c, how="inner")
        lvl = a.level.iloc[0]
        for K, g in j.groupby(level="K"):
            gew = (g[col] - g[f"{col}_fz"]).mean()
            pot = (g[f"{col}_fo"] - g[f"{col}_fz"]).mean()
            out.append(dict(variant=name, level=lvl, basis=basis, K=int(K), n_reps=len(g),
                            init=g.ba_init.mean(), plus_prototypen=(g[f"{col}_fz"] - g.ba_init).mean(),
                            plus_gewichte=gew, ref_geschaetzt=pot,
                            anteil_an_ref=gew / pot if abs(pot) > 1e-9 else np.nan,
                            x=g[col].mean(), frozen=g[f"{col}_fz"].mean(), frozen_oracle=g[f"{col}_fo"].mean(),
                            medoid_ref=medoid[lvl],
                            init_ba_gleich_frozen=bool(np.allclose(g.ba_init, g.ba_init_fz))))
    return pd.DataFrame(out)


def plot_headroom(hr, path, n_starts=None):
    """Wasserfall je K: Init, + Prototypen, + Gewichte (realisiert); Referenz mit geschaetzten Gewichten als
    offener Balken ab X(frozen);
    schwarzer Strich = X; gepunktet = Medoid-Referenz. Negative Teile schmal und schraffiert."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(hr.variant))
    nb = hr.basis.nunique()
    fig, axes = plt.subplots(nb, len(names), figsize=(3.4 * len(names), 3.6 * nb), squeeze=False, sharey=True)
    cols = dict(init="#9e9e9e", plus_prototypen="#4c78a8", plus_gewichte="#f58518")
    labels = dict(init="Init", plus_prototypen="+ Prototypen", plus_gewichte="+ Gewichte (realisiert)")
    for i, basis in enumerate(hr.basis.unique()):
        for j, name in enumerate(names):
            ax = axes[i, j]
            first = (i, j) == (0, 0)
            g = hr[(hr.variant == name) & (hr.basis == basis)].sort_values("K")
            x = np.arange(len(g))
            cum = np.zeros(len(g))
            for z, part in zip((4, 3, 2), ("init", "plus_prototypen", "plus_gewichte")):
                v = g[part].values
                neg = v < 0
                ax.bar(x[~neg] - .12, v[~neg], bottom=cum[~neg], width=.55, color=cols[part], edgecolor="k",
                       lw=.3, zorder=z, label=labels[part] if first else None)
                if neg.any():
                    ax.bar(x[neg] - .12, -v[neg], bottom=(cum + v)[neg], width=.3, color=cols[part],
                           edgecolor="k", lw=.5, hatch="///", zorder=6)
                cum = cum + v
            ax.scatter(x - .12, cum, marker="_", s=700, color="k", zorder=7, label=f"{basis}(X)" if first else None)
            fz, pot = g.frozen.values, g.ref_geschaetzt.values
            ax.bar(x + .3, np.abs(pot), bottom=np.minimum(fz, fz + pot), width=.22, facecolor="none",
                   edgecolor="#2ca02c", lw=1.2, ls="--", zorder=5,
                   label="Referenz mit geschätzten Gewichten (frozen_oracle − frozen)" if first else None)
            ax.axhline(g.medoid_ref.iloc[0], color="k", ls=":", lw=.9, label="Medoid-Referenz" if first else None)
            ax.set_xticks(x, [f"K={k}" for k in g.K])
            ax.set_ylim(0.45, 0.9)
            ax.set_title(f"{name} · {basis}", fontsize=9)
    axes[0, 0].set_ylabel("Test-BA")
    fig.legend(loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.025))
    if n_starts:
        fig.text(0.5, 0.004, "Verschiedene Startmengen über 5 Seeds je K: "
                 + ", ".join(f"K={k}: {v}" for k, v in n_starts.items()), ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def b1_step3(fits, oracle_new, K=6):
    """B1 Schritt 3 (Festlegung 6, Basis Test-BA am Ende, Mittel ueber reps)."""
    f = fits[fits.K == K]
    if f.empty:
        return []
    m = f.groupby("variant").ba_end.mean()
    singles = [v for v in ("single_A", "single_B", "single_C") if v in m]
    best_single = max(singles, key=lambda v: m[v]) if singles else None
    out = []
    for plan in ("const", "decay"):
        g, c = f"G_{plan}", f"CW_{plan}"
        if g not in m or c not in m or best_single is None:
            continue
        ladder = m[best_single] < m[g] < m[c]
        near = m[c] >= oracle_new["label"] - 0.05
        out.append(dict(plan=plan, K=K, n_reps=int(f[f.variant == c].rep.nunique()),
                        best_single=best_single, ba_single=m[best_single], ba_G=m[g], ba_CW=m[c],
                        medoid_ref_CW=oracle_new["label"], schwelle=oracle_new["label"] - 0.05,
                        leiter_ok=bool(ladder), orakel_ok=bool(near), bestanden=bool(ladder and near)))
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "?"


def _versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for p in ("numpy", "proto-dist-ml", "scipy", "scikit-learn", "pandas", "pyyaml", "joblib"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "?"
    return out


def init_variability(cfg, D, fits=None):
    """Gemeinsame Init: Anzahl verschiedener Startmengen ueber die Seeds base_seed + rep (rep 0-4, nur
    Initialisierung), dazu Init-Sensitivitaet = Streuung der Init- und End-BA von frozen ueber die reps des Laufs."""
    reps = cfg.get("init_check_reps", [0, 1, 2, 3, 4])
    rows = []
    for K in cfg["K"]:
        sets = {tuple(sorted(np.concatenate(list(common_init(D, K, r, cfg).values())))) for r in reps}
        row = dict(K=K, n_seeds=len(reps), verschiedene_startmengen=len(sets))
        if fits is not None and "frozen" in set(fits.variant):
            f = fits[(fits.variant == "frozen") & (fits.K == K)]
            row.update(n_reps_lauf=len(f), init_sd=f.ba_init.std(), frozen_end_sd=f.ba_end.std())
        rows.append(row)
    return pd.DataFrame(rows)


def write_public(run_id, cfg, cfg_path, fits, oracle, init_var, t_total, jobs, sig):
    out = ROOT / "results" / "public" / run_id
    out.mkdir(parents=True, exist_ok=True)
    fits = fits.sort_values(["variant", "K", "rep"]).reset_index(drop=True)
    pub_cols = [c for c in fits.columns if c not in ("init_protos",)]
    fits[pub_cols].to_csv(out / "fits.csv", index=False, float_format="%.5f")
    oracle.to_csv(out / "orakel.csv", index=False, float_format="%.4f")
    onew = oracle[oracle.menge == "test_neu"].iloc[0]
    hr = pd.concat([headroom(fits, onew, "end"), headroom(fits, onew, "best_val")], ignore_index=True)
    hr.to_csv(out / "headroom.csv", index=False, float_format="%.4f")
    n_starts = dict(zip(init_var.K, init_var.verschiedene_startmengen))
    if not hr.empty:
        plot_headroom(hr, out / "abb_e_headroom.png", n_starts)
    b1 = b1_step3(fits, onew)

    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "experiments")
    v = _versions()
    L = []
    L.append(f"# {run_id} · {cfg.get('title', 'Versuchsplan TN (K-Sweep H0)')}\n")
    if cfg.get("status_note"):
        L.append(f"**{cfg['status_note']}**\n")
    if cfg.get("criteria_commit"):
        L.append(f"- Kriterien festgelegt vor dem Lauf: Commit `{cfg['criteria_commit']}` "
                 f"(`docs/arbeitspakete.md` „Kriterien SG (festgelegt vor Lauf 2)“, `{cfg.get('criteria_file', '')}`)")
    if cfg.get("summary_extra"):                                   # z. B. Datensatz-Kennwerte (Versuchsplan SG, Lauf 2)
        L.append(str(cfg["summary_extra"]).rstrip() + "\n")
    L.append(f"- Datum: {time.strftime('%Y-%m-%d %H:%M')} · Rechner: {socket.gethostname()} ({platform.system()}, {os.cpu_count()} Kerne), jobs = {jobs}")
    L.append(f"- Commit: `{head}`" + (" · **lokale Änderungen in src/experiments nicht committet**" if dirty else " · sauber"))
    L.append(f"- Skript v{SCRIPT_VERSION}, Config `{Path(cfg_path).as_posix()}`, Cache-Signatur `{sig}` (cache/toy_H0/{sig}/)")
    L.append("- Versionen: " + ", ".join(f"{k} {val}" for k, val in v.items()))
    L.append(f"- Laufzeit gesamt: {t_total / 60:.1f} min · Fits: {len(fits)} (gerechnet in diesem Lauf oder aus Cache)\n")
    L.append("## Konfiguration")
    L.append("```yaml\n" + yaml.safe_dump({k: x for k, x in cfg.items() if not k.startswith("_") and k != "summary_extra"},
                                         sort_keys=False, allow_unicode=True).strip() + "\n```")
    L.append(f"Daten-Hashes (sha256, 16 Zeichen): " + ", ".join(f"{k} `{h}`" for k, h in cfg.get('_hashes', {}).items()))
    L.append("Train = 600 Punkte (Strom 2, v1.2 bitgleich), Val/Test neu gezogen (eigene Ströme, gleiche Zentren/σ); "
             "Nachbildung der Ziehung an Strom 2 und 3 bitgleich geprüft. reps = Modell-Seeds (base_seed + rep); alle Varianten starten je (K, rep) von derselben Startmenge (D19).\n")
    L.append("## Medoid-Referenz (oracle_fair, Medoide und q aus allen 600 Trainingspunkten; keine Obergrenze)")
    L.append(_md(oracle))
    L.append("")
    L.append("## Kennzahlen (Test-BA, Mittel über reps)")
    agg = fits.groupby(["variant", "K"]).agg(n=("rep", "nunique"), init=("ba_init", "mean"), end=("ba_end", "mean"),
                                             best_val=("ba_best_val", "mean"), best_test=("ba_best_test", "mean"),
                                             ep_end=("ep_end", "mean"), ep_best_val=("ep_best_val", "mean"),
                                             t_fit_s=("t_fit_s", "mean")).reset_index()
    L.append(_md(agg))
    L.append("\n## B1 Schritt 3 (Test-BA am Ende, Mittel über reps: beste single < G < CW, CW ≥ Medoid-Referenz_CW − .05)")
    if b1:
        L.append(_md(pd.DataFrame(b1)))
        if any(r["n_reps"] < 5 for r in b1):
            L.append("Hinweis: weniger als 5 reps → kein Urteil; entschieden wird über den vollen Lauf (5 reps).")
    else:
        L.append("K = 6 nicht im Lauf.")
    L.append("\n## Abb. (e) Zerlegung (Test-BA; `headroom.csv`, `abb_e_headroom.png`): Init | + Prototypen = frozen − Init | + Gewichte (realisiert) = X − frozen | Referenz mit geschätzten Gewichten = frozen_oracle_<Ebene> − frozen (Vergleichsgröße, keine Obergrenze); "
             "Medoid-Referenz als Linie. Verschiedene Startmengen über 5 Seeds: " + ", ".join(f"K={k}: {v}" for k, v in n_starts.items()))
    L.append(_md(hr) if not hr.empty else "– (Varianten-Paare fehlen)")
    L.append("\n## Best_train (Zustand, den fit() wiederherstellt: Epoche mit kleinstem Train-Loss; Vergleich zur WSOM-Praxis)")
    bt = fits.groupby(["variant", "K"]).agg(end=("ba_end", "mean"), best_train=("ba_best_train", "mean"),
                                            best_val=("ba_best_val", "mean"), ep_best_train=("ep_best_train", "mean")).reset_index()
    bt["best_train_minus_end"] = bt.best_train - bt.end
    L.append(_md(bt))
    L.append(f"- Snapshot ep_best_train = von fit() wiederhergestellter Zustand: {int(fits.best_train_ok.sum())}/{len(fits)} Fits")
    L.append(f"- Alle Fits genau T = {cfg['T']} Epochen (kein Early Stop, D20): {int(fits.end_ist_T.sum())}/{len(fits)}")
    L.append("\n## Logging-Selbstprüfung (A2)")
    L.append(f"- max. |nachgerechneter Schritt − α(t)| über alle Fits und Epochen: {fits.recon_err_max.max():.2e} (Soll < 1e-10) → "
             f"{'bestanden' if fits.recon_err_max.max() < 1e-10 else 'NICHT bestanden'}")
    L.append(f"- eigene Vorhersage = model.predict am End-Snapshot: {int(fits.predict_ok.sum())}/{len(fits)} Fits")
    L.append(f"- Log-Zeit / Fit-Zeit: {fits.t_log_s.sum() / fits.t_fit_s.sum():.2f}")
    L.append("\n## Gemeinsame Init: verschiedene Startmengen über 5 Seeds, Init-Sensitivität = Streuung von frozen über die reps")
    L.append(_md(init_var))
    starts = fits.init_protos.map(lambda t: tuple(sorted(map(int, t.split(",")))))
    same = fits.assign(s=starts).groupby(["K", "rep"]).s.nunique()
    L.append(f"- Alle Varianten je (K, rep) mit derselben Startmenge: {int((same == 1).sum())}/{len(same)} (K, rep)-Kombinationen")
    L.append("\n## Laufzeit")
    L.append(_md(fits.groupby(["variant", "K"]).agg(t_fit_s=("t_fit_s", "mean"), ep=("ep_end", "mean"),
                                                   s_je_epoche=("t_per_epoch_s", "mean"), t_log_s=("t_log_s", "mean")).reset_index()))
    L.append("\nEpochen-Logs (Gradienten-Kosinus, Prototypen-Wechsel, Gewichtspfade) liegen im Cache, nicht hier.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"results/public/{run_id}: {size / 1e6:.2f} MB")
    return out


CRITERIA_HEADING = "Kriterien SG (festgelegt vor Lauf 2)"


def check_criteria_lock(cfg):
    """Sperre fuer Pruefungslaeufe (Versuchsplan SG, Lauf 2): Start nur, wenn `criteria_commit` auf einen Commit
    zeigt, der Vorfahre von HEAD ist und in dem `docs/arbeitspakete.md` den Abschnitt CRITERIA_HEADING mit
    „**Status: endgültig“ sowie `criteria_file` mit `status: endgueltig` und xi_bestaetigung == sig_xis enthaelt
    (Kriterien vor dem Lauf festgelegt)."""
    if not cfg.get("criteria_lock"):
        return
    commit, crit = cfg.get("criteria_commit"), cfg.get("criteria_file")
    if not commit or not crit:
        raise SystemExit("Sperre: criteria_commit und criteria_file in der Config setzen (Kriterien vor Lauf 2 committen)")
    def git_ok(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    if git_ok("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise SystemExit(f"Sperre: Kriterien-Commit {commit} unbekannt oder nicht Vorfahre von HEAD")
    ap = git_ok("show", f"{commit}:docs/arbeitspakete.md")
    head = f"\n### {CRITERIA_HEADING}\n"                      # die Ueberschrift, nicht Erwaehnungen im Text
    if ap.returncode != 0 or head not in ap.stdout:
        raise SystemExit(f"Sperre: Abschnitt „{CRITERIA_HEADING}“ fehlt in arbeitspakete.md im Commit {commit}")
    sec = ap.stdout.split(head, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
    if "**Status: endgültig" not in sec:
        raise SystemExit(f"Sperre: Abschnitt „{CRITERIA_HEADING}“ im Commit {commit} ist nicht als "
                         "„**Status: endgültig …**“ markiert")
    cf = git_ok("show", f"{commit}:{crit}")
    if cf.returncode != 0:
        raise SystemExit(f"Sperre: {crit} fehlt im Commit {commit}")
    crit_yaml = yaml.safe_load(cf.stdout) or {}
    if str(crit_yaml.get("status", "")).strip() != "endgueltig":
        raise SystemExit(f"Sperre: {crit} im Commit {commit} hat nicht status: endgueltig")
    xi_c, xi_cfg = crit_yaml.get("xi_bestaetigung"), [float(x) for x in cfg.get("sig_xis", [])]
    if xi_c is None or xi_cfg != [float(xi_c)]:
        raise SystemExit(f"Sperre: sig_xis {xi_cfg} passt nicht zu xi_bestaetigung {xi_c} in {crit}")
    print(f"Sperre ok: Kriterien aus Commit {commit} ({crit})", flush=True)


def _md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        return "```\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n```"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallele Fits (joblib/loky)")
    ap.add_argument("--skip-criteria-lock", action="store_true",
                    help="Sperre criteria_lock nicht pruefen (oeffentliches Repo: der Kriterien-Commit liegt nur in der "
                         "privaten Entwicklungshistorie; die Fits haengen davon nicht ab)")
    args = ap.parse_args()
    try:                                       # Log immer UTF-8 (Windows-Konsole/Umleitung sonst cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    cfg = load_config(args.config)
    if args.skip_criteria_lock:
        print("Sperre criteria_lock uebersprungen (--skip-criteria-lock)", flush=True)
    else:
        check_criteria_lock(cfg)
    sig = cache_signature(cfg)
    cache_dir = ROOT / "cache" / "toy_H0" / sig
    t0 = time.time()
    D = build_data(cfg)
    cfg["_hashes"] = D["hashes"]
    print(f"[{args.run_id}] Signatur {sig} | Daten ok (Nachbildung bitgleich) | Hashes {D['hashes']}", flush=True)

    grid = [(v, K, r) for K in cfg["K"] for v in cfg["variants"] for r in cfg["reps"]]
    done, todo = [], []
    for name, K, rep in grid:
        pj = _paths(cache_dir, name, K, rep)[0]
        if pj.exists():
            done.append(json.loads(pj.read_text(encoding="utf-8")))
        else:
            todo.append((name, K, rep))
    N = len(grid)
    print(f"Fits: {N} gesamt, {len(done)} aus Cache, {len(todo)} zu rechnen, jobs={args.jobs}", flush=True)

    rows = list(done)
    n_new = 0
    t_run = time.time()
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if args.jobs == 1:
        _WORKER_DATA[sig] = D
        results = (_worker(n, K, r, cfg_clean, cache_dir) for n, K, r in todo)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(_worker)(n, K, r, cfg_clean, cache_dir) for n, K, r in todo)
    remaining = list(todo)
    for row in results:
        n_new += 1
        rows.append(row)
        remaining = [x for x in remaining if not (x[0] == row["variant"] and x[1] == row["K"] and x[2] == row["rep"])]
        el = time.time() - t_run
        rest = _eta_text([r for r in rows], remaining, el, n_new)
        par = f", bei {args.jobs} jobs ~{float(rest.split()[0]) / args.jobs:.1f} min" if args.jobs > 1 and rest != "?" else ""
        print(f"[{len(rows):4d}/{N}] {row['variant']:9s} K={row['K']:2d} rep={row['rep']} | "
              f"init={row['ba_init']:.3f} end={row['ba_end']:.3f} best_val={row['ba_best_val']:.3f} | "
              f"{row['t_fit_s']:6.1f}s fit, {row['ep_end']:3d} ep | {el / 60:5.1f} min | Rest ~{rest}{par}", flush=True)

    fits = pd.DataFrame(rows)
    oracle = oracle_table(D)
    init_var = init_variability(cfg_clean, D, fits)
    write_public(args.run_id, cfg_clean | {"_hashes": D["hashes"]}, args.config, fits, oracle, init_var,
                 time.time() - t0, args.jobs, sig)
    print("fertig", flush=True)


if __name__ == "__main__":
    main()
