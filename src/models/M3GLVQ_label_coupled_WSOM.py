__author__ = 'Lukas Bader, Dietlind Zühlke'
__copyright__ = ('Copyright 2019, Benjamin Paaßen; '
                 'Copyright 2026, Lukas Bader, Dietlind Zühlke')
__license__ = 'GPLv3'
__version__ = '2.1.0-label'
__maintainer__ = 'Lukas Bader'


"""
M3GLVQ -- LABEL-SPECIFIC variant (one squared-weight vector per label)

This module implements the M3GLVQ algorithm as described in:

    Lukas Bader, Ina Terwey-Scheulen, Dietlind Zühlke,
    "Implementation of Multi-Matrix Median Generalized Learning Vector Quantization",
    ESANN 2026.

The implementation is based on and extends the original MGLVQ code by
Benjamin Paaßen (proto-dist-ml, GNU GPLv3).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from src.common.proto_init import init_class_prototypes
_ERR_CUTOFF = 1e-5

def _project_simplex(v):
    """Project vector v onto the probability simplex {w >= 0, sum(w)=1}. (Duchi et al., 2008).

    Kept for backward compatibility / non-global paths. The global path now uses
    the squared-weight parametrization and _normalize_l2 instead.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector.")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def _normalize_l2(a):
    """Normalize vector a onto the unit L2-sphere {sum(a_v^2) = 1}.

    Squared-weight parametrization: the distance uses a_v^2 as group weights, so
    sum(a_v^2)=1  <=>  ||a||_2 = 1. Non-negativity of the effective weights a_v^2
    is automatic regardless of the sign of a_v, hence no simplex projection is
    needed in the global path.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("L2 normalization expects a 1D vector.")
    nrm = np.linalg.norm(a, 2)
    if nrm < 1e-12:
        # degenerate input -> fall back to uniform a_v (so a_v^2 = 1/V each)
        return np.full(a.size, 1.0 / np.sqrt(a.size))
    return a / nrm


class M3GLVQ_Label(BaseEstimator, ClassifierMixin):
    """
    VMGLVQ: Median Generalized LVQ for multiple distance matrices (DL = [D^(v)]).
    - Combines distances: D* = Σ_v a_v^2 D^(v) with learnable factors a_v.
      Squared-weight parametrization (global path): the effective group weight is
      a_v^2, which is non-negative by construction, so NO simplex projection is
      needed. The only constraint is Σ_v a_v^2 = 1, i.e. ||a||_2 = 1, enforced by
      simple L2 normalization after each gradient step.
    - get_vweight_path() / final weights expose BOTH a_v and a_v^2.
    - Vectorized implementation (no O(V·m²) double loops).
    - Supports K>1 as well as K==1 (multi-class/binary) with specialized paths.
    - Optional tracking: prototype path, weights, loss history.

    Parameters
    ----------
    K : int | dict[label -> int] | sequence[int]
        Number of prototypes per label (int = uniform for all labels).
    T : int
        Epochs.
    phi : callable or None
        Squashing function applied to the GLVQ µ-values (default: identity).
    phi_prime : callable or None
        Derivative of phi, used to weight each point in the weight gradient
        (default None: phi' = 1, i.e. identity squashing).
    track_path : bool
    track_vweights : bool
    track_metrics : bool
    eta : float
        Learning rate for matrix weights.
    v_init : array-like or None
        Initial factors a_v of the V matrices; L2-normalized so that Σ a_v^2 = 1.
    """

    def __init__(
        self,
        K,
        T=50,
        phi=None,
        *,
        track_path=False,
        track_vweights=True,
        track_metrics=False,
        eta=1.0,
        v_init=None,
        base_seed=42,
        decoupled=False,
        patience=15,
        eps_loss=1e-4,
        eps_weight=1e-3,
        require_weight_settle=False,
        eta_schedule=None,
        phi_prime=None,
    ):
        self.K = K
        self.T = T
        self.phi = (lambda mus: mus) if phi is None else phi
        # optional derivative phi'(mu) of the squashing function; None = identity (phi' = 1,
        # weight step bit-identical to the version without this option)
        self.phi_prime = phi_prime

        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)
        self.eta = float(eta)
        self.v_init = None if v_init is None else np.asarray(v_init, dtype=float)
        self.base_seed = int(base_seed)

        # ---- decoupled training (opt-in; default False = old behavior) ----
        # decoupled=True runs the per-label weight update EVERY epoch (not only
        # on an accepted swap); prototype early stop -> oscillation-immune best-
        # loss-plateau + weight-settle patience. One snapshot per epoch.
        self.decoupled = bool(decoupled)
        self.patience = int(patience)
        self.eps_loss = float(eps_loss)        # relative: plateau threshold = eps_loss*|best_loss|
        self.eps_weight = float(eps_weight)    # NET ||a_ls(t) - a_ls(t-patience)||_F over the window
        self.require_weight_settle = bool(require_weight_settle)
        self.eta_schedule = eta_schedule       # optional callable ep -> eta
        # NOTE: with a CONSTANT eta the normalized-gradient step has a fixed
        # step-size floor ~eta, so the per-label weights orbit (do not settle to a
        # point). Hence the PRIMARY stop is the oscillation-immune best-loss
        # plateau; require_weight_settle adds a NET-displacement gate that only
        # becomes satisfiable under an eta decay.

        self._w_history = None
        self._v_history = None
        self._log = None
        self._snapshot_idx = 0

        self.final_matrix_ = None  # Set after fit()

    # ---------- Tracking ----------
    def _init_tracking(self):
        self._snapshot_idx = 0
        self._w_history = [] if self.track_path else None
        self._v_history = [] if self.track_vweights else None
        self._log = {"loss": []} if self.track_metrics else None

    def _snapshot(self, *, w=None, v=None, loss=None):
        if self.track_path and w is not None:
            self._w_history.append(np.array(w, copy=True))
        if self.track_vweights and v is not None:
            self._v_history.append(np.array(v, dtype=float, copy=True))
        if self.track_metrics and loss is not None:
            self._log["loss"].append(float(loss))
        self._snapshot_idx += 1

    def get_final_matrix(self):
        if self.final_matrix_ is None:
            raise RuntimeError("Model has not been trained yet (fit has not been called).")
        return self.final_matrix_

    # ---------- Utilities ----------
    def _resolve_K_per_label(self, unique_labels):
        """
        self.K may be:
          - int → same number per label
          - dict {label: k_l}
          - sequence[int] in the order of unique_labels
        Returns: K_per (L,) as int array.
        """
        L = len(unique_labels)
        if isinstance(self.K, (int, np.integer)):
            if self.K < 1:
                raise ValueError("K must be >= 1")
            return np.full(L, int(self.K), dtype=int)

        if isinstance(self.K, dict):
            K_per = np.empty(L, dtype=int)
            for i, lab in enumerate(unique_labels):
                if lab not in self.K:
                    raise ValueError(f"K missing for label {lab!r}")
                K_per[i] = int(self.K[lab])
            if np.any(K_per < 1):
                raise ValueError("All K_l must be >= 1")
            return K_per

        try:
            K_seq = np.asarray(self.K, dtype=int)
            if K_seq.shape != (L,):
                raise ValueError(f"K must have length {L}; got shape {K_seq.shape}")
            if np.any(K_seq < 1):
                raise ValueError("All K_l must be >= 1")
            return K_seq
        except Exception as e:
            raise ValueError("Unsupported K format. Use int, dict {label:k}, or sequence length=L") from e

    def _stack_DL(self, DL):
        """DL → M with shape (V, m, m) and set base dimensions; respect v_init."""
        M = np.stack(DL, axis=0)
        self._V = M.shape[0]
        self._m = M.shape[1]

        if self.v_init is not None:
            if self.v_init.shape != (self._V,):
                raise ValueError(f"v_init must have shape ({self._V},), got {self.v_init.shape}")
            # squared-weight parametrization: L2-normalize so that Σ a_v^2 = 1
            self._vWeights = _normalize_l2(self.v_init)
        elif not hasattr(self, "_vWeights") or self._vWeights is None or self._vWeights.size != self._V:
            # uniform start: a_v = 1/sqrt(V)  ->  a_v^2 = 1/V,  Σ a_v^2 = 1
            self._vWeights = np.full(self._V, 1.0 / np.sqrt(self._V))

        if not hasattr(self, "_v_m") or self._v_m is None or self._v_m.size != self._V:
            self._v_m = np.zeros(self._V, dtype=float)

        return M

    # ===================================================================
    # ===============  LABEL-SPECIFIC WEIGHTING  ========================
    # ===================================================================
    # Each label l owns its own factor vector a^(l) (length V, ||a^(l)||_2 = 1).
    # The combined distance for label l is  D*^(l) = Σ_v (a^(l)_v)^2 D^(v).
    # Coupled scheme (Paper Eq. 6, 11-14): every distance to a prototype is
    # measured in the metric of that PROTOTYPE's label, i.e. d_i^+ uses a^(y_i)
    # and d_i^- uses the FOREIGN label's a^(1-y_i) (binary only). Hence a^(l)
    # collects mu^+ terms of points with y_i == l and mu^- terms of foreign
    # points whose nearest wrong prototype has label l; predict uses the same rule.

    def _overall_for_label(self, M, l):
        """Squared-weighted distance matrix for label index l: Σ_v a^(l)_v^2 D^(v)."""
        return np.tensordot(self._vWeights_ls[l] ** 2, M, axes=(0, 0))

    def _weights_update_label(self, l, dp, dm, dp_V, dm_V, mask):
        """
        Gradient step for the factor vector a^(l) of a single label l.

        Only points with y_i == l contribute (mask selects them). The per-point
        contribution is the same squared-parametrization gradient as in the
        global path:

            dCF/da^(l)_v = Σ_{i: y_i=l} 4 a^(l)_v (d_iv^+ d_i^- - d_iv^- d_i^+)
                                                  / (d_i^+ + d_i^-)^2

        dp, dm     : (m,)    combined d^+ , d^-      (only mask entries used)
        dp_V, dm_V : (V, m)  per-matrix d_iv^+, d_iv^-
        mask       : (m,) bool, True where y_i == l
        """
        if not np.any(mask):
            return
        a_l = self._vWeights_ls[l]
        den = (dp[mask] + dm[mask] + 1e-5) ** 2                  # (n_l,)
        num = dp_V[:, mask] * dm[mask][None, :] \
              - dm_V[:, mask] * dp[mask][None, :]                # (V, n_l)
        if self.phi_prime is not None:                           # phi'(mu_i) per point
            num = num * self.phi_prime((dp[mask] - dm[mask]) / (dp[mask] + dm[mask] + 1e-5))[None, :]
        grad =(4.0 * a_l[:, None] * num / den[None, :]).sum(axis=1)   # (V,)

        g_norm = np.linalg.norm(grad, 2) + 1e-12
        step = grad / g_norm
        a_new = a_l - self.eta * step
        self._vWeights_ls[l] = _normalize_l2(a_new)              # Σ a^(l)_v^2 = 1

    def _weights_update_coupled(self, dp, dm, dp_V, dm_V, y, closest_minus,
                                proto_lab_idx):
        """COUPLED per-class weight update (Paper Eq. 11-14).

        a^(c) receives mu^+ contributions from points of class c (c is the
        CORRECT class, via d^+, Eq. 13) AND mu^- contributions from foreign
        points whose nearest WRONG prototype belongs to class c (c is the
        wrong class for them, via d^-, Eq. 14). This is the class-coupling:
        a^(c) is shaped by BOTH classes.

            mu^(+,v)_i = a^(c)_v d^+_{i,v} d^-_i / (d^+_i + d^-_i + eps)^2
            mu^(-,v)_i = - a^(c)_v d^-_{i,v} d^+_i / (d^+_i + d^-_i + eps)^2

        The L1-normalized, eta-scaled gradient step matches the global path.
        """
        L = self._vWeights_ls.shape[0]
        den = (dp + dm + 1e-5) ** 2                              # (m,)
        cm_lab = proto_lab_idx[closest_minus]                    # class of nearest wrong proto per point
        # phi'(mu_i) per point; None = identity (phi' = 1, unchanged code path)
        w_mu = None if self.phi_prime is None else self.phi_prime((dp - dm) / (dp + dm + 1e-5))
        for c in range(L):
            a_c = self._vWeights_ls[c]
            own = np.where(y == c)[0]                            # c is correct here -> mu^+
            foreign = np.where(cm_lab == c)[0]                   # c is the wrong class here -> mu^-
            grad = np.zeros(self._V)
            if own.size:
                num_plus = (a_c[:, None] * dp_V[:, own] * dm[None, own]) / den[None, own]
                if w_mu is not None:
                    num_plus = num_plus * w_mu[None, own]
                grad = grad + num_plus.sum(axis=1)
            if foreign.size:
                num_minus = -(a_c[:, None] * dm_V[:, foreign] * dp[None, foreign]) / den[None, foreign]
                if w_mu is not None:
                    num_minus = num_minus * w_mu[None, foreign]
                grad = grad + num_minus.sum(axis=1)
            g_norm = np.linalg.norm(grad, 2) + 1e-12             # L1->L2 norm as in existing code
            a_new = a_c - self.eta * grad / g_norm
            self._vWeights_ls[c] = _normalize_l2(a_new)          # Σ a^(c)_v^2 = 1

    def fit(self, DL, y):
        """
        Train M3GLVQ with ONE squared-weight vector a^(l) per label.

        Mirrors the general (arbitrary-K) median-LVQ prototype optimization, but
        keeps an (L, V) weight matrix instead of a single (V,) vector. After each
        accepted prototype swap, every label's weight vector is updated by one
        gradient step and L2-normalized.

        DL : list of V distance matrices, each (m, m)
        y  : (m,) label array
        """
        self._init_tracking()
        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        # ---- per-label weight matrix (L, V), each row L2-normalized ----
        # v_init_per_label: dict {label: a-vector} -> eigene Brille je Label
        #   (zum Einfrieren label-spezifischer Gewichte mit eta=0)
        # v_init: ein Vektor -> auf alle Label kopiert
        if getattr(self, 'v_init_per_label', None) is not None:
            self._vWeights_ls = np.zeros((L, self._V))
            for li, lab in enumerate(unique_labels):
                self._vWeights_ls[li] = _normalize_l2(
                    np.asarray(self.v_init_per_label[lab], dtype=float))
        elif self.v_init is not None:
            base = _normalize_l2(self.v_init)
            self._vWeights_ls = np.tile(base, (L, 1))
        else:
            self._vWeights_ls = np.full((L, self._V), 1.0 / np.sqrt(self._V))
        self._ls_labels = np.array(unique_labels, copy=True)

        K_per = self._resolve_K_per_label(unique_labels)
        total_K = int(K_per.sum())
        self._y = np.repeat(unique_labels, K_per)
        self._w = np.zeros(total_K, dtype=int)

        # label index of each data point and of each prototype
        lab_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_idx = np.array([lab_to_idx[v] for v in y])
        proto_lab_idx = np.array([lab_to_idx[v] for v in self._y])

        # ---- prototype initialization: uniform hard-RNG via proto_init ----
        # Seeded per (base_seed, label) on the LABEL-SPECIFIC combined distance
        # (overall_l); raises on degenerate small classes (no medoid fallback).
        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            offset = 0
            for l, lab in enumerate(unique_labels):
                k_l = int(K_per[l])
                idx_w = np.arange(offset, offset + k_l)
                inClass = np.where(y == lab)[0]
                overall_l = self._overall_for_label(M, l)
                self._w[idx_w] = init_class_prototypes(
                    overall_l, inClass, k_l,
                    base_seed=self.base_seed, label=lab, view_tag="label",
                )
                offset += k_l

        rows = np.arange(self._m)

        if L != 2:
            raise NotImplementedError(
                "coupled variant (foreign-class d^-) is implemented for 2 classes; "
                "for L>2 use per-prototype class metric.")
        def _assign(overall_L):
            """closest +/- prototypes and d^+, d^- under per-label metrics, using
            the PROVIDED cached per-label combined matrices (no recompute)."""
            closest_plus = np.zeros(self._m, dtype=int)
            closest_minus = np.zeros(self._m, dtype=int)
            dp = np.zeros(self._m); dm = np.zeros(self._m)
            dp_V = np.zeros((self._V, self._m))
            dm_V = np.zeros((self._V, self._m))
            for l, lab in enumerate(unique_labels):
                inClass = np.where(y == lab)[0]
                overall_l = overall_L[l]
                # COUPLED (Paper Eq. 6): d^- uses the metric of the FOREIGN
                # class (the class of the wrong prototype). For binary the
                # foreign class is 1-l; for >2 classes this needs per-prototype
                # class handling (raise below to stay honest).
                l_other = 1 - l
                overall_other = overall_L[l_other]
                in_w = np.where(self._y == lab)[0]
                out_w = np.where(self._y != lab)[0]
                # d^+ : nearest OWN prototype, own metric
                Dp = overall_l[inClass, :][:, self._w[in_w]]
                closest_plus[inClass] = in_w[np.argmin(Dp, axis=1)]
                # d^- : nearest WRONG prototype, FOREIGN metric
                Dm = overall_other[inClass, :][:, self._w[out_w]]
                closest_minus[inClass] = out_w[np.argmin(Dm, axis=1)]
                dp[inClass] = overall_l[inClass, self._w[closest_plus[inClass]]]
                dm[inClass] = overall_other[inClass, self._w[closest_minus[inClass]]]
                dp_V[:, inClass] = M[:, inClass, self._w[closest_plus[inClass]]]
                dm_V[:, inClass] = M[:, inClass, self._w[closest_minus[inClass]]]
            return closest_plus, closest_minus, dp, dm, dp_V, dm_V

        # per-label combined matrices depend only on the (per-label) weights, not
        # on prototype positions -> rebuild only after a weight step, not per swap.
        overall_L = [self._overall_for_label(M, l) for l in range(L)]
        closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign(overall_L)
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]
        self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])

        # best-so-far: fixed-eta weight steps can overshoot at large eta; keep the
        # lowest-loss state and return it instead of the final iterate. No-op when
        # the run is monotone.
        best_loss = self._loss[-1]
        best_w = self._w.copy()
        best_vw = self._vWeights_ls.copy()

        def _track_best():
            nonlocal best_loss, best_w, best_vw
            if self._loss[-1] < best_loss:
                best_loss = self._loss[-1]
                best_w = self._w.copy()
                best_vw = self._vWeights_ls.copy()

        # ---- optimization loop: greedy prototype swap + per-label weight step ----
        # Candidate scoring uses the TRUE two-sided GLVQ objective (both d^+ and
        # d^-) over the cached per-label matrices -> selection delta == objective.
        pts_by_l = [np.where(y == lab)[0] for lab in unique_labels]
        in_w_by_l = [np.where(self._y == lab)[0] for lab in unique_labels]
        out_w_by_l = [np.where(self._y != lab)[0] for lab in unique_labels]

        def _dp_dm_from_w(w, overall_L):
            """Exact d^+ (nearest OWN proto) and d^- (nearest WRONG proto) for
            every point, d^+ under the own-label metric, d^- under the
            foreign-label metric (coupled), given prototypes w."""
            dp_ = np.empty(self._m)
            dm_ = np.empty(self._m)
            for l in range(L):
                ov = overall_L[l]
                ov_other = overall_L[1 - l]         # COUPLED: foreign metric for d^-
                pts = pts_by_l[l]
                dp_[pts] = ov[np.ix_(pts, w[in_w_by_l[l]])].min(axis=1)
                dm_[pts] = ov_other[np.ix_(pts, w[out_w_by_l[l]])].min(axis=1)
            return dp_, dm_

        def _loss_from_dp_dm(dp_, dm_):
            return float(self.phi((dp_ - dm_) / (dp_ + dm_ + 1e-5)).sum())

        # ================= DECOUPLED training (opt-in) =====================
        # Per-label weight step runs EVERY epoch; prototype early stop replaced
        # by best-loss-plateau + weight-settle patience. One snapshot per epoch.
        if self.decoupled:
            v_window = []
            stale = 0
            for ep in range(self.T):
                if self.eta_schedule is not None:
                    self.eta = float(self.eta_schedule(ep))
                cur_loss = self._loss[-1]

                # --- prototype step: at most ONE improving swap (same search) ---
                proto_losses = np.zeros(total_K)
                for k in range(total_K):
                    proto_losses[k] = (np.sum(dp[closest_plus == k])
                                       - np.sum(dm[closest_minus == k]))
                for k in np.argsort(-proto_losses):
                    rf_plus = np.where(closest_plus == k)[0]
                    best_delta = 0.0
                    best_i = None
                    w_try = self._w.copy()
                    for i in rf_plus:
                        if i == self._w[k]:
                            continue
                        w_try[k] = i
                        dp_c, dm_c = _dp_dm_from_w(w_try, overall_L)
                        delta = _loss_from_dp_dm(dp_c, dm_c) - cur_loss
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                    w_try[k] = self._w[k]
                    if best_i is None:
                        continue
                    self._w[k] = best_i
                    # swap leaves overall_L unchanged: reassign only
                    closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign(overall_L)
                    break

                # --- COUPLED weight step: ALWAYS (Paper Eq. 11-14) ---
                self._weights_update_coupled(dp, dm, dp_V, dm_V, y,
                                             closest_minus, proto_lab_idx)
                overall_L = [self._overall_for_label(M, l) for l in range(L)]
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign(overall_L)
                mus = self.phi((dp - dm) / (dp + dm + 1e-5))
                self._loss.append(float(mus.sum()))
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])

                # --- PRIMARY stop: best-loss plateau (oscillation-immune) ---
                prev_best = best_loss
                _track_best()
                if best_loss < prev_best - self.eps_loss * abs(prev_best):
                    stale = 0
                else:
                    stale += 1
                plateaued = stale >= self.patience
                # --- OPTIONAL weight-settle: NET displacement over the window ---
                settled = True
                if self.require_weight_settle:
                    v_window.append(self._vWeights_ls.copy())
                    if len(v_window) > self.patience + 1:
                        v_window.pop(0)
                    settled = (len(v_window) > self.patience and
                               float(np.linalg.norm(self._vWeights_ls - v_window[0])) < self.eps_weight)
                if plateaued and settled:
                    break

            # best-so-far restore (same contract as the coupled tail)
            self._w = best_w
            self._vWeights_ls = best_vw
            self.final_matrix_ls_ = {
                unique_labels[l]: self._overall_for_label(M, l) for l in range(L)
            }
            self.final_matrix_ = self.final_matrix_ls_[unique_labels[0]]
            return self
        # =================== end decoupled block ===========================

        for _ in range(self.T):
            # overall_L / assignments carry over (init or previous weight step)
            cur_loss = self._loss[-1]

            proto_losses = np.zeros(total_K)
            for k in range(total_K):
                proto_losses[k] = (np.sum(dp[closest_plus == k])
                                   - np.sum(dm[closest_minus == k]))

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                rf_plus = np.where(closest_plus == k)[0]

                best_delta = 0.0
                best_i = None
                w_try = self._w.copy()
                for i in rf_plus:
                    if i == self._w[k]:
                        continue
                    w_try[k] = i
                    dp_c, dm_c = _dp_dm_from_w(w_try, overall_L)
                    delta = _loss_from_dp_dm(dp_c, dm_c) - cur_loss
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                w_try[k] = self._w[k]

                if best_i is None:
                    continue

                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta

                # swap leaves overall_L unchanged: reassign only
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign(overall_L)
                mus = self.phi((dp - dm) / (dp + dm + 1e-5))
                self._loss.append(float(mus.sum()))
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])
                _track_best()

                # COUPLED weight step (Paper Eq. 11-14) -> rebuild overall_L
                self._weights_update_coupled(dp, dm, dp_V, dm_V, y,
                                             closest_minus, proto_lab_idx)
                overall_L = [self._overall_for_label(M, l) for l in range(L)]
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign(overall_L)
                mus = self.phi((dp - dm) / (dp + dm + 1e-5))
                self._loss.append(float(mus.sum()))
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])
                _track_best()
                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])
                break

        # return the best-so-far state (identical to final when monotone)
        self._w = best_w
        self._vWeights_ls = best_vw

        # final per-label combined matrices
        self.final_matrix_ls_ = {
            unique_labels[l]: self._overall_for_label(M, l) for l in range(L)
        }
        self.final_matrix_ = self.final_matrix_ls_[unique_labels[0]]
        return self

    def get_vweights(self):
        """Final per-label weights as a dict {label: {'a':..., 'a_sq':...}}."""
        if not hasattr(self, "_vWeights_ls") or self._vWeights_ls is None:
            raise RuntimeError("fit_label_specific has not been called yet.")
        out = {}
        for l, lab in enumerate(self._ls_labels):
            a = np.array(self._vWeights_ls[l], dtype=float, copy=True)
            out[lab] = {"a": a, "a_sq": a ** 2}
        return out

    def predict(self, DL):
        """
        Predict labels using the per-label metrics.

        For each label l the combined distance to that label's prototypes is
        computed with a^(l); a point is assigned to the label whose nearest
        prototype (under that label's own metric) is closest.
        """
        Mtest = np.stack(DL, axis=0)
        n = Mtest.shape[1]
        best_d = np.full(n, np.inf)
        best_lab = np.empty(n, dtype=self._ls_labels.dtype)
        for l, lab in enumerate(self._ls_labels):
            D = np.tensordot(self._vWeights_ls[l] ** 2, Mtest, axes=(0, 0))
            if D.shape[1] == self._m:
                D = D[:, self._w]
            cols = np.where(self._y == lab)[0]
            d_lab = D[:, cols].min(axis=1)
            upd = d_lab < best_d
            best_d[upd] = d_lab[upd]
            best_lab[upd] = lab
        return best_lab

    # ---------- Getters ----------
    def get_prototype_path(self):
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        """Path of the raw per-label factors a^(l)_v over training.

        Returns an array of shape (T, L, V): for each tracked step, the full
        (L, V) weight matrix with each row L2-normalized (||a^(l)||_2 = 1).
        For the interpretable group mixture a^(l)_v^2 use get_vweight_sq_path().
        Note: the sign of a^(l)_v is not identifiable; only a^(l)_v^2 is
        meaningful for interpretation.
        """
        if self._v_history is None:
            return None
        return np.stack(self._v_history, axis=0) if len(self._v_history) \
            else np.empty((0,))

    def get_vweight_sq_path(self):
        """Path of the squared per-label weights a^(l)_v^2, shape (T, L, V).

        Each row (over V) sums to 1 and is the interpretable group mixture
        for that label.
        """
        path = self.get_vweight_path()
        if path is None:
            return None
        return path ** 2 if path.size else path

    def get_training_log(self):
        if self._log is None:
            return None
        return {"loss": np.array(self._log["loss"], dtype=float)}