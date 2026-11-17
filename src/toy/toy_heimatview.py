"""
Home-view generator ("Heimat-View"), version 1.0 -- synthetic multi-view data with a
known relevance profile per cluster.

Data
----
Two classes with K_true clusters each (default 6, 50 points per cluster, m = 600).
Every point is observed in four 2-D views: three informative views A, B, C and one
pure-noise view N.  Each view yields one Euclidean dissimilarity matrix, scaled to
[0, 1] by its maximum.

Cluster centres are drawn independently per informative view, uniformly in a disc of
radius R = 6 with a minimum distance g = 1.6 between centres; the class membership of
a centre plays no role, so the classes are interleaved in every view.  View N has no
centres.

Relevance
---------
Each cluster r carries a relevance vector w_r over (A, B, C) that sums to one.  It
sets the isotropic noise of the cluster in every informative view,

    sigma_{r,v} = s / sqrt(3 * w_{r,v}),      x_{i,v} = mu_{r(i),v} + sigma_{r(i),v} * eps,
    eps ~ N(0, I_2),

so that w_{r,v} is proportional to 1 / sigma_{r,v}^2: relevance is the inverse noise
variance of a view for a cluster.  s is the base noise level (at w = 1/3 every view has
spread s).  View N is drawn as x ~ N(0, sigma_N^2 I_2) independently of cluster and
class (sigma_N = 3); its ground-truth weight is 0.

In plans T0-T2 every cluster has one home view with weight w_heim; the other two
informative views share the rest, (1 - w_heim) / 2 each.  The plans differ only in
which view is home:

    T0  control      all clusters: home A
    T1  class        class 0 -> A, class 1 -> B
    T2  prototype    class 0: 3 clusters -> A, 3 -> C;  class 1: 3 -> B, 3 -> C
    T2m prototype,   class 0: 4 clusters -> A, 2 -> B;  class 1: 4 clusters -> B, 2 -> A
        mirrored     (C carries (1 - w_heim) / 2 in every cluster)
    T3  graded       w_r ~ Dirichlet(1, 1, 1) per cluster (no single home view)

T0 needs one global metric, T1 one metric per class, and T2/T2m one metric per cluster;
the class profiles in T2/T2m are mixtures and therefore only partly right for each
cluster.  In T2m the class profiles point in opposite directions (A vs. B), so the
class level gains over the global one, while the two minority clusters per class keep
a gap open for the prototype level.  (v1.1 adds T2m and oracle_fair; T0-T3 and
oracle() are unchanged from v1.0.)

Heterogeneity axis (v1.2): plans H0-H3 have h = j/6 minority clusters per class,
class 0 = (6-j) x home A + j x home B, class 1 mirrored.  H0 = T1, H2 = T2m.
oracle_fair(pooling="w") is a sensitivity variant that forms the class-wise and global
weights as the mean of the per-cluster weight vectors instead of from pooled q.
For a fixed seed the centres and the standard-normal draws are shared by all plans,
so the plans differ only in their relevance profile.

Fair oracle (oracle_fair, v1.1)
-------------------------------
All levels use one estimation rule on the training sample and are evaluated on the
test sample without any selection.  For cluster r and view v let q_{r,v} be the mean
squared within-cluster dissimilarity (normalised D, all pairs i != j).  Weights are
w_v proportional to 1 / q_v, normalised to sum one over all four views (N included):
global from q pooled over all clusters, class-wise from q pooled over the clusters of a
class, prototype-wise from q_{r,.} of each cluster.  Distances are combined exactly as
in sM3GLVQ, d(x, p) = sum_v w_v D_v(x, p) with w = alpha^2; a point gets the class of its
nearest prototype, each prototype measured with the weights of its level (ties -> 0).

Oracle v1.0 (model-free ceilings)
---------------------------------
Prototypes are the medoids of the true clusters on the training sample (uniform
average of all four views), shared by all weighting levels.  Accuracy is the balanced
accuracy of the nearest-prototype rule on an independent test sample (same centres,
new points).  Levels: single view; equal weights on A, B, C, N; best global vector on
a simplex grid over (A, B, C) with N = 0 (step .1); best pair of class-wise vectors on
the same grid; prototype-wise ground-truth weights (a lower bound for that level, since
w is optimal for squared, not plain Euclidean distances).  Global and class-wise
vectors are chosen on the test sample, i.e. they are upper bounds for their level.
"""
import itertools
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

VERSION = "1.2"
VIEWS = ("A", "B", "C", "N")
N_INF = 3                      # informative views A, B, C
DIMS = 2
PLANS = ("T0", "T1", "T2", "T2m", "T3", "H0", "H1", "H2", "H3")
DEFAULTS = dict(k_true=6, n_per_cluster=50, radius=6.0, min_gap=1.6, sigma_noise=3.0)


# --------------------------------------------------------------------------- #
# Plans and ground truth
# --------------------------------------------------------------------------- #
def plan_homes(plan, k_true=6):
    """Class and home view (0=A, 1=B, 2=C; None for T3) per cluster, class 0 first."""
    h = k_true // 2
    maj = int(round(2 * k_true / 3))                    # T2m: 4 of 6 clusters in the class's own home view
    homes = {
        "T0": [0] * k_true + [0] * k_true,
        "T1": [0] * k_true + [1] * k_true,
        "T2": [0] * h + [2] * (k_true - h) + [1] * h + [2] * (k_true - h),
        "T2m": [0] * maj + [1] * (k_true - maj) + [1] * maj + [0] * (k_true - maj),
        "T3": [None] * (2 * k_true),
    }
    for j in range(k_true // 2 + 1):                    # H<j>: j minority clusters per class (mirrored)
        homes[f"H{j}"] = [0] * (k_true - j) + [1] * j + [1] * (k_true - j) + [0] * j
    homes = homes[plan]
    return [(int(r >= k_true), v) for r, v in enumerate(homes)]


def cluster_weights(plan, w_heim, seed, k_true=6):
    """Ground-truth relevance w_r over (A, B, C) per cluster, shape (2*k_true, 3)."""
    if plan == "T3":
        return np.random.default_rng([seed, 1]).dirichlet(np.ones(N_INF), size=2 * k_true)
    W = np.full((2 * k_true, N_INF), (1.0 - w_heim) / 2.0)
    for r, (_, v) in enumerate(plan_homes(plan, k_true)):
        W[r, v] = w_heim
    return W


def _centres(rng, n, radius, min_gap):
    """n centres uniform in a disc, rejection-sampled to a minimum distance."""
    pts = []
    for _ in range(n):
        for _try in range(1000):
            phi, rad = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.random())
            p = np.array([rad * np.cos(phi), rad * np.sin(phi)])
            if all(np.linalg.norm(p - q) >= min_gap for q in pts):
                break
        pts.append(p)
    return np.asarray(pts)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def generate(plan, s, w_heim, seed=0, **kw):
    """Training and test sample for one plan.

    Returns dict with X_train, X_test (m x 8: A, B, C, N), y_*, region_*, W (R x 3),
    sigma (R x 3), centres (R x 3 x 2), params.
    """
    p = {**DEFAULTS, **kw}
    k, n = p["k_true"], p["n_per_cluster"]
    R = 2 * k
    rng_c = np.random.default_rng([seed, 0])
    centres = np.stack([_centres(rng_c, R, p["radius"], p["min_gap"]) for _ in range(N_INF)], axis=1)
    W = cluster_weights(plan, w_heim, seed, k)
    sigma = s / np.sqrt(N_INF * W)

    def sample(stream):
        rng = np.random.default_rng([seed, stream])
        eps = rng.normal(size=(R, n, N_INF + 1, DIMS))
        X = np.empty((R * n, (N_INF + 1) * DIMS))
        for r in range(R):
            rows = slice(r * n, (r + 1) * n)
            for v in range(N_INF):
                X[rows, v * DIMS:(v + 1) * DIMS] = centres[r, v] + sigma[r, v] * eps[r, :, v]
            X[rows, N_INF * DIMS:] = p["sigma_noise"] * eps[r, :, N_INF]
        return X

    region = np.repeat(np.arange(R), n)
    y = (region >= k).astype(int)
    return dict(X_train=sample(2), X_test=sample(3), y_train=y, y_test=y.copy(),
                region_train=region, region_test=region.copy(), W=W, sigma=sigma,
                centres=centres, homes=plan_homes(plan, k),
                params=dict(version=VERSION, plan=plan, s=s, w_heim=w_heim, seed=seed, **p))


def _view(X, v):
    return X[:, v * DIMS:(v + 1) * DIMS]


def make_D_blocks(X):
    """One Euclidean dissimilarity matrix per view, divided by its maximum."""
    D = [squareform(pdist(_view(X, v))) for v in range(len(VIEWS))]
    return [d / d.max() for d in D]


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def simplex_grid(step=0.1, n=N_INF):
    """All weight vectors on the simplex over n views with the given step."""
    q = int(round(1 / step))
    return np.array([c for c in itertools.product(range(q + 1), repeat=n) if sum(c) == q], float) / q


def _ba(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1)]))


def oracle(data, step=0.1):
    """Model-free ceilings on the test sample (see module docstring)."""
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte, reg = data["y_train"], data["y_test"], data["region_train"]
    V = len(VIEWS)
    D_tr = make_D_blocks(Xtr)
    scale = [squareform(pdist(_view(Xtr, v))).max() for v in range(V)]
    Du = sum(D_tr) / V
    R = int(reg.max()) + 1
    proto = np.array([np.where(reg == r)[0][np.argmin(Du[np.ix_(reg == r, reg == r)].sum(1))]
                      for r in range(R)])
    pcls = ytr[proto]
    S = np.stack([cdist(_view(Xte, v), _view(Xtr, v)[proto]) / scale[v] for v in range(V)])  # (V, n, R)

    def ba_rows(Wr):                                    # one weight vector (length V) per prototype
        d = np.einsum("vnr,rv->nr", S, Wr)
        return _ba(yte, pcls[np.argmin(d, 1)])

    out = {f"single_{VIEWS[v]}": ba_rows(np.tile(np.eye(V)[v], (R, 1))) for v in range(V)}
    out["single_best"] = max(out[f"single_{v}"] for v in VIEWS)
    out["equal"] = ba_rows(np.full((R, V), 1.0 / V))

    G = np.hstack([simplex_grid(step), np.zeros((len(simplex_grid(step)), 1))])   # N = 0
    d = np.einsum("vnr,gv->gnr", S, G)                  # (g, n, R)
    pred = pcls[np.argmin(d, 2)]                        # (g, n)
    ba_g = np.array([_ba(yte, p) for p in pred])
    out["global"], gi = float(ba_g.max()), int(ba_g.argmax())
    out["global_w"] = G[gi, :N_INF].round(2).tolist()

    m0 = d[:, :, pcls == 0].min(2)                      # (g, n) distance to nearest class-0 prototype
    m1 = d[:, :, pcls == 1].min(2)
    i0, i1 = yte == 0, yte == 1
    rec0 = (m0[:, None, i0] <= m1[None, :, i0]).mean(2)  # (g0, g1)
    rec1 = (m1[None, :, i1] < m0[:, None, i1]).mean(2)
    ba_l = (rec0 + rec1) / 2
    a, b = np.unravel_index(int(ba_l.argmax()), ba_l.shape)
    out["label"] = float(ba_l[a, b])
    out["label_w0"], out["label_w1"] = G[a, :N_INF].round(2).tolist(), G[b, :N_INF].round(2).tolist()

    Wp = np.hstack([data["W"], np.zeros((R, 1))])
    out["proto"] = ba_rows(Wp)
    return out


def _medoids_and_test_dist(data):
    """Medoid prototypes on the training sample and normalised test-to-prototype dissimilarities.
    Same construction as in oracle() (v1.0)."""
    Xtr, Xte, ytr, reg = data["X_train"], data["X_test"], data["y_train"], data["region_train"]
    V = len(VIEWS)
    D_tr = make_D_blocks(Xtr)
    scale = [squareform(pdist(_view(Xtr, v))).max() for v in range(V)]
    Du = sum(D_tr) / V
    R = int(reg.max()) + 1
    proto = np.array([np.where(reg == r)[0][np.argmin(Du[np.ix_(reg == r, reg == r)].sum(1))]
                      for r in range(R)])
    S = np.stack([cdist(_view(Xte, v), _view(Xtr, v)[proto]) / scale[v] for v in range(V)])  # (V, n, R)
    return D_tr, proto, ytr[proto], S


def within_q(D_tr, region):
    """q[r, v]: mean squared within-cluster dissimilarity (pairs i != j) per cluster and view."""
    R = int(region.max()) + 1
    q = np.empty((R, len(D_tr)))
    for r in range(R):
        idx = np.where(region == r)[0]
        n = len(idx)
        for v, D in enumerate(D_tr):
            q[r, v] = (D[np.ix_(idx, idx)] ** 2).sum() / (n * (n - 1))
    return q


def _inv_weights(q):
    w = 1.0 / q
    return w / w.sum(-1, keepdims=True)


def oracle_fair(data, pooling="q"):
    """Fair oracle (v1.1): one estimation rule w ~ 1/q for all levels, no selection on the test.
    pooling="q": class-wise/global weights from pooled q (default, v1.1);
    pooling="w": mean of the per-cluster weight vectors (sensitivity, v1.2)."""
    yte, reg = data["y_test"], data["region_train"]
    V = len(VIEWS)
    D_tr, proto, pcls, S = _medoids_and_test_dist(data)
    R = len(proto)
    q = within_q(D_tr, reg)
    size = np.bincount(reg, minlength=R).astype(float)

    w_r = _inv_weights(q)

    def pooled(mask):                                   # returns a "q" whose inverse gives the level's weights
        if pooling == "w":
            w = (w_r[mask] * size[mask, None]).sum(0) / size[mask].sum()
            return 1.0 / w
        return (q[mask] * size[mask, None]).sum(0) / size[mask].sum()

    def ba_rows(Wr):                                    # d(x, p_r) = sum_v W[r, v] D_v(x, p_r)
        d = np.einsum("vnr,rv->nr", S, Wr)
        return _ba(yte, pcls[np.argmin(d, 1)])          # argmin -> first prototype (class 0) on ties

    rcls = np.array([pcls[r] for r in range(R)])        # cluster r <-> prototype r
    w_glob = _inv_weights(pooled(np.ones(R, bool)))
    w_lab = {c: _inv_weights(pooled(rcls == c)) for c in (0, 1)}
    w_proto = _inv_weights(q)

    out = {f"single_{VIEWS[v]}": ba_rows(np.tile(np.eye(V)[v], (R, 1))) for v in range(V)}
    out["single_best"] = max(out[f"single_{v}"] for v in VIEWS)
    out["equal"] = ba_rows(np.full((R, V), 1.0 / V))
    out["global"] = ba_rows(np.tile(w_glob, (R, 1)))
    out["label"] = ba_rows(np.vstack([w_lab[c] for c in rcls]))
    out["proto"] = ba_rows(w_proto)
    out["global_w"] = w_glob.round(2).tolist()
    out["label_w0"], out["label_w1"] = w_lab[0].round(2).tolist(), w_lab[1].round(2).tolist()
    return out


def ladder(o):
    """Gaps of the full ladder and the acceptance test of plan B1 (per seed)."""
    g = {"gap_global_single": o["global"] - o["single_best"],
         "gap_global_equal": o["global"] - o["equal"],
         "gap_label_global": o["label"] - o["global"],
         "gap_proto_label": o["proto"] - o["label"]}
    g["min_gap"] = min(g.values())
    tol = 1e-9                                          # BA granularity is 1/300; avoid float artefacts at .03
    g["pass"] = bool(g["min_gap"] >= 0.03 - tol and o["single_best"] >= 0.60 - tol and o["proto"] <= 0.97 + tol)
    return g
