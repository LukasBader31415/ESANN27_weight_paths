"""Count how often the initial prototypes of the paper runs needed an alternative seed (collapse rescue).

Recomputes only the initialization (no fits) for every start set of the paper runs: datasets 3-17 with
K in {2, 6, 12} and repetitions 0-4, plus the K sweep (K = 2, ..., 20) on datasets 3, 4 and 5. One start set per
(dataset, K, repetition) is shared by all variants (`run_tn.common_init`); it consists of one call of
`init_class_prototypes` per class. The paper runs had the rescue log switched off (`RESCUE_LOG_PATH = None`); here
it is switched on.

Checks that the recomputed start sets are the ones used in the paper runs:
- example run (dataset 3, K = 6, repetition 0): `w_history[0]` of the stored snapshots in
  `results/public/2026-10-20_fig2_example_run_ds3/hist/` must equal the recomputed start set;
- stored fit `results/public/2026-10-20_SG_explore_H0/reference_fit.csv`: `ba_init` (test BA at epoch 0, equal
  weights) must be reproduced from the recomputed start set (5 decimals);
- optionally `--cache DIR` (the `cache/toy_H0/` folder of a full re-run): every stored snapshot of every start set.

    python experiments/TN_ksweep/count_init_rescues.py [--cache cache/toy_H0] [--out output/init_rescues.csv]

Runs locally in a few minutes.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

import numpy as np
import pandas as pd

import run_tn
import src.common.proto_init as proto_init

# configs of the paper runs (README, "Configs and result folders"); config_full / config_lr_explore use the same
# start sets as config_sg9_k_s3 / config_sig_explore (same data, base_seed and K) and are covered by them
CONFIGS = ["config_sig_explore.yaml", "config_sig_confirm_s4.yaml"] + \
          [f"config_sig_s{s}.yaml" for s in range(5, 18)] + \
          ["config_sg9_k_s3.yaml", "config_sg9_k_s4.yaml", "config_sg9_k_s5.yaml"]
EXAMPLE = ROOT / "results/public/2026-10-20_fig2_example_run_ds3/hist"
REFERENCE = ROOT / "results/public/2026-10-20_SG_explore_H0/reference_fit.csv"


def start_sets(log_path):
    """All start sets of the paper runs; returns rows (one per start set) and {(cfg, K, rep): start}."""
    proto_init.RESCUE_LOG_PATH = str(log_path)
    rows, starts, seen = [], {}, set()
    for name in CONFIGS:
        cfg = run_tn.load_config(HERE / name)
        D = run_tn.build_data(cfg)
        sig = run_tn.cache_signature(cfg)
        for K in cfg["K"]:
            for rep in cfg["reps"]:
                n_before = len(open(log_path).readlines()) if log_path.exists() else 0
                start = run_tn.common_init(D, K, rep, cfg)
                new = open(log_path).readlines()[n_before:] if log_path.exists() else []
                # keep the data only for the reference check (dataset 3), the matrices are large
                starts[(name, K, rep)] = (start, D if name == "config_sig_explore.yaml" else None, sig)
                key = (cfg["data_seed"], K, rep)
                if key in seen:          # same start set as in an earlier config (datasets 3-5, K 2/6/12)
                    continue
                seen.add(key)
                rows.append(dict(dataset=cfg["data_seed"], K=K, rep=rep, config=name, n_class_inits=2,
                                 n_rescued=len(new),
                                 retry_attempts=";".join(str(json.loads(l)["rescued_attempt"]) for l in new)))
    proto_init.RESCUE_LOG_PATH = None
    return pd.DataFrame(rows), starts


def check_example(starts):
    start, _, _ = starts[("config_sig_explore.yaml", 6, 0)]
    w0 = np.concatenate([start[0], start[1]])
    out = []
    for f in sorted(EXAMPLE.glob("*/K=06_rep=0.npz")):
        stored = np.load(f)["w_history"][0]
        out.append((f.parent.name, bool(np.array_equal(stored, w0))))
    return out


def check_reference(starts):
    ref = pd.read_csv(REFERENCE)
    out = []
    for r in ref.itertuples():
        start, D, _ = starts[("config_sig_explore.yaml", int(r.K), int(r.rep))]
        w = np.concatenate([start[0], start[1]])
        proto_lab = np.repeat([0, 1], int(r.K))
        V = D["M_test"].shape[0]
        a2 = np.full((2, V), 1.0 / V)
        ba = run_tn._ba(D["y_test"], run_tn._predict(D["M_test"], w, proto_lab, a2))
        out.append((r.variant, int(r.K), int(r.rep), round(ba, 5), float(r.ba_init), round(ba, 5) == float(r.ba_init)))
    return out


def check_cache(starts, cache):
    n, bad = 0, []
    for (name, K, rep), (start, _, sig) in starts.items():
        w0 = np.sort(np.concatenate([start[0], start[1]]))
        for f in (cache / sig / "hist").glob(f"*/K={K:02d}_rep={rep}.npz"):
            n += 1
            if not np.array_equal(np.sort(np.load(f)["w_history"][0]), w0):
                bad.append(str(f))
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None, help="cache/toy_H0 of a full re-run (optional)")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "init_rescues.csv")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        rows, starts = start_sets(Path(tmp) / "rescues.jsonl")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(a.out, index=False)

    for label, sub in (("all paper start sets", rows),
                       ("Table 1 runs (datasets 3-17, K in {2, 6, 12})", rows[rows.K.isin([2, 6, 12])])):
        n_sets, n_inits, n_resc = len(sub), int(sub.n_class_inits.sum()), int(sub.n_rescued.sum())
        print(f"{label}: {n_sets} start sets (dataset, K, repetition), {n_inits} class initializations; "
              f"rescued (alternative seed needed): {n_resc} ({100 * n_resc / n_inits:.2f} %), "
              f"in {int((sub.n_rescued > 0).sum())} start sets")
    if rows.n_rescued.sum():
        print(rows[rows.n_rescued > 0].groupby(["dataset", "K"]).n_rescued.sum().to_string())
        print("retry attempt used:", sorted(int(x) for s in rows.retry_attempts if s for x in s.split(";")))
    print("per start set:", a.out)

    ex = check_example(starts)
    print("example run, w_history[0] identical:", ex)
    rf = check_reference(starts)
    print("reference_fit.csv, ba_init reproduced:", rf)
    ok = all(x[1] for x in ex) and all(x[-1] for x in rf)
    if a.cache is not None:
        n, bad = check_cache(starts, a.cache)
        print(f"cache: {n} stored snapshots compared, {len(bad)} differ")
        ok = ok and not bad and n > 0
    print("start sets identical to the stored ones:", ok)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
