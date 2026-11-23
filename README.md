# When do relevance weights learn?

Code and aggregated results for

> Lukas Bader and Dietlind Zühlke. *When do relevance weights learn?* ESANN 2027 (submitted, paper ID ES2027-119).

```bibtex
@inproceedings{bader_2027_weights,
  author    = {Bader, Lukas and Z{\"u}hlke, Dietlind},
  title     = {When do relevance weights learn?},
  booktitle = {European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2027)},
  year      = {2027},
  note      = {submitted, paper ID ES2027-119}
}
```

## Summary

On synthetic multi-view data with known relevances, we decompose the gain of multi-matrix median GLVQ (global weights G, class-wise weights CW) into prototype learning and weight learning. With the identity in the GLVQ cost, the view weights pass a state of high validation accuracy and often continue to a corner where a single view carries almost all weight (max α_v² > .99), while the training cost keeps decreasing. Replacing the identity with a sigmoid stops the weight path for G and slows it for CW, and improves balanced accuracy over the identity at the end of training on all 15 datasets.

## Quickstart

Python 3.12 (`numpy<2` is required by `proto-dist-ml` 1.0.1).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
jupyter nbconvert --to notebook --execute notebooks/reproduce_paper.ipynb --output-dir output
```

or open `notebooks/reproduce_paper.ipynb` and run all cells. Part 1 builds Table 1, Fig. 1 and Fig. 2 (PDF and PNG in final size, in `output/`) from the aggregated CSVs and prints the numbers quoted in the text, in a few seconds. Part 2 re-fits a single run (about 15 s) and lists the commands for the full re-run. `docs/numbers.md` maps every number in the paper to a file, column and row.

## Repository layout

```
src/toy/toy_heimatview.py           data generator (plan H0: one home view per class)
src/models/                         M3GLVQ_global_decoupled.py (G), M3GLVQ_label_coupled_WSOM.py (CW)
src/common/proto_init.py            initial prototypes (relational neural gas from proto-dist-ml)
src/paper_figs/                     Table 1 (table.py), Fig. 1 and Fig. 2 (figures.py), data access (data.py)
experiments/TN_ksweep/run_tn.py     runner: one config = one dataset, all variants x K x repetitions
experiments/TN_ksweep/config_*.yaml configs of the paper runs (see below)
experiments/TN_ksweep/predictions_sg7.yaml   expectations E1-E5, fixed before datasets 9-15
experiments/TN_ksweep/criteria_sg.yaml       criteria for choosing beta (exploration, datasets 3 and 4)
experiments/TN_ksweep/eval_*.py, m_textzahlen_fig2.py   evaluation scripts that produced results/public/
experiments/TN_ksweep/count_init_rescues.py  recomputes the initial prototypes, counts collapse rescues
results/public/                     aggregated results that feed the paper (CSV), example run for Fig. 2
notebooks/reproduce_paper.ipynb     Table 1, Fig. 1, Fig. 2, numbers, single re-fit
docs/numbers.md                     every number in the paper -> file / column / row
```

Comments and docstrings in the code are partly in German; identifiers follow the paper as follows: `frozen` = prototype learning only, `orakel` / `frozen_oracle_*` = oracle weights, `id` = identity, `sig` = sigmoid (`xi` in the code = β in the paper), `Best_val` = best validation epoch, `Ecke` = corner, `Seed` / `DS` = dataset, `rep` = repetition.

## Datasets and roles

The paper does not number the datasets; this repository does. A dataset is identified by the seed of the generator (`data_seed` in the configs; plan H0, `s = 2.5`, home-view relevance 0.8; 600 training, 600 validation and 3000 test points).

| Role | Datasets (seeds) | Used for |
|---|---|---|
| Exploration | 3–8 | example run and learning rates on seed 3; β chosen on seed 3 and checked on seed 4 (`criteria_sg.yaml`); expectations E1–E5 derived from seeds 3–8 |
| K sweep | 3, 4, 5 (chosen in advance) | K = 2, …, 20 (Fig. 1) |
| Confirmation | 9–15 | the only datasets on which E1–E5 are judged |
| Descriptive | 16, 17 | run later, reported without a verdict |
| Table 1 | all 15 (3–17) | K ∈ {2, 6, 12}, five repetitions; corner counts over 3–15 |

A repetition `rep` uses its own seed for the initial prototypes (`base_seed + rep`, `base_seed = 42`) on the same data; within a repetition all variants start from the same set of initial prototypes with equal weights (see *Initialization and scaling*).

### Initialization and scaling

As implemented in the code of the paper runs:

- **Equal start weights**: α_v = 1/√V, i.e. α_v² = 1/V (V = 4 views). This is the default of both models when no `v_init` is given (`src/models/M3GLVQ_global_decoupled.py` and `M3GLVQ_label_coupled_WSOM.py`, `_vWeights`); `experiments/TN_ksweep/run_tn.py` (`common_init`) uses the same weights for the dissimilarity of the initialization.
- **Initial prototypes per class**: relational neural gas (`proto_dist_ml.rng.RNG(K)`, `is_squared=True`) on the squared equally weighted dissimilarity of the class's training points; each of the K prototypes is moved to the closest data point of its class (`src/common/proto_init.py`, `init_class_prototypes`, called by `run_tn.common_init`).
- **Seeds**: per class from `base_seed + rep` (`base_seed = 42`), the class label and the view tag `global` (SHA-256 of `"<seed>|global|<label>"`, combined with the seed by `numpy.random.SeedSequence`; `proto_init._seed_for`). There is one start set per (dataset, K, repetition), shared by all variants (`run_tn.preset_init`; every fit checks that it started from this set).
- **Collapse handling**: if the canonical seed gives fewer than K distinct points, up to `MAX_INIT_RETRIES = 8` alternative deterministic seeds (view tags `global#retry1`, …) are tried; the first with K distinct points is used, otherwise `DegenerateInitError` is raised. The rescue log was off in the paper runs (`RESCUE_LOG_PATH = None`). Recomputed with `experiments/TN_ksweep/count_init_rescues.py` (initialization only, no fits, a few minutes): of the 930 class initializations of the paper runs (465 start sets: datasets 3–17 with K ∈ {2, 6, 12}, and K = 2, …, 20 on datasets 3–5), 18 (1.9 %) needed an alternative seed (15 the first, 3 the second); in the runs of Table 1 (K ∈ {2, 6, 12}), 5 of 450 (1.1 %). The recomputed start sets are identical to those of the stored runs (example run and `reference_fit.csv`).
- **Scaling**: each view matrix is the Euclidean distance matrix of that view divided by its maximum pairwise distance on the training data (`src/toy/toy_heimatview.py`, `make_D_blocks`, called by `run_tn.build_data`); validation and test distances to the training points are divided by the same factor (`run_tn.build_data`, `scale`).

### Configs and result folders

| Config (`experiments/TN_ksweep/`) | Run id (`results/public/<run id>/`) | Dataset | Content |
|---|---|---|---|
| `config_full.yaml` | `2026-10-15_TN_ksweep_H0` | 3 | identity, frozen, oracle, K 2–20 |
| `config_lr_explore.yaml` | `2026-10-15_LR_explore_H0` | 3 | 14 further learning-rate schedules (identity) |
| `config_sig_explore.yaml` | `2026-10-20_SG_explore_H0` | 3 | sigmoid β ∈ {5, 10, 20}, K {2, 6, 12} |
| `config_sig_confirm_s4.yaml` | `2026-10-20_SG_confirm_H0s4` | 4 | check of β = 10 |
| `config_sig_s<s>.yaml`, s = 5…17 | `2026-10-20_SG_seed<s>_H0` | 5–17 | identity, sigmoid β = 10, frozen, oracle, K {2, 6, 12} |
| `config_sg9_k_s3.yaml`, `_s4`, `_s5` | `2026-10-20_SG9_k_H0s3`, `_s4`, `_s5` | 3, 4, 5 | K sweep 2–20 with sigmoid |

Evaluation: `eval_seeds.py` → `2026-10-20_SG7_seeds3-15/`, `eval_sg9.py` → `2026-10-20_SG9_seeds3-17/`, `m_textzahlen_fig2.py` → `2026-10-29_M_textzahlen_fig2/`. Only the aggregated CSVs that feed the paper are included, plus the epoch logs and weight snapshots of the example run (dataset 3, K = 6, repetition 0) in `2026-10-20_fig2_example_run_ds3/` and one stored fit for the single re-fit check (`2026-10-20_SG_explore_H0/reference_fit.csv`). The per-fit outputs of the full runs are not included; the evaluation scripts need them and run after a full re-run (commands in the notebook, about 18 h on 4 cores). `run_tn.py --skip-criteria-lock` is needed for `config_sig_confirm_s4.yaml`, whose lock refers to a commit of the private development repository.

## Expectations E1–E5, fixed in advance

Before datasets 9–15 were run, five expectations for the sigmoid (β = 10) were fixed in `experiments/TN_ksweep/predictions_sg7.yaml`. The file is reproduced here verbatim (German keys; `Identitaet` = identity, `erfuellt_wenn` = met if, `plaene` = learning-rate schedules, `ebene` = level):

```yaml
# Versuchsplan SG7 · Erwartungen E1–E5, festgelegt VOR dem Lauf SG_seeds_H0s9-15 (Commit vor dem Hub-Start).
# Quelle: beschreibende Befunde aus Seeds 3–8 (SG6-Auswertung im Chat, Auftrag SG7 Teil 1).
status: festgelegt
festgelegt_am: 2026-10-20
xi: 10                         # im Wiki nu = 10
mittel: "alle Groessen je Seed: Mittel ueber K {2, 6, 12} x 5 reps, gepaart je (K, rep), Test-BA, xi = 10"
erwartungen:
  E1_G_gewinn:
    groesse: "BA_end(Sigmoid) - BA_end(Identitaet)"
    plaene: [const_0.003]
    ebene: G
    erfuellt_wenn: ">= +0.02"
  E2_CW_gewinn:
    groesse: "BA_end(Sigmoid) - BA_end(Identitaet)"
    plaene: [const_0.003]
    ebene: CW
    erfuellt_wenn: "> 0"
  E3_sigmoid_ersetzt_fruehstopp:
    groesse: "BA_end(Sigmoid) - BA_best_val(Identitaet)"
    plaene: [const_0.003, const_0.01]
    ebene: {G: ">= 0", CW: ">= -0.01"}
    erfuellt_wenn: "je Plan"
  E4_stehenbleiben:
    groesse: "BA_best_val(Sigmoid) - BA_end(Sigmoid)"
    plaene: [const_0.003, const_0.01]
    ebene: [G, CW]
    erfuellt_wenn: "<= 0.015"
  E5_keine_ecke:
    groesse: "Anteil Fits mit max alpha^2 > .99 (Sigmoid, mind. eine Klasse)"
    plaene: [const_0.003, const_0.01, decay_0.003_t133]
    ebene: [G, CW]
    erfuellt_wenn: "<= 0.20"
bewertung:
  seeds: [9, 10, 11, 12, 13, 14, 15]
  je_erwartung: "Anzahl Seeds (von 7), auf denen sie erfuellt ist (bei mehreren Plaenen/Ebenen: alle Zellen der Erwartung erfuellt)"
  gestuetzt: "auf >= 6 von 7 Seeds erfuellt; sonst nicht gestuetzt"
  toleranz: 1.0e-9             # Gleitkomma-Toleranz wie in SG6
  zusaetzlich_zeigen: [3, 4, 5, 6, 7, 8]   # Herkunft der Erwartungen: getrennt markieren, nicht in die Bewertung einrechnen
ohne_erwartung:
  - "CW-End-Gewinn-Verteilung ueber alle 13 Seeds"
  - "Luecke Train - Val (Sigmoid, Identitaet, frozen)"
  - "Zerlegung Strang A je Seed (frozen -> + Gewichte, Identitaet End / Identitaet Best_val / Sigmoid End, frozen_oracle_CW als Bezug)"
  - "Anteil echte Ecke mit Identitaet (B1)"
```

Each quantity is a mean per dataset over K ∈ {2, 6, 12} × 5 repetitions, paired per (K, repetition), test BA. An expectation counts as supported if it is met on at least 6 of the 7 confirmation datasets (all cells of the expectation met, tolerance 1e-9).

| Expectation | Met on datasets 9–15 | Verdict | Datasets 16 / 17 (descriptive) |
|---|---|---|---|
| E1 (G gain) | 7/7 | supported | met / met |
| E2 (CW gain) | 7/7 | supported | met / met |
| E3 (sigmoid replaces early stopping) | 7/7 | supported | met / not met (CW) |
| E4 (stays at its best validation state) | 7/7 | supported | met / not met (CW) |
| E5 (no corner) | 7/7 | supported | met / met |

Sources: `results/public/2026-10-20_SG7_seeds3-15/bewertung_sg7.csv` (verdict) and `erwartungen_sg7.csv` (seeds 3–15), `results/public/2026-10-20_SG9_seeds3-17/e1e5_seeds9-17.csv` (seeds 9–17 with values per cell).

Order of events: in the private development repository, the expectations were committed in `80e0b2b` (2026-10-19, 16:58 UTC) and the results of datasets 9–15 in `0452f22` (2026-10-20, 05:48 UTC). The run summaries of datasets 9–15 record `80e0b2b` as the code version they were computed with. The field `festgelegt_am` in the YAML file states 2026-10-20; the commit times above are authoritative. The development history itself is not published; this repository starts without history.

## Definitions

- **Corner**: at the end of training (epoch T = 400), max_v α_v² > .99 in at least one class (`ECKE_ECHT = 0.99` in `experiments/TN_ksweep/eval_sigmoid.py`, function `per_fit`).
- **Best validation epoch (Best_val)**: the first epoch with maximal validation BA (`argmax` over the logged validation BA, `run_tn.run_fit`).
- **Oracle weights**: inverse within-cluster spread on the training data (`run_tn.build_data`, `W_TRUE`). For cluster r and view v, q_{r,v} is the mean squared within-cluster dissimilarity (normalized dissimilarity matrices, pairs i ≠ j); α_v² ∝ 1 / q̄_v, normalized to sum one over the four views, with q̄ averaged over all clusters (G) or over the clusters of class c (CW, one vector per class). The oracle is a reference, not an upper bound.
- **Path length**: R = Σ_t η_t; every weight step has length η.

## Dependency proto-dist-ml

The initial prototypes are computed with relational neural gas (Hammer and Hasenfuss, KI 2007) from [proto-dist-ml](https://gitlab.ub.uni-bielefeld.de/bpaassen/proto-dist-ml) by Benjamin Paaßen (version 1.0.1 from PyPI, module `proto_dist_ml.rng`, imported in `src/common/proto_init.py`). proto-dist-ml is licensed under the GNU General Public License v3 or later. It is installed as a dependency via `requirements.txt` and is not included in this repository. The models in `src/models/` are based on and extend the MGLVQ code from proto-dist-ml.

## Use of AI

An AI assistant supported code development and language editing; the authors verified all results and take full responsibility.

## License

Code: GNU General Public License v3.0 (`LICENSE`), because the models in `src/models/` are derived from the MGLVQ code in proto-dist-ml by Benjamin Paaßen, which is licensed under GPL-3.0-or-later. Results in `results/`: Creative Commons Attribution 4.0 International (CC BY 4.0, `results/LICENSE.md`).
