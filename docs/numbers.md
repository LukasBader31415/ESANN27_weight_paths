# Numbers in the paper and where they come from

All paths below are relative to `results/public/`. Abbreviations for the files:

| Short | File |
|---|---|
| T1 | `2026-10-20_SG9_seeds3-17/tabelle1.csv` (rows with `seeds` = `3–17`, n = 15) |
| T1s | `2026-10-20_SG9_seeds3-17/tabelle1_seeds.csv` (one row per dataset and level) |
| KV | `2026-10-20_SG9_seeds3-17/kverlauf_s3-5.csv` (datasets 3, 4, 5; K 2–20) |
| E9 | `2026-10-20_SG9_seeds3-17/e1e5_seeds9-17.csv` |
| BW | `2026-10-20_SG7_seeds3-15/bewertung_sg7.csv` |
| KS | `2026-10-20_SG7_seeds3-15/kennzahlen_seeds.csv` |
| TZ | `2026-10-29_M_textzahlen_fig2/zahlen.csv` (columns `ebene` = level, `lauf` = run, `groesse` = quantity, `wert` = value) |
| KB | `2026-10-29_M_textzahlen_fig2/sg9_kosten_ba_fits.csv` (one row per identity fit) |

In T1, `groesse` names the quantity: `plus_id_end` = identity at the end of training, `plus_id_best_val` = identity at the best validation epoch, `plus_sig_end` = sigmoid (β = 10) at the end of training, `plus_orakel` = oracle weights; all as Δ test BA over *frozen*. Column `mittel` = mean over the datasets, `n_ueber_frozen` = number of datasets with Δ > 0, `n_ge_orakel` = number of datasets at or above the oracle weights. The rows *sigmoid wins* are computed from T1s by `src/paper_figs/table.py`, function `sigmoid_wins` (per dataset, strictly greater, at or above for the oracle column). The notebook `notebooks/reproduce_paper.ipynb` prints all values below.

## Abstract and introduction

| Location | Number | File | Column / row |
|---|---|---|---|
| Abstract, Intro (contribution 3) | sigmoid better than identity at the end on all 15 datasets | T1s | `sig_end` > `id_end` per `seed`, both levels (15/15 each; also Table 1, *sigmoid wins*, column *identity end*) |
| Intro (contribution 2) | training cost keeps decreasing after the best validation epoch in all runs; training accuracy falls in most | KB | `kosten_sinkt` (225/225 per level), `trainba_sinkt` (G 220/225, CW 150/225) |

## Section 2, method

| Location | Number | File | Column / row |
|---|---|---|---|
| Text | equal weights, relational neural gas initialization, scaling by the training maximum | code | equal weights α_v² = 1/V: `src/models/*.py` (`_vWeights`, default without `v_init`); initial prototypes: `src/common/proto_init.py` (`init_class_prototypes`), `experiments/TN_ksweep/run_tn.py` (`common_init`); scaling: `src/toy/toy_heimatview.py` (`make_D_blocks`), `run_tn.build_data` (`scale`); see README, *Initialization and scaling* |

## Section 3, setup

| Location | Number | File | Column / row |
|---|---|---|---|
| Data | 6 clusters per class, 50 points, 4 views, relevance 0.8 / 0.1 / 0.1, noise view | generator | `src/toy/toy_heimatview.py` (`DEFAULTS`, plan `H0`), configs: `w_heim: 0.8` |
| Data | 15 datasets; 600 training, 600 validation, 3000 test points | configs | `data_seed` 3–17; `val: n_per_cluster 50`, `test: n_per_cluster 250` (12 clusters) |
| Protocol | five repetitions, T = 400, η = 0.003 | configs | `reps: [0..4]`, `T: 400`, `eta: 0.003` |
| Protocol | six exploration, three for the K sweep, seven confirmation, two descriptive datasets | README | section *Datasets and roles* |
| Protocol | K ∈ {2, 6, 12}; K = 2, …, 20 | configs | `K` in `config_sig_s*.yaml` and `config_sg9_k_s*.yaml` |

## Section 4.1, decomposition (Table 1)

| Location | Number | File | Column / row |
|---|---|---|---|
| Text, Table 1 caption | prototype learning +.060 | T1 | `level` = `–`, `groesse` = `plus_prototypen`, `mittel` |
| Table 1 caption | start BA .695 | T1 | `level` = `–`, `groesse` = `start`, `mittel` |
| Text | G identity end −.012, above *frozen* on 4 of 15 | T1 | `G`, `plus_id_end`: `mittel`, `n_ueber_frozen` |
| Text | CW identity end +.045, 15/15 | T1 | `CW`, `plus_id_end`: `mittel`, `n_ueber_frozen` |
| Text | best validation epoch: G +.036, CW +.068, each 15 of 15 | T1 | `plus_id_best_val`: `mittel`, `n_ueber_frozen` |
| Text | oracle weights G +.031, CW +.068 | T1 | `plus_orakel`, `mittel` |
| Text | G matches or exceeds the oracle on 12 of 15, CW on 10 of 15 | T1 | `plus_id_best_val`, `n_ge_orakel` |
| Text | CW gain at the best validation epoch exceeds the prototype gain | T1 | CW `plus_id_best_val` (.068) > `plus_prototypen` (.060) |
| Text | CW share shrinks as K increases | KV | `plus_id_end`, `level` = `CW`, over `K` (mean over `seed` 3–5) |
| Fig. 1 caption | mean over three datasets; range over the datasets | KV | `seed` 3, 4, 5; mean, min, max per `level` and `K` (`src/paper_figs/data.py`, `kverlauf_start`) |

## Section 4.2, weight path (Fig. 2)

| Location | Number | File | Column / row |
|---|---|---|---|
| Text, Fig. 2 caption | example run K = 6, η = 0.003 (dataset 3, repetition 0) | – | `src/paper_figs/data.py` (`RUN_S3`), `figures.py` (`fig_pfad2`) |
| Text | other learning rates: deviation below 2° | TZ | `groesse` = `größte Winkelabweichung zwischen den Pfaden, R ≤ 1.25 (Grad)`, `lauf` with `16 Identitäts-Pläne`: G 1.75, CW 1.48 |
| Text | training BA .817 to .757 (G) | TZ | `lauf` = `DS 3, G_const, K 6, rep 0`, `groesse` = `Train-BA / Val-BA bei Best_val` and `… bei Ende` (first value) |
| Text | training BA .855 to .815 (CW) | TZ | same with `DS 3, CW_const, K 6, rep 0` |
| Text | before: points near the boundary lower the cost; after: points correct with a margin | TZ | `groesse` = `Kostenbeitrag ΔΣμ vor / nach Best_val: …` (sicher richtig, Grenzbereich, falsch), G and CW |
| Text | 70 (G) and 60 (CW) training points flip from correct to wrong | TZ | `groesse` = `Trainingspunkte richtig → falsch (Best_val → Ende)` |
| Text | 225 identity runs per variant | KB | rows per `level`: 15 datasets × K {2, 6, 12} × 5 repetitions |
| Text | cost at the end lower than at the best validation epoch in all runs | KB | `kosten_sinkt`: 225/225 (G and CW); also TZ `Anteil Σμ(Ende) < Σμ(Best_val)` |
| Text | training BA drops in 220 of 225 (G), 150 of 225 (CW) | KB | `trainba_sinkt` sum per `level`; also TZ `Anteil Train-BA(Ende) < Train-BA(Best_val)` |
| Text | highly dataset-dependent (CW) | KB | `trainba_sinkt` per dataset (`sig` = cache signature of the dataset), `level` = `CW` |
| Fig. 2 caption | corner: max. α_v² > .99 | code | `experiments/TN_ksweep/eval_sigmoid.py`, `ECKE_ECHT = 0.99` |

## Section 4.3, sigmoid

| Location | Number | File | Column / row |
|---|---|---|---|
| Text | G final angle 25.5° (sigmoid) instead of 53.1° (identity) | TZ | `groesse` = `Winkel am Ende, Grad (Epoche 400, R 1.200)`, `lauf` = `DS 3, G_sig10_e0.003_const, K 6, rep 0` and `DS 3, G_const, K 6, rep 0` |
| Text | CW does not stop but slows down | TZ | `groesse` = `Netto-Drehung ∠(a(T−50), a(T)) …` and `Halt-R: …`, CW runs |
| Text | 13 exploration and confirmation datasets: identity in a corner CW 97 %, G 65 %, sigmoid in none | KS | `plan` = `e0.003_const`, `seed` 3–15: mean of `ecke_echt_id` per `level` (CW .974, G .646); `ecke_echt` = 0 for all |
| Text, Table 1 | sigmoid vs. *frozen* G +.053, CW +.073 | T1 | `plus_sig_end`, `mittel` |
| Text, Table 1 | sigmoid above *frozen* and identity end on all 15 datasets | T1s | `sig_end` > `frozen` and > `id_end` per `seed` (15/15 per level) |
| Text, Table 1 | G: above identity best val. and oracle on all datasets | T1s | `sig_end` > `id_best_val`, `sig_end` ≥ `orakel` per `seed` (G 15/15) |
| Text, Table 1 | CW: 11 and 12 of 15 | T1s | same comparisons for `CW` (11/15, 12/15) |
| Text | example dataset: CW sigmoid end −.003 below the early-stopped identity | T1s | `seed` = 3, `level` = `CW`: `sig_end` − `id_best_val` (.834 − .836) |
| Text | all five expectations met on all seven confirmation datasets | BW | `erfuellt_seeds_9_15` = 7/7 for E1–E5 |
| Text | one descriptive dataset misses two expectations for CW | E9 | `seed` = 17: `E3`, `E4` = False (values in `E3_werte`, `E4_werte`, CW cells); `seed` = 16: all True |
| Text | CW sigmoid falls below identity best val. and oracle at large K, stays above *frozen* and identity end | KV | `level` = `CW`: `sig_end` vs. `id_best_val`, `orakel`, `frozen`, `id_end` over `K` (Fig. 1, diamonds) |
| Table 1 | all cells | T1, T1s | see Section 4.1 and the rows *sigmoid wins* above |

## Section 5, discussion

| Location | Number | File | Column / row |
|---|---|---|---|
| Limitations | β = 10 chosen on the exploration datasets | `experiments/TN_ksweep/criteria_sg.yaml` | `xi_bestaetigung: 10` (mean validation BA at the end: β 5 .815, β 10 .828, β 20 .826, dataset 3) |
| Limitations | train–test gap widens for CW with the sigmoid at large K | – | not in the aggregated files of this repository (see note below) |

Note: the train–test gap at large K (Section 5) is a qualitative statement in the paper. It is not derived from the files in `results/public/` of this repository.
