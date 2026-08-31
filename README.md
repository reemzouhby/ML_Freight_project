# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts `posted_rate` for 12,000 unlabelled freight loads (Nov–Dec 2025) from
48,000 labelled loads (Jan–Oct 2025), plus a fixed 31-day December scenario.

**Result:** 1.61% mean absolute percentage error on held-out non-corrupted rows,
against 3.93% for a specification without temporal features. Validated on four
chronological folds.

## Approach in one paragraph

Rate is built multiplicatively — a power law in distance (exponent 0.871), times
an equipment premium, times a market-index factor — so the model is additive in
log space. Two data defects were repaired: 292 sign-flipped `weight` values, and
1.41% of `posted_rate` labels corrupted by random multipliers of ×2 to ×5.5 and
their reciprocals (detected via an empty band in the log-residual distribution,
not an arbitrary percentile). The decisive finding is temporal: rates follow a
**quarterly cycle**, flat for eight weeks then ramping ~5% into quarter-end.
Encoding time as position-within-quarter rather than calendar date makes the
Nov–Dec horizon reachable: the within-quarter shape is interpolated, since
day-of-quarter 61–91 is densely observed in Q1–Q3, and only the level drift is
extrapolated — along a single monotone dimension, anchored by October. The model is an additive hybrid: OLS
carries the parts that must extrapolate (elasticities, geography, trend, cubic
ramp), and gradient boosting fits its residual for local non-linearity.

Full detail, including the validation design and rejected alternatives, is in
[`report/report.pdf`](report/report.pdf).

## Layout

```
data/                            input CSVs (as supplied)
src/pipeline.py                  the full solution, runnable end to end
notebooks/exploration.ipynb      step-by-step EDA and model selection
report/report.tex                report source
report/report.pdf                report
score.py                         provided scorer (unmodified)
validation_predictions.csv       submission file
```

## Run

```bash
python -m pip install -r requirements.txt
python src/pipeline.py --data-dir data --out-dir .
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Expected output:

```
training on 47,323 of 48,000 rows (677 corrupted targets excluded)
wrote validation_predictions.csv  ($196-$7082)
filled december_chart_inputs.csv  Dec 1 $828.05 -> Dec 31 $879.93 (+6.3%)
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

Runs in about 30 seconds on a laptop. No GPU, no external services.

## Validation

Four strictly chronological folds — a random split would let same-period rows
predict each other and produce a meaningless estimate.

| Fold | Train | Predict | Tests |
|---|---|---|---|
| A | Jan 1 – Apr 30 | May 1 – Jun 30 | a general 61-day horizon |
| B | Jan 1 – Jun 30 | Jul 1 – Aug 31 | crossing a quarter boundary |
| C | Jan 1 – Aug 31 | Sep 1 – Oct 31 | primary mirror of the scored task |
| D | Jan 1 – Jul 31 | Sep 1 – Sep 30 | **the December analogue** |

Fold D reproduces the geometry of the scored task inside labelled data: train
through July (Q3 days 0–30 observed), skip August, predict September (Q3 days
62–91) — the same shape as observing October and predicting December.

Corrupted rows are removed from training only, never from a held-out fold: the
grading scorer sees all 12,000 rows, so cleaning the evaluation side would
produce a number that cannot be reproduced.

Clean-row MAPE by fold:

| Model | A | B | C | D | mean |
|---|---|---|---|---|---|
| No temporal features | 2.76% | 2.49% | 4.64% | 5.84% | 3.93% |
| Linear + temporal features | 2.12% | 2.38% | 2.17% | 2.21% | 2.22% |
| Gradient boosting alone | 2.32% | 2.30% | 2.08% | 1.73% | 2.11% |
| **Additive hybrid (submitted)** | **1.43%** | **1.82%** | **1.56%** | **1.61%** | **1.61%** |

RMSE was rejected as a selection metric: a *perfect* model scores ≈$657 against
corrupted targets, and across simulated models of increasing error RMSE came out
non-monotonic ($657, $761, $631, $690). Observed fold RMSE varied 4% across
architectures whose clean MAE differed by a factor of 2.4.

## Notes

- `quote_signal` was tested and dropped — 0.3% in-sample gain, worse on three of
  four folds. This also removes the last dependency on a column the December
  scenario file lacks.
- Eight cities appear only in the validation set (12% of scored rows), so
  geography enters through coordinates rather than city labels.
- The daily market-index table is built from all feature rows including
  validation. No target values enter it, and `market_index` is supplied for
  every row requiring a prediction, so this is information available at
  prediction time rather than leakage. The outlier rule was refitted inside a
  training fold as a check: identical coefficients to three decimals, same 677
  rows flagged, zero disagreements.
