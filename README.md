# Freight Rate Prediction

My solution for the Spotter.AI machine learning engineer assessment. The task is to predict a price (`posted_rate`) for truck loads using past load data, then output predictions for 12,000 new loads plus a fixed 31-day December lane.

## Data

All input files are in `Data/` (as provided):

| File | Rows | What it is |
| --- | --- | --- |
| `train-test.csv` | 48,000 | Labeled loads, Jan-Oct 2025. Has the target `posted_rate`. |
| `validation.csv` | 12,000 | Unlabeled loads, Nov-Dec 2025. IDs `TE-000001` to `TE-012000`. |
| `validation-predictions-template.csv` | 12,000 | Template with the `load_id`s to fill in. |
| `december-chart-inputs.csv` | 31 | Fixed lane Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb. One row per day in Dec 2025. |
| `december_chart_inputs.csv` | 31 | Same content as above, underscore name used by the scorer example. |

Target stats in training: mean ~$2,374, median ~$2,031, std ~$1,486. Right-skewed, with a small tail above $12k.

## What I did

**Cleaning.** Negative weights (292 train / 145 validation) looked like sign-entry mistakes, so I took absolute values. Missing weights were filled with the per-equipment median, missing `market_index` with the training median. Every imputation value is computed on training data only.

**Features (28).** Haversine distance from pickup/delivery coordinates, road-vs-straight-line gap, date parts (weekday, month, week of year, weekend flag), log transforms, `weight_per_mile`, `distance x weight`, market/quote interactions (`distance x market`, `market x quote`, `distance x quote`), route popularity, smoothed target encodings for pickup/delivery/route (smoothing 100, fitted on train only), and equipment one-hot. ID columns, raw city names, raw coordinates and dates are dropped after encoding.

**Validation.** Time-based splits, since the real test is future data: Jan-Aug trains and Sep-Oct validates (the model-selection fold), plus two rolling checks (Jan-Jun -> Jul, Jan-Jul -> Aug). Random splitting would leak future prices into training, so I avoided it. I also compared raw vs `log1p` targets for every model.

**Models.** LightGBM, XGBoost and CatBoost with comparable budgets (~1500 trees, lr 0.03, depth 7), plus median / mean / $/mile baselines. Best was **CatBoost on log1p target**: MAE **$130.30**, RMSE $639.63, R2 0.8243 on the Sep-Oct fold. It also held up best on the rolling folds ($154.25 / $143.53 vs higher for the others), so I retrained it on all 48k rows for the final predictions.

**December lane.** That file has no coordinates, market or quote values, so I used training medians for the Lexington/Fort Wayne coordinates and the Sep-Oct trailing median for market and Dry Van quote. Only the date features vary across the 31 rows. Range came out $809.61-$833.66.

## How to run

```bash
python -m pip install -r requirements.txt
python train.py
python score.py --predictions validation_predictions.csv --december-predictions Data/december_chart_inputs.csv
```

`train.py` writes `validation_predictions.csv`, fills both December files, and saves `outputs/feature_importance.png`. `score.py` (provided scorer) validates both files and creates `scorer_results/candidate_december.png`.

## Files

- `train.py` - the full pipeline: clean, features, validation, model comparison, final predictions
- `freight_rate_prediction.ipynb` - full walkthrough of the same pipeline (run from repo root)
- `score.py` - provided validation + December chart script, unchanged
- `validation_predictions.csv` - final predictions for the 12,000 loads
- `Data/` - input data (see table above)
- `outputs/` - EDA charts and feature importance
- `freight-rate-ml-assessment.pdf` - the original assessment brief

## Submission notes

- Report (DOCX with split approach and the December chart) and the 2-3 min Loom walkthrough are submitted separately, not kept in this repo.
- Loom link: _to be added_
