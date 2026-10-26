# CV Classification and Candidate Screening

A laptop-scale pipeline inspired by [Mistry et al. (2022)](https://doi.org/10.1093/mnras/stac2760) for classifying Gaia
Science Alerts as cataclysmic variables (CVs) or non-CVs. The current pipeline
prepares nine light-curve features and four Gaia DR3 metadata features, compares
Random Forest with LightGBM, and exports repeatable batch predictions.

## [Evaluation report](reports/evaluation.ipynb)

View the four-model validation comparison, permutation importance, the selected
model's test results, and representative error light curves. Saved tables and
figures can be viewed without running the pipeline. The report notebook reads
the saved report tables and figures; it does not load models or recompute evaluation.

## Reproduce the included snapshot

The repository includes the frozen feature table, cleaned light curves, split
manifest, six fitted model bundles, and `selected_model.json`. To evaluate the
saved models without downloading data or refitting:

```sh
python -m pip install -r requirements.txt
python scripts/07_evaluate_models.py
```

To retrain on the same snapshot, run `python scripts/06_train_models.py` before
evaluation. Reuse the included split manifest; step 05 is only needed when
preparing a new snapshot.

## Run the data pipeline

To rebuild the data from its upstream sources, run these steps in order.
Tested with Python 3.12:

```sh
python -m pip install -r requirements.txt
python scripts/01_download_alerts.py
python scripts/02_lightcurves.py
python scripts/03_lightcurve_features.py
python scripts/04_gaia_dr3.py
```

Scripts orchestrate the workflow; reusable functions live in `src`. Paths are
relative to the project location. No command-line arguments are required.

Data sources and credits are listed in
[data/raw/acknowledgement.md](data/raw/acknowledgement.md). Raw downloads stay
local; the repository includes the cleaned snapshot used for the reported results.

## Download and cleaning rules

- Downloads use exponential backoff, jitter, and at most two retries after the
  initial attempt. HTTP failures are retried only when temporary.
- Files are saved through a sibling `.part` file. Before renaming, validation
  checks only the filename, optional title, column names, and column counts.
  Cached downloads receive the same structure checks.
- Failed light-curve downloads are recorded in
  `data/interim/lightcurve_download_errors.csv`. The cleaning stage waits until
  all selected files have passed the download-stage structure checks.
- Gaia queries search only within 2 arcsec of the selected coordinates.
  Result retries reuse the same TAP job. Ambiguous and unmatched sources stay
  in the sample with missing DR3 metadata.
- Cleaned files keep their complete original filename with a `cleaned_` prefix.
  The final table is `data/interim/cleaned_features_with_dr3.csv`.
- Missing feature values are retained for training-time imputation. Labels
  come from the selected Gaia Science Alerts `Class` values: CV is positive;
  the retained non-CV classes are negative.

## Train and evaluate

After preparing the frozen feature table, run:

```sh
python scripts/05_prepare_training.py
python scripts/06_train_models.py
python scripts/07_evaluate_models.py
```

The split manifest in `data/processed/split_manifest.csv` retains every feature
row. Connected groups share an accepted DR3 source ID or have alert coordinates
within 2 arcsec. All members stay in the same split. With seed 42, five stratified
group folds assign fold 0 to testing, fold 1 to validation, and folds 2-4 to
training. Three further stratified group folds support training-only searches.
Treat the feature table and manifest as one frozen snapshot throughout the run.

Four formal experiments use identical objects and splits: RF and LightGBM,
each with nine light-curve features or all thirteen features. A median imputer
inside each Pipeline is fitted only on training rows. No scaling, extra missing
indicators, or complete-case filtering is applied to the formal experiments.

Each search evaluates eight parameter settings using mean cross-validation AP:

| Model | Fixed settings | Search grid |
| --- | --- | --- |
| Random Forest | 300 trees, `max_features='sqrt'` | depth: 5/10; minimum leaf size: 1/5; class weight: None/balanced |
| LightGBM | 300 trees, learning rate 0.05, depth 5 | leaves: 7/15; minimum child samples: 20/50; `is_unbalance`: False/True |

Searches run sequentially; estimators use at most four CPU threads. Each final
training fit receives a validation F1-maximizing threshold, breaking ties by
precision and then higher threshold. The recommended formal model is chosen by
validation AP, then validation F1. Models and thresholds are locked before
testing; there is no subsequent training-plus-validation refit.

`DummyClassifier(strategy='prior')` supplies a reference. A separate depth-three
decision tree uses only match status and four metadata-missing flags to diagnose
catalogue coverage effects. It is not part of the thirteen-feature experiment.

## Outputs and interpretation

- `models/`: six fitted Pipeline bundles and `selected_model.json`. Each bundle
  includes ordered input columns, threshold, parameters, seed and library versions.
- `data/interim/`: the published `cleaned_features_with_dr3.csv` and
  `cleaned_lightcurves.csv`, plus local intermediate files generated by the data pipeline.
- `data/processed/`: the published `split_manifest.csv` and local calculation
  records: `cv_results.csv`, `validation_predictions.csv`, `test_predictions.csv`,
  and batch `predictions.csv`. `representative_errors.csv` lists up to ten
  highest-scoring false positives and ten lowest-scoring false negatives.
- `reports/`: public summary tables and figures. `metrics.csv` is the single
  performance table: it contains validation results for all six models and the
  selected formal model's test result. `permutation_importance.csv` contains
  validation feature-importance summaries. All report PNGs, including
  `error_lightcurves.png`, are saved here only. Validation figures compare the
  four formal models; test figures show the selected model.
- [reports/evaluation.ipynb](reports/evaluation.ipynb): the main report. Its code
  cells only read `metrics.csv` and display tables; figures use saved PNGs.
  Narrative numbers are maintained manually and should be checked when results change.

Run `python scripts/07_evaluate_models.py` to regenerate evaluation details and
all report tables and figures together from the frozen data and saved models.
Then rerun the report notebook to refresh its displayed tables. The notebook
does not write evaluation files or require local datasets and fitted models.

CV is the positive class. Precision, recall and F1 use the locked threshold;
AP is **average precision**, not trapezoidal PR-AUC. ROC-AUC evaluates ranking.
Permutation importance uses validation AP, ten repeats and seed 42; it does not
trigger feature reselection on the test set.

Results describe this labelled alert snapshot. Catalogue availability, alert
selection and sparse light curves can affect performance. Model probabilities
are not calibrated. A higher score on one split does not establish general
superiority, and repeated tuning against the same test set invalidates its role
as a held-out evaluation.

## Batch predictions

Create `data/interim/prediction_features.csv` containing `#Name` and the saved
model's required features, computed with the same definitions and cleaning
rules. Then run:

```sh
python scripts/08_predict.py
```

This loads the recommended bundle without fitting anything and writes
`data/processed/predictions.csv` with object name, `p_cv`, thresholded prediction
and model identifier. Only load trusted joblib bundles. Predictions are not labels
for further evaluation.

