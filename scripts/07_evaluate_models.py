"""Compare validation models and evaluate the locked selected model on test data."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM, MODELS, PROCESSED, REPORTS
from src.evaluation import evaluate_models
from src.splitting import attach_manifest


def main():
    frame = pd.read_csv(INTERIM / 'cleaned_features_with_dr3.csv', dtype={'source_id': 'string'})
    manifest = pd.read_csv(PROCESSED / 'split_manifest.csv')
    frame = attach_manifest(frame, manifest)
    lightcurves = pd.read_csv(INTERIM / 'cleaned_lightcurves.csv')
    metrics = evaluate_models(
        frame, models_dir=MODELS, processed_dir=PROCESSED,
        reports_dir=REPORTS, lightcurves=lightcurves)
    print(metrics.loc[metrics['split'].eq('test'), ['model', 'precision', 'recall', 'f1', 'ap', 'roc_auc']])
    print(f'Saved object-level predictions and error details: {PROCESSED}')
    print(f'Saved report tables and figures: {REPORTS}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
