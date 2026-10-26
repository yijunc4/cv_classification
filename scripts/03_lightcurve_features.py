"""Extract nine light-curve features and attach the existing labels."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM, LC_FEATURES, cleaned_path
from src.features import extract_lightcurve_features


def main():
    lightcurves = pd.read_csv(INTERIM / 'cleaned_lightcurves.csv')
    selected = pd.read_csv(INTERIM / 'cleaned_selected_alerts.csv')
    features = extract_lightcurve_features(lightcurves)
    feature_table = features.merge(selected[['#Name', 'Class', 'y']], on='#Name', how='left')
    print(feature_table['y'].value_counts().rename(index={1: 'CV', 0: 'non-CV'}))
    print('Missing feature values:')
    print(feature_table[LC_FEATURES].isna().sum())
    path = cleaned_path(INTERIM / 'lightcurve_features.csv')
    feature_table.to_csv(path, index=False)
    print(f'Saved: {path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
