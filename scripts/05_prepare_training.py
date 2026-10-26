"""Freeze object groups and the shared training/validation/test split."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM, PROCESSED
from src.splitting import make_manifest


def main():
    frame = pd.read_csv(INTERIM / 'cleaned_features_with_dr3.csv', dtype={'source_id': 'string'})
    manifest = make_manifest(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(PROCESSED / 'split_manifest.csv', index=False)
    print(manifest.groupby('split').agg(n_samples=('y', 'size'), n_cv=('y', 'sum'),
                                       n_groups=('group_id', 'nunique')))
    print('Related alerts:')
    print(manifest.loc[manifest['group_id'].duplicated(keep=False), ['#Name', 'group_id', 'split']])


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
