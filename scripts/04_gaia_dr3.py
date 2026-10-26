"""Query nearby Gaia DR3 candidates, match sources, and clean the feature table."""

import json
from pathlib import Path
import sys

from astropy.table import Table
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DR3_FEATURES, INTERIM, LC_FEATURES, RAW, cleaned_path
from src.download import save_csv_atomic
from src.features import clean_feature_table
from src.gaia import (DR3_CANDIDATE_COLUMNS, build_search_query, fetch_candidates,
                      match_candidates, prepare_targets, submit_search)
from src.validation import validate_csv


def main():
    lc_features = pd.read_csv(INTERIM / 'cleaned_lightcurve_features.csv')
    selected = pd.read_csv(INTERIM / 'cleaned_selected_alerts.csv')
    targets = prepare_targets(lc_features, selected)
    raw_path = RAW / 'gaia_dr3_candidates.csv'

    def validate(path):
        validate_csv(path, DR3_CANDIDATE_COLUMNS, filename=raw_path.name)

    if raw_path.exists():
        validate(raw_path)
        print(f'Skipped (already exists): {raw_path}')
    else:
        query = build_search_query(radius_arcsec=2.0)
        print(query)
        job = submit_search(query, Table.from_pandas(targets), max_retries=2)
        print(f'TAP job: {job.jobid}', flush=True)
        candidates = fetch_candidates(job, max_retries=2)
        save_csv_atomic(candidates, raw_path, validate=validate)

    candidates = pd.read_csv(raw_path, dtype={'source_id': 'string'})
    matches = match_candidates(targets, candidates)
    print(matches['match_status'].value_counts())
    feature_table = lc_features.merge(matches, on='#Name', how='left')
    feature_path = INTERIM / 'features_with_dr3.csv'
    matches.to_csv(INTERIM / 'gaia_dr3_matches.csv', index=False)
    feature_table.to_csv(feature_path, index=False)

    cleaned, summary = clean_feature_table(feature_table)
    path = cleaned_path(feature_path)
    cleaned.to_csv(path, index=False)
    feature_path.with_name(feature_path.stem + '_cleaning_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Missing feature values:')
    print(cleaned[LC_FEATURES + DR3_FEATURES].isna().sum())
    print(f'Saved: {path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
