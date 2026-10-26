"""Download validated light curves, combine them, and clean observations."""

import json
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM, RAW, cleaned_path
from src.download import download_csv
from src.lightcurves import LIGHTCURVE_COLUMNS, LIGHTCURVE_URL, clean_lightcurves, load_lightcurves


def main():
    selected = pd.read_csv(INTERIM / 'cleaned_selected_alerts.csv')
    names = selected['#Name'].drop_duplicates().tolist()
    directory = RAW / 'lightcurves'
    failed = []
    downloaded = skipped = 0
    error_path = INTERIM / 'lightcurve_download_errors.csv'
    with requests.Session() as session:
        for i, name in enumerate(names, 1):
            path = directory / f'{name}.csv'
            existed = path.exists()
            try:
                download_csv(LIGHTCURVE_URL.format(name=name), path, LIGHTCURVE_COLUMNS,
                             filename=f'{name}.csv', title=name, session=session)
            except Exception as e:
                print(f'{name}: {e}', flush=True)
                failed.append({'#Name': name, 'error': str(e)})
                pd.DataFrame(failed).to_csv(error_path, index=False)
            else:
                skipped += int(existed)
                downloaded += int(not existed)
            if i % 25 == 0 or i == len(names):
                print(f'{i}/{len(names)} | downloaded={downloaded}, skipped={skipped}, failed={len(failed)}', flush=True)
    pd.DataFrame(failed, columns=['#Name', 'error']).to_csv(error_path, index=False)
    if failed:
        raise RuntimeError('Some downloads or structure checks failed; cleaning was not run.')

    lightcurves = load_lightcurves(names, directory)
    cleaned, summary = clean_lightcurves(lightcurves)
    path = cleaned_path(INTERIM / 'lightcurves.csv')
    cleaned.to_csv(path, index=False)
    (INTERIM / 'lightcurve_cleaning_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved: {path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
