from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alerts import ALERT_COLUMNS, ALERTS_URL, select_alerts
from src.config import INTERIM, RAW, cleaned_path
from src.download import download_csv


def main():
    raw_path = RAW / 'alerts.csv'
    download_csv(ALERTS_URL, raw_path, ALERT_COLUMNS, filename=raw_path.name)
    alerts = pd.read_csv(raw_path)
    selected = select_alerts(alerts)
    print(f'Before: {len(alerts)}, selected: {len(selected)}')
    print(selected.groupby(['y', 'Class']).size())
    classes = alerts.rename(columns=str.strip)['Class'].astype('string').str.strip()
    print('Excluded classes:')
    print(classes.loc[~classes.isin(selected['Class'])].value_counts(dropna=False))

    # Keep the existing review step; do not silently repair labels or duplicates.
    review_columns = ['#Name', 'Class', 'Comment', 'TNSid']
    for label, title in [(1, 'CV'), (0, 'Non-CV')]:
        print(f'{title} label examples:')
        print(selected.loc[selected['y'] == label, review_columns].head(20))
    print('Repeated alert names:')
    print(selected.loc[selected.duplicated('#Name', keep=False), review_columns])
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = cleaned_path(INTERIM / 'selected_alerts.csv')
    selected.to_csv(path, index=False)
    print(f'Saved: {path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
