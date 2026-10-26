"""Score a prepared feature CSV with the saved recommended model."""

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM, MODELS, PROCESSED
from src.modeling import predict_bundle


def main():
    selected = json.loads((MODELS / 'selected_model.json').read_text(encoding='utf-8'))
    bundle = joblib.load(MODELS / f"{selected['model']}.joblib")
    frame = pd.read_csv(INTERIM / 'prediction_features.csv')
    predictions = predict_bundle(frame, bundle)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / 'predictions.csv'
    predictions.to_csv(path, index=False)
    print(f'Saved {len(predictions)} predictions: {path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        raise SystemExit(1)
