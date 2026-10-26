"""Small grouped searches, validation thresholds, and reproducible model bundles."""

from importlib.metadata import version
import json

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import DR3_FEATURES, LC_FEATURES, SEED


def choose_threshold(y, probabilities):
    """Maximize CV F1 on validation data; break ties by precision then threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = max(range(len(thresholds)), key=lambda i: (f1[i], precision[i], thresholds[i]))
    return float(thresholds[best])


def diagnostic_features(frame):
    """Describe catalogue coverage separately from the formal feature sets."""
    result = pd.DataFrame(index=frame.index)
    for status in ['matched', 'unmatched', 'ambiguous']:
        result[f'status_{status}'] = frame['match_status'].eq(status).astype(int)
    for column in DR3_FEATURES:
        result[f'missing_{column}'] = frame[column].isna().astype(int)
    return result


def bundle_features(frame, bundle):
    if bundle['kind'] == 'diagnostic':
        return diagnostic_features(frame)[bundle['features']]
    return frame[bundle['features']]


def predict_bundle(frame, bundle):
    """Predict using the saved column order, imputer, model, and threshold."""
    probability = bundle['pipeline'].predict_proba(bundle_features(frame, bundle))[:, 1]
    return pd.DataFrame({
        '#Name': frame['#Name'].to_numpy(), 'p_cv': probability,
        'prediction': (probability >= bundle['threshold']).astype(int),
        'model': bundle['name'],
    })


def model_spec(family):
    if family == 'rf':
        estimator = RandomForestClassifier(
            n_estimators=300, max_features='sqrt', random_state=SEED, n_jobs=4)
        grid = {'model__max_depth': [5, 10], 'model__min_samples_leaf': [1, 5],
                'model__class_weight': [None, 'balanced']}
    else:
        estimator = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=SEED,
            n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
        grid = {'model__num_leaves': [7, 15], 'model__min_child_samples': [20, 50],
                'model__is_unbalance': [False, True]}
    return estimator, grid


def make_pipeline(estimator):
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', estimator)])


def check_training_features(features, folds):
    """Do not silently discard an entirely missing training feature."""
    for label, rows in [('full training set', np.arange(len(features)))]+[
            (f'CV fold {i}', train) for i, (train, _) in enumerate(folds)]:
        missing = features.iloc[rows].isna().all()
        if missing.any():
            raise ValueError(f'Entirely missing features in {label}: {list(missing[missing].index)}')


def fit_experiment(train, validation, columns, family):
    estimator, grid = model_spec(family)
    x = train[columns]
    folds = [(np.flatnonzero(train['cv_fold'].ne(i)), np.flatnonzero(train['cv_fold'].eq(i)))
             for i in range(3)]
    check_training_features(x, folds)
    search = GridSearchCV(make_pipeline(estimator), grid, scoring='average_precision',
                          cv=folds, n_jobs=1, refit=True, error_score='raise')
    search.fit(x, train['y'])
    bundle = make_bundle(f'{family}_{len(columns)}', search.best_estimator_, columns,
                         validation, search.best_params_, kind='formal')
    bundle['cv_ap'] = float(search.best_score_)
    results = pd.DataFrame(search.cv_results_)
    results = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].copy()
    results['params'] = results['params'].map(json.dumps)
    results.insert(0, 'model', bundle['name'])
    return bundle, results


def make_bundle(name, pipeline, columns, validation, parameters, kind):
    bundle = {'name': name, 'kind': kind, 'pipeline': pipeline, 'features': list(columns),
              'parameters': parameters, 'seed': SEED,
              'versions': {p: version(p) for p in ['scikit-learn', 'lightgbm', 'pandas', 'numpy', 'joblib']}}
    probabilities = pipeline.predict_proba(bundle_features(validation, bundle))[:, 1]
    bundle['threshold'] = choose_threshold(validation['y'], probabilities)
    bundle['validation_ap'] = float(average_precision_score(validation['y'], probabilities))
    return bundle


def train_models(frame, models_dir, processed_dir):
    """Fit only training rows; lock all thresholds using validation rows."""
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train = frame.loc[frame['split'].eq('train')]
    validation = frame.loc[frame['split'].eq('validation')]
    bundles, searches = [], []
    for columns in [LC_FEATURES, LC_FEATURES + DR3_FEATURES]:
        for family in ['rf', 'lgbm']:
            print(f'Training {family}, {len(columns)} features (8 settings x 3 folds)...', flush=True)
            bundle, results = fit_experiment(train, validation, columns, family)
            searches.append(results)
            bundles.append(bundle)
    dummy = make_pipeline(DummyClassifier(strategy='prior'))
    dummy.fit(train[LC_FEATURES], train['y'])
    bundles.append(make_bundle('dummy', dummy, LC_FEATURES, validation, {'strategy': 'prior'}, 'reference'))
    x_diagnostic = diagnostic_features(train)
    diagnostic = make_pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED))
    diagnostic.fit(x_diagnostic, train['y'])
    bundles.append(make_bundle('coverage_diagnostic', diagnostic, list(x_diagnostic), validation,
                               {'max_depth': 3, 'min_samples_leaf': 20}, 'diagnostic'))
    predictions = []
    for bundle in bundles:
        joblib.dump(bundle, models_dir / f"{bundle['name']}.joblib")
        prediction = predict_bundle(validation, bundle)
        prediction['y'] = validation['y'].to_numpy()
        predictions.append(prediction)
    pd.concat(searches, ignore_index=True).to_csv(processed_dir / 'cv_results.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(processed_dir / 'validation_predictions.csv', index=False)
    formal = [bundle for bundle in bundles if bundle['kind'] == 'formal']
    from_scores = []
    for bundle in formal:
        p = predict_bundle(validation, bundle)
        from_scores.append((bundle['validation_ap'], float(f1_score(validation['y'], p['prediction']))))
    best = max(range(len(formal)), key=lambda i: from_scores[i])
    selected = {'model': formal[best]['name'], 'selection': 'validation AP, then validation F1',
                'seed': SEED, 'features': formal[best]['features']}
    (models_dir / 'selected_model.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print(f"Locked recommended model: {selected['model']}", flush=True)
    return bundles
