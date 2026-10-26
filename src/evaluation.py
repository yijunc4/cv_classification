"""Evaluate locked classifiers, feature importance, and representative errors."""

import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix, precision_score, recall_score,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from src.config import DR3_FEATURES, SEED
from src.modeling import bundle_features, predict_bundle


def metric_row(y, probability, threshold):
    prediction = np.asarray(probability) >= threshold
    y = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        'n_samples': len(y), 'n_cv': int(y.sum()), 'cv_prevalence': float(y.mean()),
        'threshold': float(threshold),
        'precision': float(precision_score(y, prediction, zero_division=0)),
        'recall': float(recall_score(y, prediction, zero_division=0)),
        'f1': float(f1_score(y, prediction, zero_division=0)),
        'ap': float(average_precision_score(y, probability)) if y.sum() else None,
        'roc_auc': float(roc_auc_score(y, probability)) if both else None,
        'roc_auc_note': '' if both else 'Undefined: sample contains only one class.',
        'ap_note': '' if y.sum() else 'Undefined: sample has no positive examples.',
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def evaluate_predictions(frame, bundles, splits=('validation', 'test')):
    rows, predictions = [], []
    for split in splits:
        subset = frame.loc[frame['split'].eq(split)]
        for bundle in bundles:
            prediction = predict_bundle(subset, bundle)
            prediction['y'] = subset['y'].to_numpy()
            prediction['split'] = split
            predictions.append(prediction)
            row = metric_row(subset['y'], prediction['p_cv'], bundle['threshold'])
            rows.append({'model': bundle['name'], 'kind': bundle['kind'], 'split': split, **row})
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def plot_comparisons(metrics, predictions, reports_dir):
    """Plot the four formal models on validation data at locked thresholds."""
    validation_metrics = metrics.loc[metrics['split'].eq('validation') & metrics['kind'].eq('formal')]
    validation = predictions.loc[predictions['split'].eq('validation') &
                                 predictions['model'].isin(validation_metrics['model'])]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, (_, row) in zip(axes.ravel(), validation_metrics.iterrows()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        axis.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                axis.text(j, i, matrix[i, j], ha='center', va='center',
                          color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['non-CV', 'CV'],
                 yticklabels=['non-CV', 'CV'], xlabel='Predicted', ylabel='True', title=row.model)
    fig.suptitle('Validation confusion matrices')
    fig.tight_layout()
    fig.savefig(reports_dir / 'validation_confusion_matrices.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, subset in validation.groupby('model', sort=False):
        precision, recall, _ = precision_recall_curve(subset['y'], subset['p_cv'])
        fpr, tpr, _ = roc_curve(subset['y'], subset['p_cv'])
        axes[0].plot(recall, precision, label=name)
        axes[1].plot(fpr, tpr, label=name)
    axes[0].axhline(validation['y'].mean(), color='gray', linestyle=':', label='CV prevalence')
    axes[1].plot([0, 1], [0, 1], ':', color='gray')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='Validation precision-recall curves')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC curves')
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_dir / 'validation_pr_roc_curves.png', dpi=160)
    plt.close(fig)


def feature_importances(frame, bundles, reports_dir):
    validation = frame.loc[frame['split'].eq('validation')]
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    formal = [bundle for bundle in bundles if bundle['kind'] == 'formal']
    for axis, bundle in zip(axes.ravel(), formal):
        result = permutation_importance(bundle['pipeline'], bundle_features(validation, bundle),
                                        validation['y'], scoring='average_precision',
                                        n_repeats=10, random_state=SEED, n_jobs=1)
        table = pd.DataFrame({'model': bundle['name'], 'feature': bundle['features'],
                              'ap_decrease_mean': result.importances_mean,
                              'ap_decrease_std': result.importances_std}).sort_values('ap_decrease_mean')
        results.append(table)
        axis.barh(table['feature'], table['ap_decrease_mean'], xerr=table['ap_decrease_std'])
        axis.axvline(0, color='gray', linewidth=0.5)
        axis.set(title=bundle['name'], xlabel='Validation AP decrease after permutation')
    fig.tight_layout()
    fig.savefig(reports_dir / 'permutation_importance.png', dpi=160)
    plt.close(fig)
    table = pd.concat(results, ignore_index=True)
    table.to_csv(reports_dir / 'permutation_importance.csv', index=False)
    return table


def plot_selected_test(row, predictions, reports_dir):
    """Plot test performance for the model chosen before test evaluation."""
    matrix = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]], dtype=int)
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=['non-CV', 'CV']).plot(
        ax=axis, cmap='Blues', colorbar=False, values_format='d')
    axis.set_title(f"{row['model']}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(reports_dir / 'test_confusion_matrix.png', dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(predictions['y'], predictions['p_cv'])
    fpr, tpr, _ = roc_curve(predictions['y'], predictions['p_cv'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(recall, precision, label=f"{row['model']}: AP={row['ap']:.4f}")
    axes[0].axhline(predictions['y'].mean(), color='gray', linestyle=':', label='CV prevalence')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='Test precision-recall curve')
    axes[1].plot(fpr, tpr, label=f"{row['model']}: AUC={row['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], ':', color='gray', label='Random ranking')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC curve')
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_dir / 'test_pr_roc_curves.png', dpi=160)
    plt.close(fig)


def evaluate_selected_test(frame, bundle):
    """Return test scores and predictions without fitting or writing files."""
    test = frame.loc[frame['split'].eq('test')]
    predictions = predict_bundle(test, bundle)
    predictions['y'] = test['y'].to_numpy()
    predictions['split'] = 'test'
    row = metric_row(test['y'], predictions['p_cv'], bundle['threshold'])
    metrics = pd.DataFrame([{'model': bundle['name'], 'kind': bundle['kind'], 'split': 'test', **row}])
    return metrics, predictions


def inspect_errors(frame, predictions, name, processed_dir, reports_dir, lightcurves=None):
    """Save object-level errors locally and their light-curve gallery in reports."""
    prediction = predictions.loc[predictions['split'].eq('test') & predictions['model'].eq(name)]
    fp = prediction.loc[prediction['y'].eq(0) & prediction['prediction'].eq(1)].nlargest(10, 'p_cv')
    fn = prediction.loc[prediction['y'].eq(1) & prediction['prediction'].eq(0)].nsmallest(10, 'p_cv')
    errors = pd.concat([fp.assign(error='FP'), fn.assign(error='FN')], ignore_index=True)
    columns = ['#Name', 'Class', 'n_obs', 'match_status'] + DR3_FEATURES
    errors = errors.merge(frame[columns], on='#Name', how='left')
    errors.to_csv(processed_dir / 'representative_errors.csv', index=False)
    if lightcurves is not None and len(errors):
        fig, axes = plt.subplots(4, 5, figsize=(17, 11))
        for axis, (_, row) in zip(axes.ravel(), errors.iterrows()):
            curve = lightcurves.loc[lightcurves['#Name'].eq(row['#Name'])].sort_values('jd')
            axis.scatter(curve['jd'] - curve['jd'].min(), curve['mag'], s=8)
            axis.invert_yaxis()
            axis.set(title=f"{row['error']} {row['#Name']} (p={row['p_cv']:.2f})",
                     xlabel='Days since first observation', ylabel='G magnitude')
        for axis in axes.ravel()[len(errors):]:
            axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(reports_dir / 'error_lightcurves.png', dpi=130)
        plt.close(fig)
    elif lightcurves is not None:
        fig, axis = plt.subplots(figsize=(6, 3))
        axis.text(0.5, 0.5, 'No misclassified test objects.', ha='center', va='center')
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(reports_dir / 'error_lightcurves.png', dpi=130)
        plt.close(fig)
    return errors


def evaluate_models(frame, models_dir, processed_dir, reports_dir, lightcurves=None):
    """Generate evaluation details and report assets from locked models."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    names = ['rf_9', 'lgbm_9', 'rf_13', 'lgbm_13', 'dummy', 'coverage_diagnostic']
    bundles = [joblib.load(models_dir / f'{name}.joblib') for name in names]
    selected = json.loads((models_dir / 'selected_model.json').read_text(encoding='utf-8'))
    validation_metrics, validation_predictions = evaluate_predictions(frame, bundles, splits=('validation',))
    selected_bundle = next(bundle for bundle in bundles if bundle['name'] == selected['model'])
    test_metrics, test_predictions = evaluate_selected_test(frame, selected_bundle)
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)
    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    metrics.to_csv(reports_dir / 'metrics.csv', index=False)
    test_predictions.to_csv(processed_dir / 'test_predictions.csv', index=False)
    plot_comparisons(metrics, predictions, reports_dir)
    plot_selected_test(test_metrics.iloc[0], test_predictions, reports_dir)
    feature_importances(frame, bundles, reports_dir)
    inspect_errors(frame, predictions, selected['model'], processed_dir, reports_dir, lightcurves)
    return metrics
