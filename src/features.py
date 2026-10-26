import pandas as pd

from .config import DR3_FEATURES, LC_FEATURES


def extract_lightcurve_features(lightcurves):
    lightcurves = lightcurves.sort_values(['#Name', 'jd'])
    rows = []
    for name, curve in lightcurves.groupby('#Name'):
        mag = curve['mag']
        n = len(mag)
        median = mag.median()
        q = mag.quantile([0.05, 0.25, 0.75, 0.95], interpolation='linear')
        rolling_median = mag.rolling(7, center=True, min_periods=7).median()
        peaks = int((mag <= rolling_median - 2).sum()) if n >= 7 else float('nan')
        skew = mag.skew() if n >= 3 and mag.nunique() > 1 else float('nan')
        rows.append({
            '#Name': name,
            'n_obs': n,
            'time_span_days': curve['jd'].max() - curve['jd'].min(),
            'median_mag': median,
            'std_mag': mag.std(ddof=1),
            'iqr_mag': q.loc[0.75] - q.loc[0.25],
            'amplitude_p95_p05': q.loc[0.95] - q.loc[0.05],
            'bright_excursion_mag': median - mag.min(),
            'n_peaks_rm_2_7': peaks,
            'skew': skew,
        })
    return pd.DataFrame(rows)


def clean_feature_table(feature_table):
    """Clean 13 inputs without removing sources or filling missing values."""
    cleaned = feature_table.copy()
    summary = {'initial_rows': len(feature_table), 'non_numeric_values_replaced': 0,
               'non_finite_values_replaced': 0}
    for col in LC_FEATURES + DR3_FEATURES:
        numeric = pd.to_numeric(cleaned[col], errors='coerce')
        summary['non_numeric_values_replaced'] += int((cleaned[col].notna() & numeric.isna()).sum())
        summary['non_finite_values_replaced'] += int(numeric.isin([float('inf'), -float('inf')]).sum())
        cleaned[col] = numeric.replace([float('inf'), -float('inf')], float('nan'))

    invalid_pm = cleaned['pm'] < 0
    summary['negative_pm_values_replaced'] = int(invalid_pm.sum())
    cleaned.loc[invalid_pm, 'pm'] = float('nan')
    nonunique = cleaned['match_status'] != 'matched'
    summary['nonunique_match_values_cleared'] = int(cleaned.loc[nonunique, DR3_FEATURES].notna().sum().sum())
    cleaned.loc[nonunique, DR3_FEATURES] = float('nan')
    summary['rows_removed'] = len(feature_table) - len(cleaned)
    summary['final_rows'] = len(cleaned)
    return cleaned, summary
