import pandas as pd

LIGHTCURVE_COLUMNS = ['#Date', 'JD(TCB)', 'averagemag']
LIGHTCURVE_URL = 'https://gsaweb.ast.cam.ac.uk/alerts/alert/{name}/lightcurve.csv/'


def load_lightcurves(names, directory):
    """Combine previously validated files without checking their values."""
    curves = []
    for name in names:
        curve = pd.read_csv(directory / f'{name}.csv', skiprows=2, header=None,
                            names=['date', 'jd', 'mag'])
        curve['#Name'] = name
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)


def clean_lightcurves(lightcurves):
    """Clean numeric observations, deduplicate, and require two time points."""
    cleaned = lightcurves.copy()
    summary = {'initial_rows': len(lightcurves)}
    for col in ['jd', 'mag']:
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        cleaned[col] = cleaned[col].replace([float('inf'), -float('inf')], float('nan'))
    cleaned = cleaned.dropna(subset=['jd', 'mag']).copy()
    summary['invalid_numeric_rows_removed'] = len(lightcurves) - len(cleaned)

    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=['#Name', 'jd', 'mag'])
    summary['duplicate_rows_removed'] = rows_before - len(cleaned)

    rows_before = len(cleaned)
    cleaned = cleaned.groupby(['#Name', 'jd'], as_index=False).agg(
        date=('date', 'first'), mag=('mag', 'median'))
    cleaned = cleaned.sort_values(['#Name', 'jd']).reset_index(drop=True)
    summary['same_time_rows_merged'] = rows_before - len(cleaned)

    rows_before = len(cleaned)
    n_obs = cleaned.groupby('#Name')['jd'].transform('size')
    cleaned = cleaned.loc[n_obs >= 2].reset_index(drop=True)
    summary['insufficient_observation_rows_removed'] = rows_before - len(cleaned)
    summary['final_rows'] = len(cleaned)
    return cleaned, summary
