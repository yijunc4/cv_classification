"""Project paths and the fixed feature schema."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
INTERIM = ROOT / 'data' / 'interim'
PROCESSED = ROOT / 'data' / 'processed'  # Local split, search, and object-level records.
MODELS = ROOT / 'models'  # Fitted bundles and the selected model specification.
REPORTS = ROOT / 'reports'  # Public report, summary tables, and figures.
SEED = 42

LC_FEATURES = [
    'n_obs', 'time_span_days', 'median_mag', 'std_mag', 'iqr_mag',
    'amplitude_p95_p05', 'bright_excursion_mag', 'n_peaks_rm_2_7', 'skew',
]
DR3_FEATURES = ['bp_rp', 'parallax', 'parallax_over_error', 'pm']


def cleaned_path(path):
    """Prefix the complete filename once, without shortening its stem."""
    path = Path(path)
    return path if path.name.startswith('cleaned_') else path.with_name('cleaned_' + path.name)
