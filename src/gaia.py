"""Build Gaia DR3 searches and retrieve candidates from an existing TAP job."""

import socket

import requests

from .download import http_status, retry_call

DR3_CANDIDATE_COLUMNS = [
    'alert_name', 'source_id', 'dr3_ra', 'dr3_dec', 'bp_rp', 'parallax',
    'parallax_over_error', 'pm', 'match_distance_arcsec',
]


def prepare_targets(lc_features, selected_alerts):
    """Get alert coordinates only for sources with light-curve features."""
    targets = lc_features[['#Name']].merge(
        selected_alerts[['#Name', 'RaDeg', 'DecDeg']], on='#Name', how='left')
    return targets.rename(columns={'#Name': 'alert_name', 'RaDeg': 'ra', 'DecDeg': 'dec'})


def match_candidates(targets, candidates):
    """Accept unique matches and retain all unmatched or ambiguous sources."""
    matches = targets.copy()
    counts = candidates['alert_name'].value_counts()
    matches['n_candidates'] = matches['alert_name'].map(counts).fillna(0).astype(int)
    matches['match_status'] = 'unmatched'
    matches.loc[matches['n_candidates'] == 1, 'match_status'] = 'matched'
    matches.loc[matches['n_candidates'] > 1, 'match_status'] = 'ambiguous'
    unique_names = matches.loc[matches['n_candidates'] == 1, 'alert_name']
    unique_candidates = candidates.loc[candidates['alert_name'].isin(unique_names)]
    matches = matches.merge(unique_candidates, on='alert_name', how='left')
    return matches.rename(columns={'alert_name': '#Name'})


def build_search_query(radius_arcsec=2.0):
    """Return all DR3 candidates near the uploaded alert coordinates."""
    # Gaia translates this to Q3C; put the indexed catalog coordinates second.
    return f"""
SELECT
    a.alert_name,
    g.source_id,
    g.ra AS dr3_ra,
    g.dec AS dr3_dec,
    g.bp_rp,
    g.parallax,
    g.parallax_over_error,
    g.pm,
    DISTANCE(POINT('ICRS', a.ra, a.dec), POINT('ICRS', g.ra, g.dec)) * 3600 AS match_distance_arcsec
FROM tap_upload.alerts AS a
JOIN gaiadr3.gaia_source AS g
ON 1 = CONTAINS(
    POINT('ICRS', a.ra, a.dec),
    CIRCLE('ICRS', g.ra, g.dec, {radius_arcsec / 3600})
)
"""


def _can_retry_submission(error):
    """Avoid duplicate jobs when a submission response might have been lost."""
    status = http_status(error)
    if status is not None:
        return status in {429, 503}
    return isinstance(error, (requests.ConnectTimeout, socket.gaierror))


def submit_search(query, upload_table, *, max_retries=2):
    """Import the Gaia client on demand and submit one asynchronous search."""
    from astroquery.gaia import Gaia

    return retry_call(
        lambda: Gaia.launch_job_async(
            query, upload_resource=upload_table,
            upload_table_name='alerts', background=True,
        ),
        max_retries=max_retries,
        retryable=_can_retry_submission,
    )


def fetch_candidates(job, *, max_retries=2):
    """Fetch one job's results, retaining source IDs as exact strings."""
    result = retry_call(job.get_results, max_retries=max_retries)
    candidates = result.to_pandas()
    # The inner join guarantees IDs exist before any missing values are added.
    candidates['source_id'] = candidates['source_id'].astype('string')
    return candidates
