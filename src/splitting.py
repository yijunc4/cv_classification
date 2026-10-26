"""Keep related alerts together in fixed stratified data splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree

from src.config import SEED


def object_groups(frame):
    """Connect shared accepted DR3 IDs and alert positions within 2 arcsec."""
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    seen = {}
    for i, (status, source_id) in enumerate(zip(frame['match_status'], frame['source_id'])):
        if status == 'matched' and pd.notna(source_id):
            if source_id in seen:
                union(i, seen[source_id])
            seen[source_id] = i

    positions = np.deg2rad(frame[['dec', 'ra']].to_numpy())
    neighbors = BallTree(positions, metric='haversine').query_radius(
        positions, r=np.deg2rad(2 / 3600))
    for i, nearby in enumerate(neighbors):
        for j in nearby:
            if j > i:
                union(i, int(j))

    roots = pd.Series([find(i) for i in range(len(frame))], index=frame.index)
    return frame['#Name'].groupby(roots).transform('min').rename('group_id')


def make_manifest(frame):
    """Create grouped splits and check class coverage before saving them."""
    manifest = frame[['#Name', 'y']].copy()
    manifest['group_id'] = object_groups(frame)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    manifest['split'] = 'train'
    manifest['cv_fold'] = -1
    for fold, (_, held_out) in enumerate(outer.split(frame, frame['y'], manifest['group_id'])):
        if fold < 2:
            manifest.iloc[held_out, manifest.columns.get_loc('split')] = ['test', 'validation'][fold]
    for split, rows in manifest.groupby('split'):
        if rows['y'].nunique() != 2:
            raise ValueError(f'{split} must contain both classes.')
    train = manifest['split'].eq('train')
    subset = manifest.loc[train]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (_, held_out) in enumerate(inner.split(subset, subset['y'], subset['group_id'])):
        manifest.loc[subset.index[held_out], 'cv_fold'] = fold
    train_rows = manifest.loc[train]
    for fold in range(3):  
        mask = train_rows['cv_fold'].eq(fold)
        if train_rows.loc[mask, 'y'].nunique() != 2:
            raise ValueError(f'CV fold {fold} must contain both classes.')

    return manifest

def attach_manifest(frame, manifest):
    """Join saved split assignments onto the prepared feature table."""
    return frame.merge(manifest[['#Name', 'group_id', 'split', 'cv_fold']], on='#Name', how='left')
