"""CSV structure checks for downloaded files; no data-value validation."""

import csv
from pathlib import Path


def validate_csv(path, columns, *, filename, title=None):
    """Check filename, optional title, column names, and nonblank row widths."""
    path = Path(path)
    if path.name not in {filename, filename + '.part'}:
        raise ValueError(f'Unexpected filename: {path.name}; expected {filename}')

    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = csv.reader(file, strict=True)
        if title is not None and next(rows, []) != [title]:
            raise ValueError(f'Title does not match {title}')
        header = [name.strip() for name in next(rows, [])]
        if header != list(columns):
            raise ValueError(f'Unexpected columns: {header}')
        first_line = 3 if title is not None else 2
        for line, row in enumerate(rows, first_line):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(columns):
                raise ValueError(f'Line {line}: expected {len(columns)} columns, got {len(row)}')
