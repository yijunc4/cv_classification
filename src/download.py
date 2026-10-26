"""Shared retry policies and atomic CSV output."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import os
from pathlib import Path
import random
import re
import socket
import ssl
import time

import requests

from .validation import validate_csv


def http_status(error):
    """Read an HTTP status from a response or a client's error message."""
    response = getattr(error, 'response', None)
    if response is not None:
        return response.status_code
    # Some clients retain the status only in the exception message.
    if isinstance(error, requests.HTTPError):
        match = re.search(r'\b([45]\d{2})\b', str(error))
        if match:
            return int(match.group(1))
    return None


def can_retry(error):
    """Identify temporary failures in repeatable network operations."""
    status = http_status(error)
    if status is not None:
        return status in {408, 429, 500, 502, 503, 504}
    if isinstance(error, (requests.exceptions.SSLError, ssl.SSLError)):
        return False
    return isinstance(error, (requests.Timeout, requests.ConnectionError,
                              TimeoutError, ConnectionError, socket.gaierror, HTTPException))


def _retry_after(error):
    response = getattr(error, 'response', None)
    value = response.headers.get('Retry-After') if response is not None else None
    if not value:
        return 0
    try:
        return float(value)
    except Exception as e:
        try:
            return (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except Exception as e:
            return 0


def retry_call(operation, *, max_retries=2, retryable=can_retry):
    """Retry a temporary failure at most twice after the initial attempt.

    Use exponential backoff with jitter and honor an exposed Retry-After
    header. Stop when the server requests a wait longer than 60 seconds.
    The caller supplies a stricter policy for non-idempotent operations.
    """
    if max_retries < 0:
        raise ValueError('max_retries must be non-negative')
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries or not retryable(e):
                raise

            # Default waits are 2 and 4 seconds, plus 0-1 second of jitter.
            delay = min(2 * 2 ** attempt, 59) + random.uniform(0, 1)
            # Retry-After may contain seconds or an HTTP date.
            seconds = _retry_after(e)
            if seconds > 60:
                raise RuntimeError(f'Server requested a wait of {seconds:.0f}s; retry later.') from e
            delay = max(delay, seconds)

            print(f'Retry {attempt + 1}/{max_retries} in {delay:.1f}s: {e}', flush=True)
            time.sleep(delay)


def save_csv_atomic(frame, path, *, validate=None):
    """Write a sibling .part CSV, optionally check its structure, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    try:
        with part.open('w', encoding='utf-8', newline='') as file:
            frame.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        if validate is not None:
            validate(part)
        part.replace(path)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise

    return path


def download_csv(url, path, columns, *, filename, title=None, session=None,
                 max_retries=2, timeout=(10, 60)):
    """Download with bounded retries and validate structure before renaming.

    Existing files are checked and reused. Numeric values, missing values,
    labels, and scientific plausibility are left to the cleaning stages.
    """
    path = Path(path)
    if path.exists():
        validate_csv(path, columns, filename=filename, title=title)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    client = session if session is not None else requests

    def transfer():
        try:
            with client.get(url, timeout=timeout) as response:
                response.raise_for_status()
                with part.open('wb') as file:
                    file.write(response.content)
                    file.flush()
                    os.fsync(file.fileno())
            validate_csv(part, columns, filename=filename, title=title)
            part.replace(path)
        except Exception as e:
            part.unlink(missing_ok=True)
            raise
        return path

    return retry_call(transfer, max_retries=max_retries)
