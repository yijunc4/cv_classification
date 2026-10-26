ALERTS_URL = 'https://gsaweb.ast.cam.ac.uk/alerts/alerts.csv'
ALERT_COLUMNS = [
    '#Name', 'Date', 'RaDeg', 'DecDeg', 'AlertMag', 'HistoricMag',
    'HistoricStdDev', 'Class', 'Published', 'Comment', 'TNSid',
]
SN_CLASSES = [
    'SN', 'SN I', 'SN I-pec', 'SN Ia', 'SN Ia-CSM', 'SN Ia-pec',
    'SN Iax', 'SN Ib', 'SN Ib/c', 'SN Ibn', 'SN Ic', 'SN Ic-BL',
    'SN Ic-pec', 'SN II', 'SN IIb', 'SN IIn', 'SN IIn-pec',
    'SN IIP', 'SN IIL', 'SLSN', 'CCSN',
]


def select_alerts(alerts):
    """Trim headers and class labels, apply the whitelist, and assign y."""
    alerts = alerts.copy()
    alerts.columns = alerts.columns.str.strip()
    alerts['Class'] = alerts['Class'].astype('string').str.strip()
    class_to_label = {'CV': 1}
    class_to_label.update(dict.fromkeys(SN_CLASSES + ['AGN', 'QSO', 'BL Lac', 'YSO'], 0))
    selected = alerts.loc[alerts['Class'].isin(class_to_label)].copy()
    selected['y'] = selected['Class'].map(class_to_label).astype(int)
    return selected
