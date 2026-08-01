import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / 'dataset'
OUTPUT_PATH = DATASET_DIR / 'output.csv'
MESSAGE_HISTORY_PATH = DATASET_DIR / 'message_history.csv'


def main() -> None:
    rows = list(csv.DictReader(OUTPUT_PATH.open('r', encoding='utf-8', newline='')))
    messages = list(csv.DictReader((DATASET_DIR / 'messages.csv').open('r', encoding='utf-8', newline='')))
    history_ids = {row.get('message_id') for row in csv.DictReader(MESSAGE_HISTORY_PATH.open('r', encoding='utf-8', newline=''))}

    if len(rows) != len(messages):
        raise ValueError(f'Expected {len(messages)} rows, got {len(rows)}')
    if [row.get('message_id') for row in rows] != [message.get('message_id') for message in messages]:
        raise ValueError('Output message_ids do not match messages.csv ordering')

    allowed_actions = {'notify', 'digest', 'mute'}
    allowed_types = {'personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown'}
    for row in rows:
        if row.get('action') not in allowed_actions:
            raise ValueError(f'Invalid action: {row.get("action")}')
        if row.get('message_type') not in allowed_types:
            raise ValueError(f'Invalid message_type: {row.get("message_type")}')
        try:
            conf = float(row.get('confidence', 'nan'))
        except Exception as exc:
            raise ValueError(f'Invalid confidence: {row.get("confidence")}') from exc
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f'Confidence out of range: {conf}')
        evidence_ids = [item.strip() for item in row.get('evidence_message_ids', '').split(';') if item.strip() and item.strip() != 'none']
        invalid = [item for item in evidence_ids if item not in history_ids]
        if invalid:
            raise ValueError(f'Invalid evidence IDs: {invalid}')

    print('evaluation_ok')
    print('rows', len(rows))
    print('header', list(rows[0].keys()) if rows else [])


if __name__ == '__main__':
    main()
