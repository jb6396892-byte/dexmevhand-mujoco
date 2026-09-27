#!/usr/bin/env python3
"""Read only recent five-hour usage metadata; never inspect credentials."""
import datetime
import json
from pathlib import Path


def latest_usage(root=None):
    root = Path(root) if root else Path.home()/'.codex/sessions'
    candidates = []
    files = sorted(root.rglob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)[:4]
    for path in files:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-2000000))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
                limits = event.get('payload', {}).get('rate_limits') or {}
                primary = limits.get('primary') or {}
                if primary.get('window_minutes') != 300:
                    continue
                stamp = event['timestamp']
                age = (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(stamp.replace('Z', '+00:00'))).total_seconds()
                candidates.append(dict(timestamp=stamp, used_percent=float(primary['used_percent']),
                                       resets_at=primary.get('resets_at'), age_s=age, source='local_session_rate_limits'))
                break
            except (ValueError, TypeError, KeyError):
                continue
    return max(candidates, key=lambda x: x['timestamp']) if candidates else None


if __name__ == '__main__':
    print(json.dumps(latest_usage(), indent=2))
