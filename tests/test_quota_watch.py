import datetime
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
latest_usage = importlib.import_module('check_codex_budget').latest_usage


class QuotaWatchTest(unittest.TestCase):
    def test_ignores_weekly_limits_and_partial_lines(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'session.jsonl'
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            future = datetime.datetime.now(datetime.timezone.utc).timestamp()+300
            lines = [dict(timestamp=stamp,payload=dict(rate_limits=dict(primary=dict(window_minutes=300,used_percent=84,resets_at=future)))),
                     dict(timestamp=stamp,payload=dict(rate_limits=dict(primary=dict(window_minutes=10080,used_percent=99))))]
            p.write_text('\n'.join(json.dumps(x) for x in lines)+'\n{"incomplete":')
            self.assertEqual(latest_usage(root)['used_percent'],84)

    def test_missing_usage_is_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(latest_usage(root))

    def test_latest_event_wins_across_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            for i in range(2):
                stamp = (datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=2-i)).isoformat()
                (Path(root)/('%d.jsonl'%i)).write_text(json.dumps(dict(timestamp=stamp,
                    payload=dict(rate_limits=dict(primary=dict(window_minutes=300,used_percent=80+i))))))
            self.assertEqual(latest_usage(root)['used_percent'],81)

    def test_old_or_reset_window_is_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            now = datetime.datetime.now(datetime.timezone.utc)
            for stamp, reset in [(now-datetime.timedelta(hours=2), now.timestamp()+300), (now, now.timestamp()-1)]:
                (Path(root)/'old.jsonl').write_text(json.dumps(dict(timestamp=stamp.isoformat(),
                    payload=dict(rate_limits=dict(primary=dict(window_minutes=300,used_percent=79,resets_at=reset))))))
                self.assertIsNone(latest_usage(root))
