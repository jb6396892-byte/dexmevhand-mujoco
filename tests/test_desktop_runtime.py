import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.runtime import JsonLines,MAX_PACKET,physics_environment,language_environment


class DesktopRuntimeTests(unittest.TestCase):
    def test_split_utf8_packet(self):
        parser=JsonLines(); raw=(json.dumps(dict(type='status',message='抓起杯子'),ensure_ascii=False)+'\n').encode()
        result=[]
        for byte in raw: result+=parser.feed(bytes([byte]))
        self.assertEqual(result,[dict(type='status',message='抓起杯子')])

    def test_multiple_packets(self):
        self.assertEqual(len(JsonLines().feed(b'{"type":"ready"}\n{"type":"frame"}\n')),2)

    def test_bad_packets_fail(self):
        for raw in (b'garbage\n',b'[]\n',b'{"hello":1}\n',b'x'*(MAX_PACKET+1)):
            with self.assertRaises(ValueError): JsonLines().feed(raw)

    def test_gpu_environment_isolated(self):
        with patch('pathlib.Path.is_file',return_value=True):
            env=physics_environment(dict(DISPLAY=':0',MUJOCO_PY_FORCE_CPU='1',PYTHONPATH='unsafe',QT_PLUGIN_PATH='bad'))
        self.assertNotIn('MUJOCO_PY_FORCE_CPU',env); self.assertNotIn('QT_PLUGIN_PATH',env)
        self.assertIn('mujoco-py-gpu',env['PYTHONPATH']); self.assertIn('libGLEW.so',env['LD_PRELOAD'])
        self.assertEqual(env['__GLX_VENDOR_LIBRARY_NAME'],'nvidia')

    def test_missing_display_fail_closed(self):
        with patch('pathlib.Path.is_file',return_value=True):
            with self.assertRaises(RuntimeError): physics_environment({})

    def test_language_does_not_inherit_qt(self):
        env=language_environment('/shared',dict(PYTHONPATH='qt',LD_PRELOAD='bad',QT_PLUGIN_PATH='bad'))
        self.assertNotIn('LD_PRELOAD',env); self.assertNotIn('QT_PLUGIN_PATH',env)
        self.assertTrue(env['PYTHONPATH'].startswith('/shared/language/runtime/packages:'))


if __name__=='__main__': unittest.main()
