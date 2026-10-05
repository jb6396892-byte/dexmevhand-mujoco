"""Stop-aware IPC. Simulation remains paused while the isolated vision worker runs."""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np
from ..desktop.runtime import ROOT
from .camera import read_rgbd


class VisionClient:
    def __init__(self, root, run, mesh, stop, timeout):
        self.run, self.stop, self.timeout = Path(run), stop, timeout
        self.counter = 0; self.responses = queue.Queue()
        np.savez(str(self.run/'mug_model.npz'), **mesh)
        self.log = (self.run/'vision-stderr.log').open('w')
        self.process = subprocess.Popen(['bash', str(ROOT/'scripts/tabletop_python.sh'),
            str(ROOT/'scripts/142_tabletop_vision_worker.py'), '--run', str(self.run),
            '--model', str(Path(root)/'models/grounding-dino-tiny')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            universal_newlines=True, bufsize=1, cwd=str(ROOT))
        threading.Thread(target=self._read, daemon=True).start()
        try:
            if self.wait()['type'] != 'ready': raise ValueError('Vision worker did not initialize')
        except BaseException:
            self.close(); raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 2*1024*1024: raise ValueError('Oversized vision response')
                self.responses.put(json.loads(line))
        except Exception as error: self.responses.put(dict(type='error', reason=str(error)))
        finally: self.responses.put(dict(type='error', reason='Vision worker disconnected'))

    def wait(self):
        start = time.monotonic()
        while time.monotonic()-start < self.timeout:
            if self.stop.is_set(): raise RuntimeError('User cancelled vision')
            try:
                response = self.responses.get(timeout=.05)
                if response['type'] == 'error': raise RuntimeError(response['reason'])
                return response
            except queue.Empty: pass
        raise TimeoutError('Vision worker timeout')

    def capture(self, sim, context, camera_name='rgbd'):
        import cv2
        frame = '%06d'%self.counter; self.counter += 1
        directory = self.run/'observations'/frame; directory.mkdir(parents=True, exist_ok=False)
        rgb, depth, camera = read_rgbd(sim, context, camera_name)
        cv2.imwrite(str(directory/'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(str(directory/'depth.npy'), depth)
        (directory/'camera.json').write_text(json.dumps(camera, indent=2)+'\n')
        preview = cv2.applyColorMap(np.uint8(np.clip((depth-.2)/1.0, 0, 1)*255), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(directory/'depth-preview.png'), preview)
        started = time.monotonic()
        self.process.stdin.write(json.dumps(dict(command='estimate', frame=frame))+'\n'); self.process.stdin.flush()
        row = self.wait()
        if (row.get('type') != 'estimate' or row.get('frame') != frame
                or row.get('camera_time_s') != camera['time_s']
                or row.get('source_rgb_sha256') != hashlib.sha256((directory/'rgb.png').read_bytes()).hexdigest()):
            raise ValueError('Vision response does not match current RGB-D capture')
        row.update(wall_latency_s=time.monotonic()-started, observation_directory=str(directory), calibration=camera,
                   simulation_paused_during_inference=True)
        return row

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        for stream in (self.process.stdin, self.process.stdout): stream.close()
        self.log.close()
