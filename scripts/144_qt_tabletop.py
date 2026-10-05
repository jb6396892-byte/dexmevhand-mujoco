#!/usr/bin/env python3
"""New visual-control console, locked by default until separately authorized tests."""
import argparse
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from fromrealhand.desktop.tabletop_window import TabletopWindow

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
p.add_argument('--visual-root',type=Path,default=Path('/media/smgbro/shared/visual_grasp'))
p.add_argument('--allow-unvalidated-tabletop',action='store_true')
p.add_argument('--offline-layout-screenshot',type=Path)
p.add_argument('--checkpoint',type=Path)
a=p.parse_args()
if a.offline_layout_screenshot and a.allow_unvalidated_tabletop:
    p.error('Offline layout capture must not enable execution')
app=QApplication(sys.argv[:1]); app.setStyle('Fusion')
window=TabletopWindow(a.root,a.visual_root,a.allow_unvalidated_tabletop,a.checkpoint); window.show()
if a.offline_layout_screenshot:
    window.load_offline_evidence()
    def capture():
        a.offline_layout_screenshot.parent.mkdir(parents=True,exist_ok=True)
        if not window.grab().save(str(a.offline_layout_screenshot)): app.exit(1)
        else: app.exit(0)
    QTimer.singleShot(400,capture)
raise SystemExit(app.exec())
