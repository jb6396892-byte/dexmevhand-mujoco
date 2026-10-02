#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PySide6.QtWidgets import QApplication
from fromrealhand.desktop.window import GraspWindow
parser=argparse.ArgumentParser(description='Qt language-guided MuJoCo grasp console')
parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
args=parser.parse_args()
app=QApplication(sys.argv[:1]); app.setStyle('Fusion')
window=GraspWindow(args.root); window.show()
raise SystemExit(app.exec())
