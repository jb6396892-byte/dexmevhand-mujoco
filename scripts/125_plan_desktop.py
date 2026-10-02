#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fromrealhand.desktop.planning import plan_instruction
from fromrealhand.desktop.runtime import emit

parser = argparse.ArgumentParser(description='Frozen language model inference for the Qt desktop')
parser.add_argument('instruction')
parser.add_argument('--scene',choices=['first','second'],required=True)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
try:
    raise SystemExit(0 if plan_instruction(args.root,args.instruction,args.scene,args.output) else 2)
except Exception as error:
    emit('error',message=str(error),error_type=type(error).__name__)
    raise SystemExit(1)
