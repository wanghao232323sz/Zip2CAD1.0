"""Compatibility command; the maintained implementation is parameterized."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zip2cad.release_cli import main
if __name__=='__main__':raise SystemExit(main(['infer-joint',*sys.argv[1:]]))
