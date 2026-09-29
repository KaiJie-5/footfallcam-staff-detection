from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence import collect_evidence

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Collect marker evidence using cached person detections')
    p.add_argument('--video',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--sample-seconds',type=float,default=0.2)
    p.add_argument('--reference-dir')
    collect_evidence(**vars(p.parse_args()))
