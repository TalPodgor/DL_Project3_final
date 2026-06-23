#!/usr/bin/env python3
"""Check whether FEN(s) were seen by the model (train games 4,5,6,7 or test game 2).

A position counts as SEEN if its placement field (the part before the first space)
appears anywhere in gt.csv. gt.csv = all 876 rows = 736 train + 140 test, so one
check covers both. Usage:
    python check_fen_unseen.py "<fen1>" "<fen2>" ...
    python check_fen_unseen.py              # checks the built-in candidate list
"""
import csv, sys, os

GT = os.path.join(os.path.dirname(__file__), "data", "gt.csv")
if not os.path.exists(GT):
    GT = "gt.csv"  # fallback: dataset_root/gt.csv

def placement(fen): return fen.strip().split()[0]

def load_seen(path):
    seen = set()
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            seen.add(placement(row[1]))
    return seen

def main(argv):
    seen = load_seen(GT)
    print(f"loaded {len(seen)} unique seen positions from {GT}\n")
    cands = argv or [
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",   # Italian Game
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1",# Two Knights-ish
        "1n1Rkb1r/p4ppp/4q3/4p1B1/4P3/8/PPP2PPP/2K5",             # Opera Game finish
        "8/8/8/4k3/8/8/3QK3/8",                                    # K+Q vs K endgame
        "8/8/8/4k3/4P3/4K3/8/8",                                   # K+P vs K endgame
        "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1", # symmetric middlegame
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",             # start (CONTROL: should be SEEN)
    ]
    for fen in cands:
        tag = "SEEN  ❌" if placement(fen) in seen else "UNSEEN ✅"
        print(f"{tag}  {fen}")

if __name__ == "__main__":
    main(sys.argv[1:])
