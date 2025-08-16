#!/usr/bin/env python3
import json, glob
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def load_histories(symbol_dir):
    files = sorted(Path(symbol_dir).glob("history_*.json"))
    hs = []
    for f in files:
        try:
            with open(f,"r") as fh:
                hs.append((f.name, json.load(fh)))
        except Exception:
            pass
    return hs

def plot_symbol(symbol, hs, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for name,h in hs:
        ep = h.get("epoch", list(range(1, len(h.get("loss", []))+1)))
        if "loss" in h:
            plt.plot(ep, h["loss"], label=f"{name}-loss")
        if "val_loss" in h:
            plt.plot(ep, h["val_loss"], linestyle="--", label=f"{name}-val")
    plt.title(f"Learning Curves: {symbol}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir/f"{symbol}_learning_curves.png", dpi=140)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/*/history_*.json")
    args = ap.parse_args()
    files = glob.glob(args.glob)
    symbols = sorted(set(Path(f).parent.name for f in files))
    for sym in symbols:
        hs = load_histories(Path("CODE/reports")/sym)
        if hs:
            plot_symbol(sym, hs, Path("CODE/reports")/sym)
            print(f"[OK] Wrote CODE/reports/{sym}/{sym}_learning_curves.png")
        else:
            print(f"[WARN] No histories for {sym}")

if __name__ == "__main__":
    main()
