#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PATTERNS_JSON = ["history*.json", "*history*.json", "*train*history*.json"]
PATTERNS_CSV  = ["history*.csv", "*history*.csv", "train_log*.csv", "*train*history*.csv"]

def load_json_curve(p):
    with open(p, "r") as f:
        h = json.load(f)
    # accept common keys
    loss     = h.get("loss") or h.get("train_loss") or h.get("training_loss")
    val_loss = h.get("val_loss") or h.get("valid_loss") or h.get("validation_loss")
    if isinstance(loss, dict): loss = list(loss.values())
    if isinstance(val_loss, dict): val_loss = list(val_loss.values())
    return loss, val_loss

def load_csv_curve(p):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    loss = None; val_loss = None
    for k in ["loss","train_loss","training_loss"]:
        if k in cols: loss = df[cols[k]].tolist(); break
    for k in ["val_loss","valid_loss","validation_loss"]:
        if k in cols: val_loss = df[cols[k]].tolist(); break
    return loss, val_loss

def save_plot(out_path, loss, val_loss, title):
    plt.figure()
    if loss is not None:
        plt.plot(range(1, len(loss)+1), loss, label="train")
    if val_loss is not None:
        plt.plot(range(1, len(val_loss)+1), val_loss, linestyle="--", label="val")
    plt.title(title)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    if (loss is not None and len(loss)>0) or (val_loss is not None and len(val_loss)>0):
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="CODE/reports")
    ap.add_argument("--outdir", default="CODE/reports/learning_curves")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    index_lines = []
    found = 0

    # JSON
    for pat in PATTERNS_JSON:
        for p in root.rglob(pat):
            try:
                loss, val_loss = load_json_curve(p)
                if loss is None and val_loss is None: 
                    continue
                sym = p.parent.name
                local_out = p.with_name(p.stem + "_learning.png")
                save_plot(local_out, loss, val_loss, f"{sym} — {p.name}")
                central_out = outdir / f"{sym}__{p.stem}_learning.png"
                save_plot(central_out, loss, val_loss, f"{sym} — {p.name}")
                index_lines.append(str(central_out))
                found += 1
            except Exception:
                continue

    # CSV
    for pat in PATTERNS_CSV:
        for p in root.rglob(pat):
            try:
                loss, val_loss = load_csv_curve(p)
                if loss is None and val_loss is None: 
                    continue
                sym = p.parent.name
                local_out = p.with_name(p.stem + "_learning.png")
                save_plot(local_out, loss, val_loss, f"{sym} — {p.name}")
                central_out = outdir / f"{sym}__{p.stem}_learning.png"
                save_plot(central_out, loss, val_loss, f"{sym} — {p.name}")
                index_lines.append(str(central_out))
                found += 1
            except Exception:
                continue

    (outdir / "INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"[OK] curves={found} index={outdir/'INDEX.txt'}")
if __name__ == "__main__":
    main()
