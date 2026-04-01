#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

def best_threshold(y_true, y_prob):
    """Return dict with best threshold by max F1; ties broken by higher recall."""
    unique_classes = np.unique(y_true)
    
    # If no positive class (1) is found, raise a clear error or handle gracefully
    if len(unique_classes) < 2 or 1 not in unique_classes:
        print("Warning: No positive class (1) found in y_true, the thresholds cannot be computed.")
        return {"threshold": 0.5, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "avg_precision": float("nan")}
    
    # PR curve thresholds length = len(thresholds) = len(prec)-1
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Compute F1 at points where threshold exists
    f1_vals = 2 * prec[1:] * rec[1:] / np.clip(prec[1:] + rec[1:], 1e-12, None)
    if len(f1_vals) == 0:
        return {"threshold": 0.5, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "avg_precision": float("nan")}
    
    # argmax, tie-break by recall
    idx = np.lexsort(( -rec[1:], -f1_vals ))[0]  # maximize f1 then recall
    thr_star = float(thr[idx])
    
    # Compute metrics at the chosen threshold
    y_hat = (y_prob >= thr_star).astype(int)
    p = float((y_hat & (y_true==1)).sum() / max((y_hat==1).sum(), 1))
    r = float(((y_hat==1) & (y_true==1)).sum() / max((y_true==1).sum(), 1))
    f1 = float(f1_score(y_true, y_hat, zero_division=0))
    ap = float(average_precision_score(y_true, y_prob))
    return {"threshold": thr_star, "precision": p, "recall": r, "f1": f1, "avg_precision": ap}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="validation_predictions.csv", help="Path to validation_predictions.csv")
    ap.add_argument("--outdir", default=None, help="Where to write outputs (defaults to preds file's folder)")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(f"File not found: {preds_path}")

    outdir = Path(args.outdir) if args.outdir else preds_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)

    # Ensure 'rating' and 'drug_name' columns exist, or create them
    if "rating" not in df.columns:
        if "y_true" in df.columns:
            df["rating"] = df["y_true"]  # Assuming y_true is equivalent to rating
        else:
            raise ValueError("Column 'rating' or 'y_true' is required.")

    if "drug_name" not in df.columns:
        df["drug_name"] = "default_drug_name"  # Placeholder for missing drug_name, adjust if needed

    # Check for available 'prob_*' columns
    prob_cols = [col for col in ["prob_text", "prob_num", "prob_hybrid"] if col in df.columns]
    if not prob_cols:
        raise ValueError(f"No probability columns found. Expected any of: prob_text, prob_num, prob_hybrid.")

    y_true = (df["rating"] >= 4).astype(int).values  # Assume 4+ rating as positive

    # Build calibrated predictions DataFrame (start as a copy)
    calibrated = df.copy()

    rows = []
    for col in prob_cols:
        res = best_threshold(y_true, df[col].values)
        rows.append({"model": col.replace("prob_", ""), **res})
        calibrated[f"y_pred_{col.replace('prob_','')}"] = (df[col].values >= res["threshold"]).astype(int)

    # Save summary + calibrated predictions
    summary = pd.DataFrame(rows).sort_values("f1", ascending=False)
    summary_path = outdir / "thresholds_summary.csv"
    summary.to_csv(summary_path, index=False)

    calibrated_path = outdir / "calibrated_predictions.csv"
    calibrated.to_csv(calibrated_path, index=False)

    # Pretty print
    print("\nBest thresholds per model (max F1):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved:\n - {summary_path}\n - {calibrated_path}")

if __name__ == "__main__":
    main()
