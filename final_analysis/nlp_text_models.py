#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlp_text_models.py

Train and evaluate text-only models on the drug review text:
  • auto-detect rating, drug, and text columns
  • build 'text_all' (concatenate chosen text fields)
  • label = rating >= smart threshold (4/5 or 8/10 by scale)
  • group-aware split by drug (prevents leakage)
  • models:
      1) TF-IDF(word 1–2g) + LogisticRegression (calibrated probs natively)
      2) TF-IDF(word 1–2g) + TF-IDF(char_wb 3–5g) -> LinearSVC + CalibratedClassifierCV
     (optional) ComplementNB (kept simple; off by default)
  • exports:
      - text_models_metrics.json
      - text_models_reports/*.txt (classification reports)
      - validation_predictions_text.csv  (y_true + prob_* + y_* at 0.5)
      - model_text_lr.joblib, model_text_svc.joblib, model_text_best.joblib
      - top_text_features.csv (from LR for interpretability)

Usage:
    python nlp_text_models.py \
      --reviews final_cleaned_train_data.csv \
      --outdir outputs \
      --text-cols benefitsReview sideEffectsReview commentsReview
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from joblib import dump

# -------------------- small utils --------------------

def smart_read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

def normalize_drug(s):
    if pd.isna(s):
        return s
    s = str(s).lower()
    s = re.sub(r'\b(hcl|hydrochloride|sr|xr|er|ir|cr|dr|mr|od|mg|mcg|ug|tablet|tablets|tab|capsule|capsules|cap|solution|suspension)\b', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+',' ', s).strip()

def find_review_drug_col(df: pd.DataFrame) -> str:
    for c in ['urlDrugName','drug_name','drug','Drug','medicine','medication']:
        if c in df.columns: return c
    raise KeyError(f"No drug column found in reviews. Columns: {list(df.columns)}")

def detect_rating_col(df: pd.DataFrame) -> str:
    for c in ['rating','ratings','score','stars']:
        if c in df.columns: return c
    raise KeyError(f"No rating column found in reviews. Columns: {list(df.columns)}")

def pick_text_cols(df: pd.DataFrame, user_cols=None):
    if user_cols:
        got = [c for c in user_cols if c in df.columns]
        if got: return got
    cands = ['benefitsReview','sideEffectsReview','commentsReview',
             'review','text','review_text','content','comments']
    got = [c for c in cands if c in df.columns]
    if not got:
        raise KeyError("No text columns found. Pass --text-cols or include review text columns.")
    return got

def choose_label_threshold(ratings: pd.Series) -> float:
    rmax = ratings.max(skipna=True)
    default = 4.0 if rmax <= 5 else 8.0
    y = (ratings >= default).astype(int)
    if y.nunique() == 2: return default
    thr = float(ratings.quantile(0.6))
    y = (ratings >= thr).astype(int)
    if y.nunique() == 2: return thr
    return float(ratings.median())

def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2: return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

# -------------------- training --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True, help="Cleaned reviews CSV/TSV")
    ap.add_argument("--outdir", default="outputs", help="Where to write outputs")
    ap.add_argument("--rating-col", default=None, help="Override rating col name")
    ap.add_argument("--text-cols", nargs="*", default=None,
                    help="Space-separated list of text columns to combine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "text_models_reports").mkdir(parents=True, exist_ok=True)

    # --- load & basic prep ---
    df = smart_read(args.reviews)
    # drop unnamed
    bad = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if bad: df = df.drop(columns=bad)

    drug_col   = find_review_drug_col(df)
    rating_col = args.rating_col or detect_rating_col(df)
    text_cols  = pick_text_cols(df, args.text_cols)

    # normalize & build text_all
    df = df.rename(columns={drug_col: "drug_name"})
    df["drug_norm"] = df["drug_name"].astype(str).map(normalize_drug)
    for c in text_cols:
        if c not in df.columns: df[c] = ""
    df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    # labels
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])
    thr = choose_label_threshold(df[rating_col])
    df["label"] = (df[rating_col] >= thr).astype(int)

    # split (grouped by drug to avoid leakage)
    groups = df["drug_name"].fillna("na").astype(str).values
    X_text = df["text_all"].fillna("")
    y = df["label"].astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, valid_idx = next(gss.split(X_text, y, groups))

    Xtr_s = X_text.iloc[train_idx]   # Series for LR pipeline
    Xva_s = X_text.iloc[valid_idx]
    ytr   = y[train_idx]
    yva   = y[valid_idx]

    # Also keep DF with named column for ColumnTransformer-based pipeline
    Xtr_df = pd.DataFrame({"text_all": Xtr_s.values})
    Xva_df = pd.DataFrame({"text_all": Xva_s.values})

    # ---------- Model A: TF-IDF(word 1–2g) + Logistic Regression ----------
    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=3, max_df=0.9)),
        ("lr",    LogisticRegression(max_iter=400, class_weight="balanced", C=2.0, n_jobs=None))
    ])
    pipe_lr.fit(Xtr_s, ytr)
    prob_lr = pipe_lr.predict_proba(Xva_s)[:,1]
    yhat_lr = (prob_lr >= 0.5).astype(int)

    # ---------- Model B: word + char TF-IDF -> LinearSVC (calibrated) ----------
    # ColumnTransformer needs column names, so we use DataFrame
    featurizer = ColumnTransformer([
        ("w", TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=3, max_df=0.95), "text_all"),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3),               "text_all"),
    ])
    pipe_svc = Pipeline([
        ("feats", featurizer),
        ("svc",   LinearSVC(C=1.0))
    ])
    # Calibrate the whole pipeline (gives predict_proba)
    cal_svc = CalibratedClassifierCV(pipe_svc, method="isotonic", cv=5)
    cal_svc.fit(Xtr_df, ytr)
    prob_svc = cal_svc.predict_proba(Xva_df)[:,1]
    yhat_svc = (prob_svc >= 0.5).astype(int)

    # ---------- Metrics ----------
    def pack(name, y_true, y_pred, y_prob):
        return {
            "model": name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": (None if (y_prob is None or len(np.unique(y_true))<2) else float(safe_auc(y_true, y_prob)))
        }

    m_lr  = pack("text_lr",  yva, yhat_lr,  prob_lr)
    m_svc = pack("text_svc", yva, yhat_svc, prob_svc)

    metrics = {
        "label_threshold_used": float(thr),
        "text_lr":  m_lr,
        "text_svc": m_svc
    }
    (outdir / "text_models_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== TEXT MODEL METRICS ===")
    print(json.dumps(metrics, indent=2))

    # classification reports
    rep_lr  = classification_report(yva, yhat_lr,  digits=3, zero_division=0)
    rep_svc = classification_report(yva, yhat_svc, digits=3, zero_division=0)
    (outdir / "text_models_reports" / "text_lr.txt").write_text(rep_lr + "\n" + str(confusion_matrix(yva, yhat_lr)))
    (outdir / "text_models_reports" / "text_svc.txt").write_text(rep_svc + "\n" + str(confusion_matrix(yva, yhat_svc)))

    # pick best by F1 (tie-breaker: AUC, then accuracy)
    rows = [("lr", m_lr["f1"], m_lr["roc_auc"] or -1, m_lr["accuracy"]),
            ("svc", m_svc["f1"], m_svc["roc_auc"] or -1, m_svc["accuracy"])]
    best_name = sorted(rows, key=lambda t: (t[1], t[2], t[3]), reverse=True)[0][0]

    # save models
    dump(pipe_lr,  outdir / "model_text_lr.joblib")
    dump(cal_svc,  outdir / "model_text_svc.joblib")
    best_path = outdir / "model_text_best.joblib"
    dump(pipe_lr if best_name=="lr" else cal_svc, best_path)
    print(f"\nSaved models:\n - model_text_lr.joblib\n - model_text_svc.joblib\n - model_text_best.joblib (best = {best_name})")

    # validation predictions (for threshold_tuner.py)
    val_out = pd.DataFrame({
        "text_all": Xva_s.values,
        "y_true":   yva,
        "prob_text_lr":  prob_lr,
        "prob_text_svc": prob_svc,
        "y_text_lr":  yhat_lr,
        "y_text_svc": yhat_svc,
    })
    val_path = outdir / "validation_predictions_text.csv"
    val_out.to_csv(val_path, index=False)
    print(f"Wrote {val_path}")

    # top words (from LR for interpretability)
    try:
        vec = pipe_lr.named_steps["tfidf"]
        lr  = pipe_lr.named_steps["lr"]
        feats = vec.get_feature_names_out()
        coefs = lr.coef_[0]
        top_pos_idx = np.argsort(coefs)[-30:][::-1]
        top_neg_idx = np.argsort(coefs)[:30]
        top_words_df = pd.DataFrame({
            "feature": list(feats[top_pos_idx]) + list(feats[top_neg_idx]),
            "weight":  list(coefs[top_pos_idx]) + list(coefs[top_neg_idx]),
            "sign":    ["positive"]*len(top_pos_idx) + ["negative"]*len(top_neg_idx)
        })
        top_words_df.to_csv(outdir / "top_text_features.csv", index=False)
        print("Exported top_text_features.csv")
    except Exception as e:
        print("Skipping top word export:", e)

    print("\nDone.")

if __name__ == "__main__":
    main()
