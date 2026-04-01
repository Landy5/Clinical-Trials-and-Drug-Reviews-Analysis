#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reviews_trials_model.py (stacked hybrid, NaN-safe, grouped split)

End-to-end pipeline to:
  • load cleaned drug reviews + clinical trials
  • normalize/align drug names
  • engineer rich trial-based features per drug
  • join features onto each review row
  • split by drug (prevents leakage)
  • train: text-only, numeric-only, and a stacked hybrid (text prob + numeric)
  • save outputs (engineered CSV, metrics, predictions, models)

USAGE:
    python reviews_trials_model.py \
        --reviews final_cleaned_train_data.csv \
        --trials  cleaned_clinical_trials.csv \
        --outdir  outputs

Optional flags:
    --rating-col RATINGCOL
    --text-cols benefitsReview sideEffectsReview commentsReview
    --review-drug-col urlDrugName
    --trial-drug-col  drug_name
"""

import re
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)

# text + models
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from joblib import dump


# -------------------- helpers --------------------

def smart_read(path: str) -> pd.DataFrame:
    """Read CSV/TSV by sniffing the delimiter."""
    path = str(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


def normalize_drug(s):
    """Simple normalizer so names match better between reviews & trials."""
    if pd.isna(s):
        return s
    s = str(s).lower()
    # remove common salts/forms/dosage tokens
    s = re.sub(r'\b(hcl|hydrochloride|sr|xr|er|ir|cr|dr|mr|od|mg|mcg|ug|tablet|tablets|tab|capsule|capsules|cap|solution|suspension)\b', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def find_review_drug_col(df: pd.DataFrame) -> str:
    cands = ['urlDrugName', 'drug_name', 'drug', 'Drug', 'medicine', 'medication']
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a drug column in reviews. Columns: {list(df.columns)}")


def find_trial_drug_col(df: pd.DataFrame) -> str:
    cands = ['drug_name', 'drug', 'Drug', 'intervention_name', 'intervention',
             'interventions', 'trial_drug', 'Intervention Name', 'compound']
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a drug column in trials. Columns: {list(df.columns)}")


def detect_rating_col(df: pd.DataFrame) -> str:
    for c in ['rating', 'ratings', 'score', 'stars']:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a rating column in reviews. Columns: {list(df.columns)}")


def pick_text_cols(df: pd.DataFrame, user_cols=None):
    if user_cols:
        exist = [c for c in user_cols if c in df.columns]
        if exist:
            return exist
    # fallbacks
    cands = ['benefitsReview', 'sideEffectsReview', 'commentsReview',
             'review', 'text', 'review_text', 'content', 'comments']
    found = [c for c in cands if c in df.columns]
    if not found:
        raise KeyError("No text columns found. Pass --text-cols or ensure reviews have text fields.")
    return found


def build_trial_features(tr: pd.DataFrame, drug_col: str) -> pd.DataFrame:
    """Group trial rows by normalized drug and compute richer features."""
    tr = tr.copy()
    tr['drug_norm'] = tr[drug_col].astype(str).map(normalize_drug)

    # ----- base counts -----
    feats = tr.groupby('drug_norm').size().rename('n_trials').reset_index()

    # ----- phase features -----
    if 'phase' in tr.columns:
        phase_map = {
            '1':1,'i':1,'phase1':1,
            '2':2,'ii':2,'phase2':2,
            '3':3,'iii':3,'phase3':3,
            '4':4,'iv':4,'phase4':4
        }
        pnorm = (tr['phase'].astype(str)
                    .str.lower()
                    .str.replace(r'\s+','',regex=True))
        pnum = pnorm.map(phase_map)
        tmp_max = tr.assign(_p=pnum).groupby('drug_norm')['_p'].max().rename('max_phase').reset_index()
        feats = feats.merge(tmp_max, on='drug_norm', how='left')
        for k,v in [('n_phase1',1),('n_phase2',2),('n_phase3',3),('n_phase4',4)]:
            tmp = tr.assign(_p=pnum==v).groupby('drug_norm')['_p'].sum().rename(k).reset_index()
            feats = feats.merge(tmp, on='drug_norm', how='left')
    else:
        feats['max_phase'] = 0
        for k in ['n_phase1','n_phase2','n_phase3','n_phase4']:
            feats[k] = 0

    # ----- status features -----
    status_col = None
    for cand in ['overall_status','status','study_status','Status']:
        if cand in tr.columns:
            status_col = cand
            break
    if status_col:
        status = tr[status_col].astype(str).str.lower()
        def add_status(label, needle):
            tmp = tr.assign(_hit=status.str.contains(needle)).groupby('drug_norm')['_hit'].sum().rename(label).reset_index()
            return tmp
        for lab, key in [('n_completed','complet'),
                         ('n_recruiting','recruit'),
                         ('n_terminated','terminat'),
                         ('n_withdrawn','withdraw')]:
            feats = feats.merge(add_status(lab, key), on='drug_norm', how='left')
    else:
        for k in ['n_completed','n_recruiting','n_terminated','n_withdrawn']:
            feats[k] = 0

    # ----- enrollment -----
    enr_cols = [c for c in ['enrollment','enrollment_actual','enrollment_anticipated'] if c in tr.columns]
    if enr_cols:
        en = pd.to_numeric(tr[enr_cols[0]], errors='coerce')
        tmp = (tr.assign(_en=en)
                 .groupby('drug_norm')['_en']
                 .agg(enr_mean='mean', enr_med='median', enr_max='max')
                 .reset_index())
        feats = feats.merge(tmp, on='drug_norm', how='left')
    else:
        feats['enr_mean'] = feats['enr_med'] = feats['enr_max'] = 0

    # ----- recency (yrs since first/last trial) -----
    date_cols = [c for c in ['start_date','study_start_date','start'] if c in tr.columns]
    if date_cols:
        dt = pd.to_datetime(tr[date_cols[0]], errors='coerce')
        tmp = (tr.assign(_dt=dt)
                 .groupby('drug_norm')['_dt']
                 .agg(first_trial='min', last_trial='max')
                 .reset_index())
        feats = feats.merge(tmp, on='drug_norm', how='left')
        for c in ['first_trial','last_trial']:
            feats[c] = (pd.Timestamp('today') - feats[c]).dt.days/365.25
        feats.rename(columns={'first_trial':'first_trial_yrs_ago',
                              'last_trial':'last_trial_yrs_ago'}, inplace=True)
    else:
        feats['first_trial_yrs_ago'] = 0
        feats['last_trial_yrs_ago']  = 0

    # finalize
    num_cols = [c for c in feats.columns if c != 'drug_norm']
    feats[num_cols] = feats[num_cols].fillna(0)
    for c in ['n_trials','n_phase3','n_completed']:
        if c not in feats.columns:
            feats[c] = 0
    feats[['n_trials','n_phase1','n_phase2','n_phase3','n_phase4',
           'n_completed','n_recruiting','n_terminated','n_withdrawn']] = \
        feats[['n_trials','n_phase1','n_phase2','n_phase3','n_phase4',
               'n_completed','n_recruiting','n_terminated','n_withdrawn']].astype(int)
    return feats


def choose_label_threshold(ratings: pd.Series) -> float:
    """Pick a positive/negative threshold that yields both classes."""
    rmax = ratings.max(skipna=True)
    default = 4.0 if rmax <= 5 else 8.0
    y = (ratings >= default).astype(int)
    if y.nunique() == 2:
        return default
    thr = float(ratings.quantile(0.6))
    y = (ratings >= thr).astype(int)
    if y.nunique() == 2:
        return thr
    return float(ratings.median())


def safe_roc_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


# -------------------- main --------------------

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -- load --
    reviews = smart_read(args.reviews)
    trials  = smart_read(args.trials)

    # drop stray index-like columns
    for df in (reviews, trials):
        bad = [c for c in df.columns if str(c).lower().startswith('unnamed')]
        if bad:
            df.drop(columns=bad, inplace=True)

    # -- detect columns --
    review_drug_col = args.review_drug_col or find_review_drug_col(reviews)
    trial_drug_col  = args.trial_drug_col or find_trial_drug_col(trials)
    rating_col      = args.rating_col or detect_rating_col(reviews)
    text_cols       = pick_text_cols(reviews, args.text_cols)

    # -- normalize & join trial features --
    rev = reviews.copy()
    rev = rev.rename(columns={review_drug_col: 'drug_name'})
    rev['drug_norm'] = rev['drug_name'].astype(str).map(normalize_drug)

    trial_feats = build_trial_features(trials, trial_drug_col)
    rev = rev.merge(trial_feats, on='drug_norm', how='left')
    for c in ['n_trials','n_phase3','n_completed']:
        if c not in rev.columns:
            rev[c] = 0
    rev[['n_trials','n_phase3','n_completed']] = rev[['n_trials','n_phase3','n_completed']].fillna(0).astype(int)

    # -- label & text --
    if rating_col not in rev.columns:
        raise KeyError(f"Rating column '{rating_col}' not found in reviews.")
    rev[rating_col] = pd.to_numeric(rev[rating_col], errors='coerce')
    rev = rev.dropna(subset=[rating_col])

    thr = choose_label_threshold(rev[rating_col])
    rev['label'] = (rev[rating_col] >= thr).astype(int)

    for c in text_cols:
        if c not in rev.columns:
            rev[c] = ""
    rev['text_all'] = rev[text_cols].fillna('').agg(' '.join, axis=1)

    # -- save engineered dataset --
    engineered_path = outdir / "engineered_reviews.csv"
    rev.to_csv(engineered_path, index=False)

    # -- numeric feature set (keep everything we engineered if present) --
    base_num = ['n_trials','n_phase3','n_completed']
    extra_num = [c for c in ['max_phase','n_phase1','n_phase2','n_phase4',
                             'n_recruiting','n_terminated','n_withdrawn',
                             'enr_mean','enr_med','enr_max',
                             'first_trial_yrs_ago','last_trial_yrs_ago']
                 if c in rev.columns]
    num_cols = base_num + extra_num

    # Build X/y
    X = rev[['text_all'] + num_cols].copy()
    y = rev['label'].astype(int).values

    # -- grouped split: keep each drug in only one split (prevents leakage) --
    groups = rev['drug_name'].fillna('na').astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(X, y, groups))
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # ---------- MODELS ----------

    # 1) TEXT-ONLY: Calibrated LinearSVC (strong text performance + probabilities)
    #    Using word 1–2grams; feel free to try char_wb 3–5grams later.
    pipe_text_svc = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=3, max_df=0.9)),
        ('svc',   LinearSVC(class_weight='balanced'))
    ])
    cal_text = CalibratedClassifierCV(pipe_text_svc, method='isotonic', cv=5)
    cal_text.fit(X_train['text_all'], y_train)
    p_text = cal_text.predict(X_valid['text_all'])
    prob_text = cal_text.predict_proba(X_valid['text_all'])[:, 1]
    auc_text = safe_roc_auc(y_valid, prob_text)

    # 2) NUMERIC-ONLY: NaN-safe booster
    pipe_num = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    HistGradientBoostingClassifier(random_state=42))
    ])
    pipe_num.fit(X_train[num_cols], y_train)
    p_num = pipe_num.predict(X_valid[num_cols])
    # HistGB has predict_proba with log_loss; otherwise map decision_function to sigmoid
    if hasattr(pipe_num, "predict_proba"):
        prob_num = pipe_num.predict_proba(X_valid[num_cols])[:, 1]
    else:
        z = pipe_num.decision_function(X_valid[num_cols])
        prob_num = 1.0 / (1.0 + np.exp(-z))
    auc_num = safe_roc_auc(y_valid, prob_num)

    # 3) STACKED HYBRID: meta-model on [text probability + numeric features]
    text_prob_train = cal_text.predict_proba(X_train['text_all'])[:, 1]
    meta_train = pd.DataFrame({'text_prob': text_prob_train})
    for c in num_cols:
        meta_train[c] = X_train[c].values

    meta_valid = pd.DataFrame({'text_prob': prob_text})
    for c in num_cols:
        meta_valid[c] = X_valid[c].values

    meta = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    HistGradientBoostingClassifier(random_state=42))
    ])
    meta.fit(meta_train, y_train)
    p_h = meta.predict(meta_valid)
    if hasattr(meta, "predict_proba"):
        prob_h = meta.predict_proba(meta_valid)[:, 1]
    else:
        z = meta.decision_function(meta_valid)
        prob_h = 1.0 / (1.0 + np.exp(-z))
    auc_h = safe_roc_auc(y_valid, prob_h)

    # ---------- METRICS ----------
    def pack_metrics(y_true, y_pred, auc):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": (None if (auc is None or np.isnan(auc)) else float(auc))
        }

    metrics = {
        "label_threshold_used": float(thr),
        "text_only":    pack_metrics(y_valid, p_text, auc_text),
        "numeric_only": pack_metrics(y_valid, p_num,  auc_num),
        "hybrid":       pack_metrics(y_valid, p_h,    auc_h)   # stacked hybrid
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(metrics, indent=2))

    cr = classification_report(y_valid, p_h, digits=3, zero_division=0)
    cm = confusion_matrix(y_valid, p_h)
    (outdir / "hybrid_classification_report.txt").write_text(cr + "\n" + str(cm))
    print("\nHybrid (stacked) classification report:\n", cr)
    print("Hybrid (stacked) confusion matrix:\n", cm)

    # ---------- SAVE VALIDATION PREDICTIONS (for threshold_tuner.py) ----------
    preds = X_valid.copy()
    preds['y_true']      = y_valid
    preds['y_text']      = p_text
    preds['y_num']       = p_num
    preds['y_h']         = p_h
    preds['prob_text']   = prob_text
    preds['prob_num']    = prob_num
    preds['prob_hybrid'] = prob_h
    preds_path = outdir / "validation_predictions.csv"
    preds.to_csv(preds_path, index=False)
    print("Wrote", preds_path)

    # ---------- SAVE MODELS & FEATURE LIST ----------
    dump(cal_text, outdir / "model_text_only.joblib")     # calibrated SVC
    dump(pipe_num, outdir / "model_numeric_only.joblib")  # HGB on numeric
    dump(meta,     outdir / "model_meta.joblib")          # stacked meta

    (pd.Series(num_cols).to_frame('num_feature')
       .to_csv(outdir / "numeric_features.csv", index=False))

    # Optional: top words from a quick LR fit (proxy for interpretability)
    try:
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=3, max_df=0.9)
        X_vec = vec.fit_transform(X_train['text_all'])
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=300, class_weight='balanced')
        lr.fit(X_vec, y_train)
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
    except Exception as e:
        print("Skipping top-word export:", e)

    print(f"\nSaved:\n - {engineered_path}\n - {preds_path}\n - metrics.json\n"
          f" - models (*.joblib)\n - numeric_features.csv\n - top_text_features.csv (if available)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default="final_cleaned_train_data.csv",
                        help="Path to cleaned reviews CSV/TSV (with rating + review text).")
    parser.add_argument("--trials", default="cleaned_clinical_trials.csv",
                        help="Path to clinical trials CSV/TSV.")
    parser.add_argument("--outdir", default="outputs", help="Directory to write outputs.")
    parser.add_argument("--rating-col", dest="rating_col", default=None, help="Name of rating column (auto-detected if omitted).")
    parser.add_argument("--text-cols", nargs="*", default=None,
                        help="Space-separated list of text columns to combine. Example: --text-cols benefitsReview sideEffectsReview commentsReview")
    parser.add_argument("--review-drug-col", dest="review_drug_col", default=None, help="Drug column in reviews (auto-detected if omitted).")
    parser.add_argument("--trial-drug-col", dest="trial_drug_col", default=None, help="Drug column in trials (auto-detected if omitted).")
    args = parser.parse_args()
    main(args)
