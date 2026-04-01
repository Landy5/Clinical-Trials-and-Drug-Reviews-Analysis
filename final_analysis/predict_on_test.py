#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from joblib import load

# reuse helpers from training script
from reviews_trials_model import (
    smart_read, normalize_drug, build_trial_features,
    find_review_drug_col, find_trial_drug_col, pick_text_cols
)

ap = argparse.ArgumentParser()
ap.add_argument("--text-model", default="outputs/model_text_only.joblib")
ap.add_argument("--meta-model", default="outputs/model_meta.joblib")
ap.add_argument("--num-feats",  default="outputs/numeric_features.csv")
ap.add_argument("--reviews",    default="final_cleaned_test_data.csv")
ap.add_argument("--trials",     default="cleaned_clinical_trials.csv")
ap.add_argument("--thr-csv",    default="outputs/thresholds_summary.csv")
ap.add_argument("--out",        default="outputs/test_predictions.csv")
args = ap.parse_args()

# load models and numeric feature order
cal_text = load(args.text_model)
meta     = load(args.meta_model)
num_cols = pd.read_csv(args.num_feats)['num_feature'].tolist()

# load data
reviews = smart_read(args.reviews)
trials  = smart_read(args.trials)

# detect cols just like training
review_drug_col = find_review_drug_col(reviews)
trial_drug_col  = find_trial_drug_col(trials)
text_cols       = pick_text_cols(reviews, None)

# build same features
reviews = reviews.rename(columns={review_drug_col: 'drug_name'})
reviews['drug_norm'] = reviews['drug_name'].astype(str).map(normalize_drug)
feats = build_trial_features(trials, trial_drug_col)
df = reviews.merge(feats, on='drug_norm', how='left').fillna(0)

df['text_all'] = reviews[text_cols].fillna('').agg(' '.join, axis=1)

# probabilities
text_prob = cal_text.predict_proba(df['text_all'])[:,1]
meta_X = pd.DataFrame({'text_prob': text_prob})
for c in num_cols:
    if c not in df.columns:
        df[c] = 0
    meta_X[c] = df[c].values

prob = meta.predict_proba(meta_X)[:,1]

# choose tuned threshold for hybrid
thr_df = pd.read_csv(args.thr_csv)
if 'model' in thr_df.columns:
    # prefer any row that mentions "hybrid"
    sub = thr_df[thr_df['model'].str.contains('hybrid', na=False)]
    if len(sub) == 0:
        sub = thr_df
    thr = float(sub.sort_values('f1', ascending=False)['threshold'].iloc[0])
else:
    thr = float(thr_df['threshold'].iloc[0])

pred = (prob >= thr).astype(int)

out = reviews.copy()
out['prob'] = prob
out['pred'] = pred
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(f"Wrote {args.out} using threshold {thr:.2f}")
