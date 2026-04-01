
"""
Compare WebMD review text with ClinicalTrials.gov "conditions" for one drug.

- Loads your reviews CSV (columns seen: Date, Overall Rating, Effectiveness, Ease of Use, Satisfaction, Textual Review)
- Loads clinical trials CSV (columns seen: queriedDrug, conditions, phases, overallStatus, ...)
- Preprocesses text (lowercase, alphas only, stopword removal)
- Builds a TF-IDF space on BOTH review texts and trial conditions (fit once on combined corpus)
- Computes cosine similarity review <-> each condition phrase
- Records top-K matching conditions per review + similarity scores
- Plots:
  1) Histogram of review ratings
  2) Histogram of similarity scores (all matches)
  3) Bar chart of trial phases for the drug (if available)

Outputs:
- reviews_with_matched_conditions_and_similarity.csv
"""

import argparse
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn / nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional NLTK stopwords (graceful fallback if not present)
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
except Exception:
    stopwords = None

# --------- helpers ---------

def preprocess_text(text: str) -> str:
    """Lower, keep letters/spaces, remove stopwords."""
    if pd.isna(text):
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if stopwords is not None:
        sw = set(stopwords.words("english"))
        t = " ".join(w for w in t.split() if w not in sw)
    return t

def split_conditions(cell: str) -> list:
    """
    Split conditions string into tokens by common separators and clean.
    e.g., "Schizophrenia; Schizoaffective Disorder" -> ["Schizophrenia", "Schizoaffective Disorder"]
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    parts = re.split(r"[;,/|]+", str(cell))
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def build_condition_corpus(trials_for_drug: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the 'conditions' column into a row-per-condition frame with both raw and processed text.
    """
    rows = []
    for _, r in trials_for_drug.iterrows():
        for cond in split_conditions(r.get("conditions", "")):
            rows.append({"condition_raw": cond, "condition_proc": preprocess_text(cond)})
    cond_df = pd.DataFrame(rows)
    if cond_df.empty:
        # ensure non-empty frame with placeholders so downstream doesn't crash
        cond_df = pd.DataFrame([{"condition_raw": "", "condition_proc": ""}])
    # drop true empties after preprocess if any
    cond_df["condition_proc"].fillna("", inplace=True)
    return cond_df.drop_duplicates().reset_index(drop=True)

# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="abilify_oral_data.csv", help="Path to WebMD reviews CSV")
    ap.add_argument("--trials",  default="cleaned_clinical_trials.csv", help="Path to cleaned clinical trials CSV")
    ap.add_argument("--drug",    default="Abilify", help="Drug name to analyze (matches trials queriedDrug)")
    ap.add_argument("--topk",    type=int, default=3, help="How many best-matching conditions to keep per review")
    ap.add_argument("--minsim",  type=float, default=0.10, help="Minimum cosine similarity to keep a match")
    args = ap.parse_args()

    # Load data
    reviews = pd.read_csv(args.reviews)
    trials  = pd.read_csv(args.trials)

    # Sanity: standardize column names present in your files
    # Reviews columns you showed:
    #   Date, Overall Rating, Effectiveness, Ease of Use, Satisfaction, Textual Review
    required_review_cols = ["Textual Review", "Overall Rating", "Date"]
    for c in required_review_cols:
        if c not in reviews.columns:
            raise KeyError(f"Reviews CSV missing required column: {c}")

    # Trials columns you showed:
    #   queriedDrug, conditions, phases, overallStatus, ...
    if "queriedDrug" not in trials.columns:
        raise KeyError("Trials CSV must contain 'queriedDrug'.")
    if "conditions" not in trials.columns:
        # make empty if truly not present, to keep pipeline working
        trials["conditions"] = ""

    # Filter trials by the requested drug
    trials_d = trials[trials["queriedDrug"].astype(str).str.strip().str.lower()
                      == args.drug.lower()].copy()
    if trials_d.empty:
        warnings.warn(f"No trials found for drug '{args.drug}'. Continuing with empty trials.", RuntimeWarning)

    # Build per-condition DF
    cond_df = build_condition_corpus(trials_d)

    # Prepare reviews: we **don't** rely on a 'Drug Name' column because it isn't in your CSV.
    # If you want to restrict to reviews that explicitly mention the drug, uncomment below.
    # reviews = reviews[reviews["Textual Review"].str.contains(args.drug, case=False, na=False)].copy()

    # Clean ratings to numeric (may have blanks)
    reviews["Overall Rating"] = pd.to_numeric(reviews["Overall Rating"], errors="coerce")

    # Preprocess review text
    reviews["text_proc"] = reviews["Textual Review"].astype(str).map(preprocess_text)

    # Drop rows with truly empty processed text (no words)
    reviews = reviews[reviews["text_proc"].str.len() > 0].reset_index(drop=True)

    if reviews.empty:
        raise ValueError("After preprocessing, no non-empty review texts remain.")

    # ----- Vectorize on combined corpus to keep vocabulary consistent -----
    # Fit on union of review texts + condition phrases
    joint_corpus = pd.concat([reviews["text_proc"], cond_df["condition_proc"]], axis=0).tolist()

    vectorizer = TfidfVectorizer(min_df=2)  # min_df=2 to reduce noise; adjust as needed
    X = vectorizer.fit_transform(joint_corpus)

    # Split back out: first N rows are reviews; remaining are conditions
    n_reviews = reviews.shape[0]
    X_reviews   = X[:n_reviews]
    X_conditions = X[n_reviews:]

    # Edge case: if there are literally no non-empty condition phrases, create a zero-matrix
    if X_conditions.shape[0] == 0:
        # Make a 1-row zero vector to avoid shape errors; we will produce no matches later
        X_conditions = np.zeros((1, X_reviews.shape[1]), dtype=float)

    # ----- Similarity -----
    sim = cosine_similarity(X_reviews, X_conditions)  # shape: (#reviews, #conditions)

    # Confirm shapes align with our review DF indexing:
    assert sim.shape[0] == len(reviews), "Similarity rows != # of reviews (unexpected)."

    # For each review, collect top-k condition matches above threshold
    topk = max(1, int(args.topk))
    mins = float(args.minsim)

    matched_conditions = []
    matched_scores     = []

    for i in range(len(reviews)):
        row_scores = sim[i]  # 1D array length = #conditions
        if len(row_scores.shape) == 0:
            # no conditions
            matched_conditions.append([])
            matched_scores.append([])
            continue

        # Get indices sorted by similarity, desc
        order = np.argsort(-row_scores)
        picks = []
        pscores = []

        for j in order[:topk*3]:  # check a little deeper; we'll filter by threshold
            score = float(row_scores[j])
            if score >= mins:
                # map back to condition text
                cond_text = cond_df.iloc[j]["condition_raw"]
                picks.append(cond_text)
                pscores.append(score)
                if len(picks) >= topk:
                    break

        matched_conditions.append(picks)
        matched_scores.append(pscores)

    # Attach to frame (lengths match exactly; no broadcast issues)
    reviews = reviews.assign(
        matched_conditions=matched_conditions,
        similarity_scores=matched_scores
    )

    # ----- Simple stats -----
    avg_rating = reviews["Overall Rating"].mean()
    print(f"\nAverage Overall Rating (reviews): {avg_rating:.3f}" if not np.isnan(avg_rating)
          else "\nAverage Overall Rating (reviews): NA")

    # ----- Visuals -----
    sns.set_theme()

    # 1) Ratings distribution (if ratings present)
    if reviews["Overall Rating"].notna().sum() > 0:
        plt.figure(figsize=(7,5))
        sns.histplot(reviews["Overall Rating"].dropna(), bins=12, kde=True)
        plt.title(f"{args.drug}: Review Ratings Distribution")
        plt.xlabel("Overall Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # 2) Similarity score distribution (flatten list-of-lists)
    flat_scores = [s for sub in matched_scores for s in sub]
    if len(flat_scores) > 0:
        plt.figure(figsize=(7,5))
        sns.histplot(flat_scores, bins=20, kde=True)
        plt.title(f"{args.drug}: Review↔Condition Similarity (TF-IDF cosine)")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        print("No similarity scores above threshold; try lowering --minsim.")

    # 3) Trial phases bar chart (if phases present)
    if "phases" in trials_d.columns and trials_d["phases"].notna().any():
        phases_counts = (trials_d["phases"]
                         .astype(str)
                         .str.upper()
                         .str.replace(r"\s+", "", regex=True)
                         .value_counts())
        if len(phases_counts):
            plt.figure(figsize=(7,5))
            sns.barplot(x=phases_counts.index, y=phases_counts.values)
            plt.title(f"{args.drug}: Trial Phases (counts)")
            plt.xlabel("Phase")
            plt.ylabel("Count")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()

    # ----- Save results -----
    out_csv = "reviews_with_matched_conditions_and_similarity.csv"
    reviews[
        ["Date", "Overall Rating", "Textual Review", "matched_conditions", "similarity_scores"]
    ].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # also print a quick sample of matches
    print("\nSample matches:")
    print(reviews[["Date", "matched_conditions", "similarity_scores"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
