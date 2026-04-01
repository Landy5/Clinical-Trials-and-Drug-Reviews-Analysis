# fetch_trials_from_ucidruglib.py
"""
Pulls drug names from UCI Drug Review (DrugLib.com) via ucimlrepo,
then fetches COMPLETED clinical trials for each drug from ClinicalTrials.gov v2 API.
No intermediate files are written; results are kept in-memory as a pandas DataFrame.

How to run:
    python3 fetch_trials_from_ucidruglib.py
"""

import re
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from ucimlrepo import fetch_ucirepo


# ---------------------------- helpers ---------------------------- #

def normalize_drug_name(name: str) -> str:
    """Simple normalization to make brand-like tokens usable in search."""
    n = str(name).strip()
    n = n.replace("-", " ").replace("_", " ")
    n = re.sub(r"[®™()*/,.:;\"'`]+", " ", n)     # strip common punctuation/symbols
    n = re.sub(r"\s+", " ", n).strip()
    # keep original capitalization hints but ensure first letter upper for readability
    return n.title()


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    """Safe nested-get for JSON dicts."""
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def extract_study_row(study: Dict[str, Any], queried_drug: str) -> Dict[str, Any]:
    """Flatten a few useful fields from the v2 Study JSON."""
    proto = study.get("protocolSection", {}) or {}
    ident = proto.get("identificationModule", {}) or {}
    status = proto.get("statusModule", {}) or {}
    design = proto.get("designModule", {}) or {}
    conds = proto.get("conditionsModule", {}) or {}
    inter = proto.get("interventionsModule", {}) or {}
    results = study.get("resultsSection", {}) or {}

    # IDs & titles
    nct_id = ident.get("nctId")
    brief_title = ident.get("briefTitle") or safe_get(proto, ["descriptionModule", "briefSummary"])

    # Phases can be str (e.g., "PHASE4") or list (["PHASE3","PHASE4"])
    phases = design.get("phases")
    if isinstance(phases, list):
        phase_str = ", ".join(phases)
    else:
        phase_str = phases

    # Conditions & Interventions (names only)
    conditions = conds.get("conditions") or []
    interventions = []
    for itm in inter.get("interventions", []) or []:
        nm = itm.get("name")
        if nm:
            interventions.append(nm)

    # Presence of results (many analyses require results to compute AEs, etc.)
    has_results = bool(results)

    # Try to summarize adverse events counts if present (best-effort; structure varies)
    ae = results.get("adverseEventsModule", {}) or {}
    # Simple flags/counts, if available
    total_serious = safe_get(ae, ["seriousEvents", "totalNumberAffected"])
    total_other = safe_get(ae, ["otherEvents", "totalNumberAffected"])

    return {
        "queriedDrug": queried_drug,
        "nctId": nct_id,
        "briefTitle": brief_title,
        "overallStatus": status.get("overallStatus"),
        "studyType": design.get("studyType"),
        "phases": phase_str,
        "conditions": "; ".join(conditions) if conditions else None,
        "interventions": "; ".join(interventions) if interventions else None,
        "hasResults": has_results,
        "ae_total_serious": total_serious,
        "ae_total_other": total_other,
        # Keep raw fields that are often useful downstream:
        "startDate": safe_get(proto, ["statusModule", "startDateStruct", "date"]),
        "primaryCompletionDate": safe_get(proto, ["statusModule", "primaryCompletionDateStruct", "date"]),
        "completionDate": safe_get(proto, ["statusModule", "completionDateStruct", "date"]),
        "sponsor": safe_get(proto, ["sponsorsModule", "leadSponsor", "name"]),
    }


def fetch_completed_trials_for_term(term: str, pause: float = 0.25) -> List[Dict[str, Any]]:
    """
    Fetch all COMPLETED studies for a search term from ClinicalTrials.gov v2 API.
    Handles pagination via nextPageToken. Returns raw 'studies' JSON items.
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    headers = {"accept": "application/json"}

    # Base params: completed only, JSON format, max page size
    params = {
        "format": "json",
        "query.term": term,
        "filter.overallStatus": "COMPLETED",
        "pageSize": 100,
    }

    studies: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    while True:
        this_params = dict(params)
        if next_token:
            this_params["pageToken"] = next_token

        resp = requests.get(url, headers=headers, params=this_params, timeout=60)
        if resp.status_code != 200:
            print(f"[WARN] API error for '{term}': {resp.status_code}")
            break

        payload = resp.json() or {}
        batch = payload.get("studies", []) or []
        studies.extend(batch)

        next_token = payload.get("nextPageToken")
        if not next_token:
            break

        time.sleep(pause)  # polite pause between pages

    return studies


# ---------------------------- main pipeline ---------------------------- #

def get_ucidruglib_names() -> List[str]:
    """Load UCI Drug Review (DrugLib.com) and return normalized unique drug names from 'urlDrugName'."""
    ds = fetch_ucirepo(id=461)  # Drug Reviews (DrugLib.com)
    feats = ds.data.features.copy()
    if "urlDrugName" not in feats.columns:
        raise KeyError("Expected 'urlDrugName' column in the UCI DrugLib features table.")

    names = feats["urlDrugName"].dropna().unique().tolist()
    normalized = sorted({normalize_drug_name(n) for n in names if str(n).strip()})
    return normalized


def fetch_trials_for_ucidruglib_drugs(limit: Optional[int] = None) -> pd.DataFrame:
    """
    End-to-end:
      1) Get drug names from UCI DrugLib dataset.
      2) Query ClinicalTrials.gov for COMPLETED studies for each drug name (in-memory).
      3) Return a tidy pandas DataFrame of study summaries.
    Set `limit` to a small integer during testing to avoid long runs.
    """
    drugs = get_ucidruglib_names()
    if limit is not None:
        drugs = drugs[:limit]

    print(f"Total unique drugs found in UCI DrugLib: {len(drugs)}")

    rows: List[Dict[str, Any]] = []
    seen_nct: set = set()  # de-duplicate across overlapping queries, per study

    for i, drug in enumerate(drugs, start=1):
        print(f"[{i}/{len(drugs)}] Fetching trials for: {drug} ...")
        raw_studies = fetch_completed_trials_for_term(drug)

        for st in raw_studies:
            # Deduplicate on NCT ID (keep first appearance)
            nct = safe_get(st, ["protocolSection", "identificationModule", "nctId"])
            if not nct or nct in seen_nct:
                # Still keep the row if you want 1 study per queried drug.
                # Here we dedupe to maintain unique trials – change logic if needed.
                continue
            seen_nct.add(nct)

            rows.append(extract_study_row(st, queried_drug=drug))

        # polite pause between different drug queries to avoid hammering the API
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    # Optional: sort for readability
    if not df.empty and "nctId" in df.columns:
        df = df.sort_values(["queriedDrug", "nctId"]).reset_index(drop=True)

    print(f"Total unique COMPLETED trials fetched: {len(df)}")
    return df


if __name__ == "__main__":
    # Tip: during your first run, use a small limit to verify everything is working.
    trials_df = fetch_trials_for_ucidruglib_drugs(limit=None)  # e.g., limit=20 for a quick test
    # Keep everything in-memory. Quick peek:
    print(trials_df.head(10))
    print(trials_df.info())
    # Save the DataFrame to a CSV file
    trials_df.to_csv("clinical_trials_for_drugs.csv", index=False)
    print("Data saved to clinical_trials_for_drugs.csv")


