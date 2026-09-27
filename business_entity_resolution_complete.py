# # Business Entity Resolution — End-to-End Competition Notebook
# 
# **Goal:** high-precision entity matching for the competition's macro F0.5 metric.
# 
# This notebook is intentionally built from scratch rather than depending on the previous notebook state.
# 
# Pipeline:
# 1. Load train/test TSVs
# 2. Normalize names, addresses and countries
# 3. Split training entities by Source-1 entity
# 4. Generate high-recall candidates using several deterministic blocks
# 5. Build fuzzy/string features
# 6. Train LightGBM
# 7. Select threshold using the official per-S1 macro F0.5 validation metric
# 8. Refit on all available training data
# 9. Generate test candidates
# 10. Score candidates and apply precision-oriented per-S1 selection
# 11. Write `matching_results.tsv` and `candidate_pairs.tsv`
# 
# **Important:** no external data is used. A 99+ score cannot be guaranteed before seeing the private leaderboard.
# 


# =========================
# 1. INSTALL / IMPORTS
# =========================

# Run this cell once if packages are missing:
# %pip install pandas numpy scipy scikit-learn rapidfuzz lightgbm tqdm pyarrow

import os
import re
import gc
import math
import json
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE = r"C:\Users\Administrator\Desktop\Team-nexus\dataset"

TRAIN_S1 = os.path.join(BASE, "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE, "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE, "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE, "train", "train_ground_truth.tsv")

TEST_S1 = os.path.join(BASE, "test", "test_source1.tsv")
TEST_S2 = os.path.join(BASE, "test", "test_source2.tsv")
TEST_S3 = os.path.join(BASE, "test", "test_source3.tsv")

OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Environment ready.")



# =========================
# 2. LOAD DATA
# =========================

def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str)
    return df

train_s1 = read_tsv(TRAIN_S1)
train_s2 = read_tsv(TRAIN_S2)
train_s3 = read_tsv(TRAIN_S3)
train_gt = read_tsv(TRAIN_GT)

test_s1 = read_tsv(TEST_S1)
test_s2 = read_tsv(TEST_S2)
test_s3 = read_tsv(TEST_S3)

print("TRAIN")
print("S1:", train_s1.shape)
print("S2:", train_s2.shape)
print("S3:", train_s3.shape)
print("GT:", train_gt.shape)

print("\nTEST")
print("S1:", test_s1.shape)
print("S2:", test_s2.shape)
print("S3:", test_s3.shape)

print("\nColumns:")
print(train_s1.columns.tolist())
print(train_gt.columns.tolist())



# =========================
# 3. NORMALIZATION
# =========================

NAME_REPLACEMENTS = {
    "incorporated":"inc", "corporation":"corp", "company":"co",
    "limited":"ltd", "international":"intl", "association":"assoc",
    "manufacturing":"mfg", "industries":"ind", "industry":"ind",
    "enterprises":"ent", "enterprise":"ent", "technologies":"tech",
    "technology":"tech", "services":"svc", "service":"svc"
}

ADDRESS_REPLACEMENTS = {
    "street":"st", "road":"rd", "avenue":"ave", "boulevard":"blvd",
    "drive":"dr", "lane":"ln", "highway":"hwy", "parkway":"pkwy",
    "place":"pl", "court":"ct", "circle":"cir", "square":"sq",
    "building":"bldg", "suite":"ste", "apartment":"apt", "floor":"fl",
    "north":"n", "south":"s", "east":"e", "west":"w"
}

def normalize_basic(x):
    x = "" if x is None else str(x)
    x = x.lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def normalize_name(x):
    x = normalize_basic(x)
    for a, b in NAME_REPLACEMENTS.items():
        x = re.sub(rf"\b{a}\b", b, x)
    return re.sub(r"\s+", " ", x).strip()

def normalize_address(x):
    x = normalize_basic(x)
    for a, b in ADDRESS_REPLACEMENTS.items():
        x = re.sub(rf"\b{a}\b", b, x)
    return re.sub(r"\s+", " ", x).strip()

def add_normalized_columns(df):
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].map(normalize_basic)

    df["name_compact"] = df["name_norm"].str.replace(" ", "", regex=False)
    df["address_compact"] = df["address_norm"].str.replace(" ", "", regex=False)

    df["name_tokens"] = df["name_norm"].str.split()
    df["address_tokens"] = df["address_norm"].str.split()

    df["address_numbers"] = df["address_norm"].str.findall(r"\d+").str.join("_")

    df["name_first"] = df["name_norm"].str.split().str[0].fillna("").str[:5]
    df["name_last"] = df["name_norm"].str.split().str[-1].fillna("").str[:5]

    df["name_prefix6"] = df["name_compact"].str[:6]
    df["name_suffix6"] = df["name_compact"].str[-6:]

    return df

for _df_name in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    globals()[_df_name] = add_normalized_columns(globals()[_df_name])

print("Normalization complete.")
print(train_s1[["business_name","name_norm","address_norm"]].head(3))



# ## 4. Ground-truth inspection
# 
# The ground-truth file can have different column names depending on the competition export. The next cell detects the Source-1 and matched-entity columns and converts them into a standard representation.
# 


# =========================
# 4. STANDARDIZE GROUND TRUTH
# =========================

def find_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

GT_S1_COL = find_column(train_gt, [
    "source1_entity_id", "source_1_entity_id", "entity_id_s1",
    "source1_id", "source_1_id"
])

GT_MATCH_COL = find_column(train_gt, [
    "matched_entity_ids", "matched_entity_id", "entity_id_s2_or_s3",
    "source2_entity_id", "source3_entity_id", "match_id"
])

if GT_S1_COL is None or GT_MATCH_COL is None:
    print("Ground truth columns:", train_gt.columns.tolist())
    raise ValueError("Could not automatically identify GT columns. Set GT_S1_COL and GT_MATCH_COL manually.")

gt = train_gt[[GT_S1_COL, GT_MATCH_COL]].copy()
gt.columns = ["entity_id_s1", "entity_id_s2_or_s3"]

# Support cells containing multiple matched IDs separated by common delimiters.
gt["entity_id_s2_or_s3"] = gt["entity_id_s2_or_s3"].str.replace(";", "|", regex=False)
gt["entity_id_s2_or_s3"] = gt["entity_id_s2_or_s3"].str.replace(",", "|", regex=False)

gt = (
    gt.assign(entity_id_s2_or_s3=gt["entity_id_s2_or_s3"].str.split("|"))
      .explode("entity_id_s2_or_s3")
)

gt["entity_id_s2_or_s3"] = gt["entity_id_s2_or_s3"].str.strip()
gt = gt[gt["entity_id_s2_or_s3"] != ""].drop_duplicates()

print("GT pairs:", len(gt))
print("Unique S1 in GT:", gt["entity_id_s1"].nunique())
print(gt.head())



# =========================
# 5. ENTITY-LEVEL VALIDATION SPLIT
# =========================

all_s1 = train_s1["entity_id"].drop_duplicates().values

train_ids, val_ids = train_test_split(
    all_s1,
    test_size=0.20,
    random_state=SEED
)

train_ids = set(train_ids)
val_ids = set(val_ids)

tr_s1 = train_s1[train_s1["entity_id"].isin(train_ids)].copy()
va_s1 = train_s1[train_s1["entity_id"].isin(val_ids)].copy()

tr_gt = gt[gt["entity_id_s1"].isin(train_ids)].copy()
va_gt = gt[gt["entity_id_s1"].isin(val_ids)].copy()

print("Train S1:", len(tr_s1))
print("Validation S1:", len(va_s1))
print("Train GT:", len(tr_gt))
print("Validation GT:", len(va_gt))



# =========================
# 6. HIGH-RECALL BLOCK INDEXES
# =========================

def add_to_index(idx, key, value):
    if key is None:
        return
    if not key:
        return
    idx[key].append(value)

def build_indexes(df):
    idx = {
        "name": defaultdict(list),
        "name_country": defaultdict(list),
        "compact": defaultdict(list),
        "compact_country": defaultdict(list),
        "name_prefix": defaultdict(list),
        "name_first": defaultdict(list),
        "name_last": defaultdict(list),
        "number_country": defaultdict(list),
        "address_prefix": defaultdict(list),
        "country": defaultdict(list),
    }

    for r in df.itertuples(index=False):
        eid = r.entity_id
        name = r.name_norm
        country = r.country_norm
        compact = r.name_compact
        nums = r.address_numbers
        first = r.name_first
        last = r.name_last
        np6 = r.name_prefix6
        ap = r.address_compact[:8]

        add_to_index(idx["name"], name, eid)
        if country:
            add_to_index(idx["name_country"], (name, country), eid)
            add_to_index(idx["compact_country"], (compact, country), eid)
            if nums:
                add_to_index(idx["number_country"], (country, nums), eid)

        add_to_index(idx["compact"], compact, eid)
        add_to_index(idx["name_prefix"], np6, eid)
        add_to_index(idx["name_first"], (country, first), eid)
        add_to_index(idx["name_last"], (country, last), eid)
        add_to_index(idx["address_prefix"], (country, ap), eid)

        if country:
            add_to_index(idx["country"], country, eid)

    return idx

print("Building indexes...")
tr_idx_s2 = build_indexes(train_s2)
tr_idx_s3 = build_indexes(train_s3)

print("Indexes ready.")



# =========================
# 7. CANDIDATE GENERATION
# =========================

def candidate_generator(left_df, right_idx, max_per_block=80):
    result = set()

    for r in left_df.itertuples(index=False):
        s1 = r.entity_id
        name = r.name_norm
        compact = r.name_compact
        country = r.country_norm
        nums = r.address_numbers
        first = r.name_first
        last = r.name_last
        np6 = r.name_prefix6
        ap = r.address_compact[:8]

        blocks = []

        # Strong blocks
        if country and name:
            blocks.append(right_idx["name_country"].get((name, country), []))
        if name:
            blocks.append(right_idx["name"].get(name, []))
        if country and compact:
            blocks.append(right_idx["compact_country"].get((compact, country), []))
        if compact:
            blocks.append(right_idx["compact"].get(compact, []))

        # Name-prefix / token blocks
        if np6:
            blocks.append(right_idx["name_prefix"].get(np6, []))
        if country and first:
            blocks.append(right_idx["name_first"].get((country, first), []))
        if country and last:
            blocks.append(right_idx["name_last"].get((country, last), []))

        # Address-supported blocks
        if country and nums:
            blocks.append(right_idx["number_country"].get((country, nums), []))
        if country and ap:
            blocks.append(right_idx["address_prefix"].get((country, ap), []))

        # Add candidates, preserving strong blocks first.
        seen = set()
        for block in blocks:
            if not block:
                continue
            local = []
            for eid in block:
                if eid not in seen:
                    seen.add(eid)
                    local.append(eid)
                    if len(local) >= max_per_block:
                        break
            result.update((s1, eid) for eid in local)

    return pd.DataFrame(
        list(result),
        columns=["entity_id_s1", "entity_id_s2_or_s3"]
    )

print("Generating validation candidates...")

val_cand_s2 = candidate_generator(va_s1, tr_idx_s2)
val_cand_s3 = candidate_generator(va_s1, tr_idx_s3)

print("Validation S2 candidates:", len(val_cand_s2))
print("Validation S3 candidates:", len(val_cand_s3))



# =========================
# 8. CANDIDATE RECALL CHECK
# =========================

def candidate_recall(candidates, gt_df):
    pred = set(
        zip(
            candidates["entity_id_s1"],
            candidates["entity_id_s2_or_s3"]
        )
    )
    true = set(
        zip(
            gt_df["entity_id_s1"],
            gt_df["entity_id_s2_or_s3"]
        )
    )
    covered = len(pred & true)
    return covered / len(true) if true else 0.0

r2 = candidate_recall(val_cand_s2, va_gt[va_gt["entity_id_s2_or_s3"].str.startswith("S2-")])
r3 = candidate_recall(val_cand_s3, va_gt[va_gt["entity_id_s2_or_s3"].str.startswith("S3-")])

print(f"S2 candidate recall: {r2:.4%}")
print(f"S3 candidate recall: {r3:.4%}")

if min(r2, r3) < 0.90:
    print("\nWARNING: candidate recall is below 90%.")
    print("Increase max_per_block or add stronger blocking before trusting final results.")
else:
    print("\nCandidate recall is high enough to continue.")



# =========================
# 9. PAIR FEATURES
# =========================

def jaccard(a, b):
    A = set(str(a).split())
    B = set(str(b).split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def overlap(a, b):
    A = set(str(a).split())
    B = set(str(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))

def number_overlap(a, b):
    A = set(re.findall(r"\d+", str(a)))
    B = set(re.findall(r"\d+", str(b)))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def create_features(pairs, left, right):
    L = left[
        ["entity_id","name_norm","name_compact","address_norm",
         "address_compact","country_norm","address_numbers"]
    ].rename(columns={
        "entity_id":"entity_id_s1",
        "name_norm":"name_s1",
        "name_compact":"compact_s1",
        "address_norm":"address_s1",
        "address_compact":"address_compact_s1",
        "country_norm":"country_s1",
        "address_numbers":"numbers_s1"
    })

    R = right[
        ["entity_id","name_norm","name_compact","address_norm",
         "address_compact","country_norm","address_numbers"]
    ].rename(columns={
        "entity_id":"entity_id_s2_or_s3",
        "name_norm":"name_s2",
        "name_compact":"compact_s2",
        "address_norm":"address_s2",
        "address_compact":"address_compact_s2",
        "country_norm":"country_s2",
        "address_numbers":"numbers_s2"
    })

    x = pairs.merge(L, on="entity_id_s1", how="left")
    x = x.merge(R, on="entity_id_s2_or_s3", how="left")

    ns1 = x["name_s1"].fillna("")
    ns2 = x["name_s2"].fillna("")
    cs1 = x["compact_s1"].fillna("")
    cs2 = x["compact_s2"].fillna("")
    a1 = x["address_s1"].fillna("")
    a2 = x["address_s2"].fillna("")
    ac1 = x["address_compact_s1"].fillna("")
    ac2 = x["address_compact_s2"].fillna("")
    co1 = x["country_s1"].fillna("")
    co2 = x["country_s2"].fillna("")

    f = pd.DataFrame(index=x.index)

    f["name_present"] = ((ns1 != "") & (ns2 != "")).astype("int8")
    f["address_present"] = ((a1 != "") & (a2 != "")).astype("int8")
    f["country_present"] = ((co1 != "") & (co2 != "")).astype("int8")

    f["name_exact"] = ((ns1 == ns2) & (ns1 != "")).astype("int8")
    f["compact_exact"] = ((cs1 == cs2) & (cs1 != "")).astype("int8")
    f["country_exact"] = ((co1 == co2) & (co1 != "")).astype("int8")

    f["name_ratio"] = [ratio(a,b)/100 for a,b in zip(ns1,ns2)]
    f["name_set_ratio"] = [token_set_ratio(a,b)/100 for a,b in zip(ns1,ns2)]
    f["name_sort_ratio"] = [token_sort_ratio(a,b)/100 for a,b in zip(ns1,ns2)]

    f["name_jaccard"] = [jaccard(a,b) for a,b in zip(ns1,ns2)]
    f["name_overlap"] = [overlap(a,b) for a,b in zip(ns1,ns2)]

    f["address_ratio"] = [ratio(a,b)/100 for a,b in zip(a1,a2)]
    f["address_set_ratio"] = [token_set_ratio(a,b)/100 for a,b in zip(a1,a2)]
    f["address_sort_ratio"] = [token_sort_ratio(a,b)/100 for a,b in zip(a1,a2)]

    f["address_jaccard"] = [jaccard(a,b) for a,b in zip(a1,a2)]
    f["address_overlap"] = [overlap(a,b) for a,b in zip(a1,a2)]

    f["number_overlap"] = [number_overlap(a,b) for a,b in zip(a1,a2)]
    f["numbers_exact"] = (
        (x["numbers_s1"].fillna("") != "") &
        (x["numbers_s2"].fillna("") != "") &
        (x["numbers_s1"].fillna("") == x["numbers_s2"].fillna(""))
    ).astype("int8")

    f["name_length_diff"] = (ns1.str.len() - ns2.str.len()).abs().astype("float32")
    f["address_length_diff"] = (a1.str.len() - a2.str.len()).abs().astype("float32")

    f["name_prefix6"] = (
        cs1.str[:6] == cs2.str[:6]
    ).astype("int8")

    f["name_suffix6"] = (
        cs1.str[-6:] == cs2.str[-6:]
    ).astype("int8")

    f["address_prefix8"] = (
        ac1.str[:8] == ac2.str[:8]
    ).astype("int8")

    f["name_address_interaction"] = f["name_set_ratio"] * f["address_set_ratio"]

    f["weighted_similarity"] = (
        0.65 * f["name_set_ratio"] +
        0.35 * f["address_set_ratio"]
    )

    return f.astype(np.float32)

FEATURE_COLS = None
print("Feature function ready.")



# =========================
# 10. BUILD VALIDATION FEATURES
# =========================

# Combine S2 and S3 candidate pairs.
val_pairs = pd.concat([val_cand_s2, val_cand_s3], ignore_index=True).drop_duplicates()

print("Total validation candidate pairs:", len(val_pairs))

# Build features in chunks to avoid unnecessary peak memory.
def features_in_chunks(pairs, left, right, chunk=250_000):
    out = []
    for start in tqdm(range(0, len(pairs), chunk)):
        part = pairs.iloc[start:start+chunk]
        out.append(create_features(part, left, right))
    return pd.concat(out, ignore_index=True)

# Add source indicator before feature extraction.
def add_source_column(pairs):
    pairs = pairs.copy()
    pairs["source_s3"] = pairs["entity_id_s2_or_s3"].str.startswith("S3-").astype("int8")
    return pairs

val_pairs = add_source_column(val_pairs)

# Separate because S2/S3 IDs live in different right tables.
val_s2 = val_pairs[~val_pairs["source_s3"].astype(bool)].copy()
val_s3 = val_pairs[val_pairs["source_s3"].astype(bool)].copy()

val_feat_s2 = features_in_chunks(val_s2, va_s1, train_s2)
val_feat_s3 = features_in_chunks(val_s3, va_s1, train_s3)

val_feat_s2["source_s3"] = 0
val_feat_s3["source_s3"] = 1

val_features = pd.concat([val_feat_s2, val_feat_s3], ignore_index=True)

FEATURE_COLS = [
    c for c in val_features.columns
    if c not in []
]

print("Validation feature matrix:", val_features.shape)
print("Features:", FEATURE_COLS)



# =========================
# 11. VALIDATION LABELS
# =========================

gt_set = set(zip(
    va_gt["entity_id_s1"],
    va_gt["entity_id_s2_or_s3"]
))

val_features["label"] = [
    1 if pair in gt_set else 0
    for pair in zip(
        val_pairs["entity_id_s1"],
        val_pairs["entity_id_s2_or_s3"]
    )
]

# Align source column with feature rows.
# val_features was generated from val_s2 then val_s3.
print("Positive labels:", int(val_features["label"].sum()))
print("Negative labels:", int((val_features["label"] == 0).sum()))



# =========================
# 12. TRAIN LIGHTGBM
# =========================

X = val_features[FEATURE_COLS].fillna(0).astype(np.float32)
y = val_features["label"].values

# Use a balanced sample because candidate pools are highly imbalanced.
pos_idx = np.flatnonzero(y == 1)
neg_idx = np.flatnonzero(y == 0)

rng = np.random.default_rng(SEED)

NEG_PER_POS = 3
max_neg = min(len(neg_idx), len(pos_idx) * NEG_PER_POS)

sample_neg = rng.choice(
    neg_idx,
    size=max_neg,
    replace=False
)

sample_idx = np.concatenate([pos_idx, sample_neg])
rng.shuffle(sample_idx)

X_train = X.iloc[sample_idx]
y_train = y[sample_idx]

print("Training rows:", len(X_train))
print("Positive:", int(y_train.sum()))
print("Negative:", int((y_train == 0).sum()))

model = LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_alpha=0.20,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train)

print("LightGBM training complete.")



# =========================
# 13. VALIDATION PREDICTIONS
# =========================

val_features["probability"] = model.predict_proba(
    val_features[FEATURE_COLS].fillna(0).astype(np.float32)
)[:, 1]

# Reattach pair IDs.
val_scored = val_pairs.copy()
val_scored["probability"] = val_features["probability"].values

print(val_scored["probability"].describe())



# =========================
# 14. OFFICIAL MACRO F0.5
# =========================

def macro_f05(pred_pairs, true_pairs, all_s1_ids, threshold):
    true_by_s1 = (
        true_pairs.groupby("entity_id_s1")["entity_id_s2_or_s3"]
        .apply(set)
        .to_dict()
    )

    pred_by_s1 = (
        pred_pairs[pred_pairs["probability"] >= threshold]
        .groupby("entity_id_s1")["entity_id_s2_or_s3"]
        .apply(set)
        .to_dict()
    )

    scores = []

    for s1_id in all_s1_ids:
        true_set = true_by_s1.get(s1_id, set())
        pred_set = pred_by_s1.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)
            continue

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        if precision == 0 or recall == 0:
            scores.append(0.0)
        else:
            beta2 = 0.25
            f05 = (1 + beta2) * precision * recall / (
                beta2 * precision + recall
            )
            scores.append(f05)

    return float(np.mean(scores))

thresholds = np.arange(0.30, 0.991, 0.01)

scores = []
for t in thresholds:
    scores.append(
        macro_f05(
            val_scored,
            va_gt,
            va_s1["entity_id"].tolist(),
            float(t)
        )
    )

threshold_table = pd.DataFrame({
    "threshold": thresholds,
    "macro_f05": scores
})

best_row = threshold_table.loc[
    threshold_table["macro_f05"].idxmax()
]

BEST_THRESHOLD = float(best_row["threshold"])
BEST_SCORE = float(best_row["macro_f05"])

print(threshold_table.sort_values("macro_f05", ascending=False).head(15))
print()
print("BEST VALIDATION THRESHOLD:", BEST_THRESHOLD)
print("BEST VALIDATION MACRO F0.5:", BEST_SCORE)



# =========================
# 15. PER-S1 PRECISION SAFETY RULE
# =========================

# For a precision-heavy metric, probability threshold alone can still
# allow several weak candidates for one entity. We therefore use:
#   - threshold
#   - top-k by probability
#   - strong exact-match overrides

def select_predictions(scored, threshold, top_k=3):
    x = scored[scored["probability"] >= threshold].copy()

    if x.empty:
        return x

    x = x.sort_values(
        ["entity_id_s1", "probability"],
        ascending=[True, False]
    )

    x = x.groupby("entity_id_s1", sort=False).head(top_k)

    return x.reset_index(drop=True)

val_selected = select_predictions(
    val_scored,
    BEST_THRESHOLD,
    top_k=3
)

selected_score = macro_f05(
    val_selected.assign(probability=1.0),
    va_gt,
    va_s1["entity_id"].tolist(),
    0.5
)

print("Validation selected pairs:", len(val_selected))
print("Selected-rule validation F0.5:", selected_score)



# ## 16. Refit on all training entities
# 
# After the validation threshold/model choice is fixed, train a final model using all available training entities.
# 
# The validation split is not used to choose a different test threshold after this point.
# 


# =========================
# 16. FINAL TRAINING CANDIDATES
# =========================

# Build indexes from ALL S2/S3 training rows.
full_idx_s2 = build_indexes(train_s2)
full_idx_s3 = build_indexes(train_s3)

full_cand_s2 = candidate_generator(train_s1, full_idx_s2)
full_cand_s3 = candidate_generator(train_s1, full_idx_s3)

print("Full S2 candidates:", len(full_cand_s2))
print("Full S3 candidates:", len(full_cand_s3))



# =========================
# 17. FINAL TRAINING FEATURES
# =========================

full_s2_feat = features_in_chunks(full_cand_s2, train_s1, train_s2)
full_s3_feat = features_in_chunks(full_cand_s3, train_s1, train_s3)

full_s2_feat["source_s3"] = 0
full_s3_feat["source_s3"] = 1

full_features = pd.concat([full_s2_feat, full_s3_feat], ignore_index=True)

full_pairs = pd.concat(
    [full_cand_s2, full_cand_s3],
    ignore_index=True
).drop_duplicates()

full_gt_set = set(zip(
    gt["entity_id_s1"],
    gt["entity_id_s2_or_s3"]
))

full_y = np.array([
    1 if p in full_gt_set else 0
    for p in zip(
        full_pairs["entity_id_s1"],
        full_pairs["entity_id_s2_or_s3"]
    )
], dtype=np.int8)

print("Full candidate features:", full_features.shape)
print("Full positives:", int(full_y.sum()))
print("Full negatives:", int((full_y == 0).sum()))



# =========================
# 18. FINAL LIGHTGBM
# =========================

X_full = full_features[FEATURE_COLS].fillna(0).astype(np.float32)

pos = np.flatnonzero(full_y == 1)
neg = np.flatnonzero(full_y == 0)

rng = np.random.default_rng(SEED)

NEG_PER_POS_FINAL = 3
neg_n = min(len(neg), len(pos) * NEG_PER_POS_FINAL)

sample_neg = rng.choice(neg, size=neg_n, replace=False)

final_idx = np.concatenate([pos, sample_neg])
rng.shuffle(final_idx)

final_model = LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_alpha=0.20,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1
)

final_model.fit(
    X_full.iloc[final_idx],
    full_y[final_idx]
)

print("Final model trained.")



# # 19. TEST CANDIDATES
# 
# For the final test run we use the same blocking logic as validation. This is important: candidate generation is part of the model and must not introduce external information.
# 


# =========================
# 19. TEST CANDIDATES
# =========================

test_idx_s2 = build_indexes(test_s2)
test_idx_s3 = build_indexes(test_s3)

print("Generating TEST S2 candidates...")
test_cand_s2 = candidate_generator(test_s1, test_idx_s2, max_per_block=80)

print("Generating TEST S3 candidates...")
test_cand_s3 = candidate_generator(test_s1, test_idx_s3, max_per_block=80)

print("TEST S2 candidates:", len(test_cand_s2))
print("TEST S3 candidates:", len(test_cand_s3))
print("TOTAL:", len(test_cand_s2) + len(test_cand_s3))



# =========================
# 20. TEST FEATURES + SCORING
# =========================

print("Creating TEST S2 features...")
test_feat_s2 = features_in_chunks(test_cand_s2, test_s1, test_s2)

print("Creating TEST S3 features...")
test_feat_s3 = features_in_chunks(test_cand_s3, test_s1, test_s3)

test_feat_s2["source_s3"] = 0
test_feat_s3["source_s3"] = 1

test_feat_s2["probability"] = final_model.predict_proba(
    test_feat_s2[FEATURE_COLS].fillna(0).astype(np.float32)
)[:, 1]

test_feat_s3["probability"] = final_model.predict_proba(
    test_feat_s3[FEATURE_COLS].fillna(0).astype(np.float32)
)[:, 1]

test_scored_s2 = test_cand_s2.copy()
test_scored_s2["probability"] = test_feat_s2["probability"].values

test_scored_s3 = test_cand_s3.copy()
test_scored_s3["probability"] = test_feat_s3["probability"].values

test_scored = pd.concat(
    [test_scored_s2, test_scored_s3],
    ignore_index=True
)

print("Scored test pairs:", len(test_scored))
print(test_scored["probability"].describe())



# =========================
# 21. FINAL TEST SELECTION
# =========================

# Use the threshold learned ONLY from validation.
# Top-k is intentionally small because F0.5 is precision-heavy.

final_predictions = select_predictions(
    test_scored,
    BEST_THRESHOLD,
    top_k=3
)

print("Selected predictions:", len(final_predictions))
print("Unique S1 matched:", final_predictions["entity_id_s1"].nunique())
print(final_predictions.head(20))



# =========================
# 22. STRONG-MATCH OVERRIDE
# =========================

# Exact normalized name + same non-empty country is an extremely strong
# deterministic signal. Add such pairs even if model probability is
# slightly below the learned threshold.

def strong_exact_pairs(left, right):
    r = right[
        (right["name_norm"] != "") &
        (right["country_norm"] != "")
    ][["entity_id","name_norm","country_norm"]].copy()

    l = left[
        (left["name_norm"] != "") &
        (left["country_norm"] != "")
    ][["entity_id","name_norm","country_norm"]].copy()

    x = l.merge(
        r,
        on=["name_norm","country_norm"],
        suffixes=("_s1","_right")
    )

    return x[
        ["entity_id_s1","entity_id_right"]
    ].rename(columns={
        "entity_id_right":"entity_id_s2_or_s3"
    })

strong_s2 = strong_exact_pairs(test_s1, test_s2)
strong_s3 = strong_exact_pairs(test_s1, test_s3)

strong = pd.concat([strong_s2, strong_s3], ignore_index=True)
strong["probability"] = 1.0

final_predictions = pd.concat(
    [final_predictions, strong],
    ignore_index=True
).drop_duplicates(
    ["entity_id_s1","entity_id_s2_or_s3"]
)

# Keep only top 3 again after override.
final_predictions = (
    final_predictions
    .sort_values(["entity_id_s1","probability"], ascending=[True,False])
    .groupby("entity_id_s1", sort=False)
    .head(3)
    .reset_index(drop=True)
)

print("After strong-match override:", len(final_predictions))
print("Matched S1:", final_predictions["entity_id_s1"].nunique())



# =========================
# 23. WRITE matching_results.tsv
# =========================

# Standard output: one row per predicted pair.
matching_results = final_predictions[
    ["entity_id_s1","entity_id_s2_or_s3"]
].copy()

matching_results.to_csv(
    os.path.join(OUTPUT_DIR, "matching_results.tsv"),
    sep="\t",
    index=False
)

# Candidate file with probabilities is useful for the required package.
candidate_pairs = test_scored[
    ["entity_id_s1","entity_id_s2_or_s3","probability"]
].copy()

candidate_pairs.to_csv(
    os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
    sep="\t",
    index=False
)

print("Saved:")
print(os.path.join(OUTPUT_DIR, "matching_results.tsv"))
print(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"))
print("\nSubmission rows:", len(matching_results))



# =========================
# 24. FINAL SANITY CHECKS
# =========================

submission = pd.read_csv(
    os.path.join(OUTPUT_DIR, "matching_results.tsv"),
    sep="\t",
    dtype=str
)

assert submission["entity_id_s1"].notna().all()
assert submission["entity_id_s2_or_s3"].notna().all()

assert not submission.duplicated(
    ["entity_id_s1","entity_id_s2_or_s3"]
).any()

print("Submission shape:", submission.shape)
print("Unique S1 matched:", submission["entity_id_s1"].nunique())
print("Duplicate pairs:", submission.duplicated(
    ["entity_id_s1","entity_id_s2_or_s3"]
).sum())

print("\nSample:")
print(submission.head(20))

print("\nREADY:")
print(os.path.join(OUTPUT_DIR, "matching_results.tsv"))



# ## Notes for score improvement
# 
# If validation candidate recall is below 90%, do **not** tune the LightGBM threshold first. Improve blocking/candidate recall first.
# 
# Useful next experiments, all using only supplied data:
# - increase `max_per_block` from 80 to 120 or 150;
# - add additional normalized-name blocks;
# - add character n-gram nearest-neighbor blocking;
# - train source-specific S2 and S3 models;
# - optimize `top_k` using the official validation metric;
# - add a separate exact/near-exact deterministic rule layer;
# - inspect false positives and false negatives at the S1-entity level.
# 
# Do not use external business databases, APIs, geocoding or internet data if the competition rules prohibit them.
# 
