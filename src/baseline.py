import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "emails.parquet")

SPAM_TAGS = r"(\*\*\*SPAM\*\*\*|\[SPAM\]|^SPAM:)\s*"
df["subject"] = df.subject.fillna("").str.replace(SPAM_TAGS, "", regex=True)

LEAK = r"(jose@monkey\.org|monkey\.org|phishing@pot|phishing\.pot)"
df["subject"]   = df.subject.str.replace(LEAK, " ", regex=True)
df["body_text"] = df.body_text.fillna("").str.replace(LEAK, " ", regex=True)

df["text"] = (df.subject + " " + df.body_text).str.slice(0, 20000)

print(f"Loaded {len(df):,} emails")
print(df.groupby(["source_corpus", "label"]).size(), "\n")


def train_eval(X_tr, y_tr, X_te, y_te, label="", show_features=False):
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, y_tr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    r = {
        "experiment": label,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, pred, zero_division=0), 4),
        "f1":        round(f1_score(y_te, pred, zero_division=0), 4),
        "pr_auc":    round(average_precision_score(y_te, proba), 4),
        "confusion": confusion_matrix(y_te, pred).tolist(),
    }

    print(f"\n=== {label} ===")
    for k in ("n_train", "n_test", "precision", "recall", "f1", "pr_auc"):
        print(f"  {k:10}: {r[k]}")
    print(f"  confusion : {r['confusion']}  [[TN FP],[FN TP]]")

    if show_features:
        names = vec.get_feature_names_out()
        top = np.argsort(clf.coef_[0])[-25:][::-1]
        bot = np.argsort(clf.coef_[0])[:25]
        r["top_malicious_features"]  = [names[i] for i in top]
        r["top_legitimate_features"] = [names[i] for i in bot]
        print(f"\n  Top malicious indicators : {r['top_malicious_features'][:15]}")
        print(f"  Top legitimate indicators: {r['top_legitimate_features'][:15]}")

    return r


results = []

# --- 1. RANDOM STRATIFIED -----------------------
Xtr, Xte, ytr, yte = train_test_split(
    df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)
results.append(train_eval(Xtr, ytr, Xte, yte, "1_random_stratified",
                          show_features=True))

# --- 2. LEAVE-ONE-CORPUS-OUT ----------------------------
tr_mask = df.source_corpus.isin(["nazario", "enron"])
te_mask = df.source_corpus.isin(["phishing_pot", "spamassassin_ham"])
results.append(train_eval(
    df[tr_mask].text, df[tr_mask].label,
    df[te_mask].text, df[te_mask].label,
    "2_leave_one_corpus_out"))

# --- 3. TEMPORAL: old -> recent -------------------------------
old_ph = df[(df.label == 1) & (df.source_corpus == "nazario")]
new_ph = df[(df.label == 1) & (df.source_corpus == "phishing_pot")]
legit = df[df.label == 0]
lg_tr, lg_te = train_test_split(legit, test_size=0.5, random_state=42)
tr = pd.concat([old_ph, lg_tr])
te = pd.concat([new_ph, lg_te])
results.append(train_eval(tr.text, tr.label, te.text, te.label,
                          "3_temporal_old_to_new"))

# --- 4. PROVENANCE PROBE --------------------------------------
print("\n" + "#" * 60)
print("# PROVENANCE PROBE -- predicting SOURCE, not the label")
print("#" * 60)

for cls, name, srcs in [
    (1, "malicious",  ["nazario", "phishing_pot"]),
    (0, "legitimate", ["enron", "spamassassin_ham"]),
]:
    sub = df[(df.label == cls) & (df.source_corpus.isin(srcs))].copy()
    sub["src_label"] = (sub.source_corpus == srcs[0]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        sub.text, sub.src_label, test_size=0.2,
        stratify=sub.src_label, random_state=42)
    r = train_eval(Xtr, ytr, Xte, yte, f"4_provenance_probe_{name}")
    r["note"] = f"predicting {srcs[0]} vs {srcs[1]}; high score = leakage risk"
    results.append(r)

out = ROOT / "results" / f"baseline_{datetime.now():%Y%m%d_%H%M}.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n\nSaved -> {out}")

print("\n" + "=" * 60)
print("SUMMARY (F1)")
print("=" * 60)
for r in results:
    print(f"  {r['experiment']:32} {r['f1']}")