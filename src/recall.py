"""
recall.py -- improving cross-source recall.

Two levers: class_weight="balanced" (penalise missed attacks) and a
threshold sweep (trade precision for recall)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "emails.parquet")

SPAM_TAGS = r"(\*\*\*SPAM\*\*\*|\[SPAM\]|^SPAM:)\s*"
df["subject"] = df.subject.fillna("").str.replace(SPAM_TAGS, "", regex=True)
LEAK = r"(jose@monkey\.org|monkey\.org|phishing@pot|phishing\.pot)"
df["subject"]   = df.subject.str.replace(LEAK, " ", regex=True)
df["body_text"] = df.body_text.fillna("").str.replace(LEAK, " ", regex=True)
df["text"] = (df.subject + " " + df.body_text).str.slice(0, 20000)

tr = df[df.source_corpus.isin(["nazario", "enron"])]
te = df[df.source_corpus.isin(["phishing_pot", "spamassassin_ham"])]
print(f"train {len(tr):,}  test {len(te):,}\n")

vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                      sublinear_tf=True, min_df=2)
Xtr = vec.fit_transform(tr.text)
Xte = vec.transform(te.text)
yte = te.label.values

def sweep(clf, title):
    clf.fit(Xtr, tr.label)
    proba = clf.predict_proba(Xte)[:, 1]
    print(f"=== {title} ===")
    print(f"  {'thresh':>7} {'prec':>7} {'recall':>7} {'f1':>7}")
    for t in (0.5, 0.4, 0.3, 0.2, 0.1):
        pred = (proba >= t).astype(int)
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)
        f = f1_score(yte, pred, zero_division=0)
        print(f"  {t:>7} {p:>7.3f} {r:>7.3f} {f:>7.3f}")
    print()
sweep(LogisticRegression(max_iter=1000),
      "default (no class weight)")
sweep(LogisticRegression(max_iter=1000, class_weight="balanced"),
      "class_weight=balanced")