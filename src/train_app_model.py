"""
train_app_model.py -- train the deployment model for the app and save it.
Cleaning and parameters are identical to baseline.py, so the app matches the
experiments. The model is trained on the FULL corpus. The 0.3/0.7 thresholds come from the cross-corpus sweep.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "emails.parquet")

SPAM_TAGS = r"(\*\*\*SPAM\*\*\*|\[SPAM\]|^SPAM:)\s*"
df["subject"] = df.subject.fillna("").str.replace(SPAM_TAGS, "", regex=True)
LEAK = r"(jose@monkey\.org|monkey\.org|phishing@pot|phishing\.pot)"
df["subject"]   = df.subject.str.replace(LEAK, " ", regex=True)
df["body_text"] = df.body_text.fillna("").str.replace(LEAK, " ", regex=True)
df["text"] = (df.subject + " " + df.body_text).str.slice(0, 20000)

vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                      sublinear_tf=True, min_df=2)
X = vec.fit_transform(df.text)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X, df.label)

out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump({"vectorizer": vec, "classifier": clf}, out / "app_model.joblib")
print("Saved:", out / "app_model.joblib")
print("Trained on", X.shape[0], "emails | vocabulary size", len(vec.vocabulary_))