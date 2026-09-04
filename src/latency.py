import time
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
m = joblib.load(ROOT / "models" / "app_model.joblib")
vec, clf = m["vectorizer"], m["classifier"]

df = pd.read_parquet(ROOT / "data" / "emails.parquet").sample(200, random_state=0)
texts = (df.subject.fillna("") + " " + df.body_text.fillna("")).str.slice(0, 20000).tolist()

clf.predict_proba(vec.transform([texts[0]]))

t0 = time.perf_counter()
for t in texts:
    clf.predict_proba(vec.transform([t]))
per_email_ms = (time.perf_counter() - t0) / len(texts) * 1000
print(f"mean inference time: {per_email_ms:.2f} ms per email over {len(texts)} emails")