"""
app.py - Streamlit phishing detector.
Get Allow / Warn / Quarantine with the probability, a threshold band, and the terms behind the decision.
Parsing matches data.py; cleaning matches baseline.py; thresholds from section 4.8.
"""
import re
import email, email.policy
from pathlib import Path

import joblib
import streamlit as st
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "app_model.joblib"

WARN, QUARANTINE = 0.3, 0.7 
SPAM_TAGS = r"(\*\*\*SPAM\*\*\*|\[SPAM\]|^SPAM:)\s*"
LEAK = r"(jose@monkey\.org|monkey\.org|phishing@pot|phishing\.pot)"


@st.cache_resource
def load_model():
    m = joblib.load(MODEL_PATH)
    return m["vectorizer"], m["classifier"]


def html_to_text(html):
    try:
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def extract_body(msg):        
    text_part, html_part = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue
            ctype = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            except Exception:
                continue
            if ctype == "text/plain" and not text_part:
                text_part = decoded
            elif ctype == "text/html" and not html_part:
                html_part = decoded
    else:
        try:
            payload = msg.get_payload(decode=True)
            decoded = payload.decode(msg.get_content_charset() or "utf-8", errors="replace") if payload else ""
        except Exception:
            decoded = str(msg.get_payload())
        if msg.get_content_type() == "text/html":
            html_part = decoded
        else:
            text_part = decoded
    if not text_part and html_part:
        text_part = html_to_text(html_part)
    return text_part.strip(), html_part


def clean(subject, body):
    subject = re.sub(SPAM_TAGS, "", subject or "")
    subject = re.sub(LEAK, " ", subject)
    body = re.sub(LEAK, " ", body or "")
    return (subject + " " + body)[:20000]


def top_terms(vec, clf, text, k=8):
    x = vec.transform([text])
    contrib = x.multiply(clf.coef_[0]).tocoo()
    names = vec.get_feature_names_out()
    pairs = sorted(zip(contrib.col.tolist(), contrib.data.tolist()), key=lambda p: -p[1])
    return [(names[c], round(float(v), 3)) for c, v in pairs[:k] if v > 0]


def score_bar_html(prob):
    pct = prob * 100
    return f"""
    <div style="position:relative; margin:26px 0 8px 0;">
      <div style="display:flex; height:34px; border-radius:6px; overflow:hidden;
                  font-size:12px; font-weight:600;">
        <div style="width:30%; background:#2e7d32; color:#fff; display:flex;
                    align-items:center; justify-content:center;">Allow</div>
        <div style="width:40%; background:#f9a825; color:#000; display:flex;
                    align-items:center; justify-content:center;">Warn</div>
        <div style="width:30%; background:#c62828; color:#fff; display:flex;
                    align-items:center; justify-content:center;">Quarantine</div>
      </div>
      <div style="position:absolute; left:{pct}%; top:-6px; height:46px; width:3px;
                  background:#ffffff;"></div>
      <div style="position:absolute; left:{pct}%; top:-26px; transform:translateX(-50%);
                  font-size:13px; font-weight:700; color:#fff; white-space:nowrap;">
        {prob:.2f}
      </div>
      <div style="position:relative; height:16px; margin-top:3px; font-size:11px; color:#888;">
        <span style="position:absolute; left:0;">0.0</span>
        <span style="position:absolute; left:30%; transform:translateX(-50%);">0.3</span>
        <span style="position:absolute; left:70%; transform:translateX(-50%);">0.7</span>
        <span style="position:absolute; right:0;">1.0</span>
      </div>
    </div>
    """


vec, clf = load_model()

st.title("Phishing detector")
st.caption("Paste an email or upload an .eml file. Headers are not used; the "
           "model reads the subject and body only. Below 0.3 the email passes; "
           "0.3 to 0.7 is shown as a warning; above 0.7 it is quarantined.")

uploaded = st.file_uploader("Upload .eml", type=["eml"])
subject_in = st.text_input("Subject")
body_in = st.text_area("Body", height=220)

if uploaded is not None:
    msg = email.message_from_bytes(uploaded.getvalue(), policy=email.policy.default)
    subject_in = str(msg.get("Subject", ""))
    body_in, _ = extract_body(msg)
    st.info("Loaded from uploaded .eml")

if st.button("Check") and (subject_in or body_in):
    text = clean(subject_in, body_in)
    prob = float(clf.predict_proba(vec.transform([text]))[0, 1])
    if prob >= QUARANTINE:
        st.error(f"QUARANTINE  —  score {prob:.2f}")
    elif prob >= WARN:
        st.warning(f"WARNING  —  score {prob:.2f}")
    else:
        st.success(f"ALLOW  —  score {prob:.2f}")

    st.markdown(score_bar_html(prob), unsafe_allow_html=True)

    terms = top_terms(vec, clf, text)
    if terms:
        st.write("Terms pushing this email toward malicious:")
        st.table({"term": [t for t, _ in terms], "weight": [w for _, w in terms]})
    else:
        st.write("No strongly malicious terms; the score rests on the absence of "
                 "legitimate signals.")