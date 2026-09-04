import tarfile
import random
import email
import email.policy
import mailbox
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"



def html_to_text(html: str) -> str:
    try:
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def extract_body(msg) -> tuple[str, str]:
    text_part, html_part = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain" and not text_part:
                text_part = decoded
            elif ctype == "text/html" and not html_part:
                html_part = decoded
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            decoded = str(msg.get_payload())
        if msg.get_content_type() == "text/html":
            html_part = decoded
        else:
            text_part = decoded

    if not text_part and html_part:
        text_part = html_to_text(html_part)
    return text_part.strip(), html_part


def msg_to_record(msg, label: int, source: str) -> dict | None:
    try:
        body, html = extract_body(msg)
        if len(body) < 5:
            return None
        try:
            date = email.utils.parsedate_to_datetime(msg.get("Date"))
        except Exception:
            date = None

        def safe(field):
            try:
                return str(msg.get(field, ""))
            except Exception:
                return ""

        return {
            "subject": safe("Subject")[:500],
            "body_text": body,
            "body_html": html,
            "from_addr": safe("From"),
            "raw_headers": "",
            "date": date,
            "label": label,
            "source_corpus": source,
        }
    except Exception:
        return None


def parse_file(path: Path, label: int, source: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
        return msg_to_record(msg, label, source)
    except Exception:
        return None


def load_spamassassin() -> list[dict]:
    records = []
    base = RAW / "spamassassin"
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        label = 1 if "spam" in folder.name.lower() else 0
        src = f"spamassassin_{'spam' if label else 'ham'}"
        for f in tqdm(list(folder.iterdir()), desc=folder.name):
            if f.is_file():
                r = parse_file(f, label, src)
                if r:
                    records.append(r)
    return records


def load_nazario() -> list[dict]:
    records = []
    base = RAW / "nazario"
    for f in base.iterdir():
        if not f.is_file():
            continue
        try:
            raw = f.read_bytes()
        except Exception as e:
            print(f"  cannot read {f.name}: {e}")
            continue
        chunks = re.split(rb"\r?\n(?=From \S+@\S+)", raw)
        good = 0
        for chunk in tqdm(chunks, desc=f.name):
            if len(chunk) < 50:
                continue
            try:
                msg = email.message_from_bytes(chunk, policy=email.policy.default)
                r = msg_to_record(msg, label=1, source="nazario")
                if r:
                    records.append(r)
                    good += 1
            except Exception:
                continue
        print(f"  {f.name}: {good} messages parsed")
    return records


def load_phishing_pot() -> list[dict]:
    base = RAW / "phishing_pot"
    files = list(base.rglob("*.eml"))
    records = []
    for f in tqdm(files, desc="phishing_pot"):
        r = parse_file(f, label=1, source="phishing_pot")
        if r:
            records.append(r)
    return records


def load_enron(cap: int = 15000) -> list[dict]:
    tar_path = next((RAW / "enron").glob("enron_mail_*.tar.gz"), None)
    if tar_path is None:
        print("  ENRON tar.gz NOT FOUND -- re-download it")
        return []

    print("  enron: scanning archive, takes a couple of minutes...")
    wanted = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if m.isfile():
                n = m.name.lower()
                if "/inbox/" in n or "/sent" in n:
                    wanted.append(m.name)
    print(f"  enron: {len(wanted):,} inbox/sent messages in archive")

    random.seed(42)
    keep = set(random.sample(wanted, min(cap, len(wanted))))
    records = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tqdm(tf, desc="enron"):
            if m.name not in keep:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
                r = msg_to_record(msg, 0, "enron")
                if r:
                    records.append(r)
            except Exception:
                continue
    return records


def build_dataset() -> pd.DataFrame:
    records = []
    records += load_spamassassin()
    records += load_nazario()
    records += load_phishing_pot()
    records += load_enron()
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["body_text"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    out = RAW.parent / "emails.parquet"
    df.to_parquet(out)
    print(f"\nSaved {len(df):,} emails -> {out}\n")
    print(df.groupby(["source_corpus", "label"]).size())