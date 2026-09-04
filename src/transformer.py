"""
transformer.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parent.parent

MODEL = "distilbert-base-uncased"
MAXLEN = 256
BATCH = 16
EPOCHS = 2
LR = 2e-5

SMOKE_TEST = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_dataset():
    df = pd.read_parquet(ROOT / "data" / "emails.parquet")

    spam_pattern = r"(\*\*\*SPAM\*\*\*|\[SPAM\]|^SPAM:)\s*"
    leak_pattern = r"(jose@monkey\.org|monkey\.org|phishing@pot|phishing\.pot)"

    df["subject"] = df["subject"].fillna("")
    df["body_text"] = df["body_text"].fillna("")

    df["subject"] = df["subject"].str.replace(
        spam_pattern,
        "",
        regex=True,
    )

    df["subject"] = df["subject"].str.replace(
        leak_pattern,
        " ",
        regex=True,
    )

    df["body_text"] = df["body_text"].str.replace(
        leak_pattern,
        " ",
        regex=True,
    )

    df["text"] = (
        df["subject"] + " " + df["body_text"]
    ).str.slice(0, 20000)

    if SMOKE_TEST:
        df = (
            df.groupby("source_corpus", group_keys=False)
            .sample(n=400, random_state=42)
            .reset_index(drop=True)
        )

        print(">>> SMOKE TEST: reduced sample, 1 epoch <<<")

    return df


df = load_dataset()

if SMOKE_TEST:
    EPOCHS = 1

print(f"Loaded {len(df):,} emails\n")


tokenizer = AutoTokenizer.from_pretrained(MODEL)


class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i],
            truncation=True,
            max_length=MAXLEN,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[i],
                dtype=torch.long,
            ),
        }


def train_eval(X_tr, y_tr, X_te, y_te, label=""):
    print(f"\n=== {label} ===")
    print(
        f"  training on {len(X_tr):,}, "
        f"testing on {len(X_te):,}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2,
    ).to(device)

    tr_loader = DataLoader(
        EmailDataset(X_tr, y_tr),
        batch_size=BATCH,
        shuffle=True,
    )

    te_loader = DataLoader(
        EmailDataset(X_te, y_te),
        batch_size=BATCH * 2,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    steps = len(tr_loader) * EPOCHS

    sched = get_linear_schedule_with_warmup(
        opt,
        int(0.1 * steps),
        steps,
    )

    model.train()

    for ep in range(EPOCHS):
        total_loss = 0.0

        for i, batch in enumerate(tr_loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()
            sched.step()
            opt.zero_grad()

            total_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"    epoch {ep + 1} "
                    f"step {i}/{len(tr_loader)} "
                    f"loss {loss.item():.4f}"
                )

        print(
            f"    epoch {ep + 1} mean loss "
            f"{total_loss / len(tr_loader):.4f}"
        )

    model.eval()

    probs = []
    trues = []

    with torch.no_grad():
        for batch in te_loader:
            labels = batch.pop("labels")

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**inputs)
            spam_prob = torch.softmax(
                outputs.logits,
                dim=1,
            )[:, 1]

            probs.extend(spam_prob.cpu().numpy())
            trues.extend(labels.numpy())

    probs = np.asarray(probs)
    trues = np.asarray(trues)

    pred = (probs >= 0.5).astype(int)

    result = {
        "experiment": label,
        "model": "distilbert",
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "precision": round(
            precision_score(
                trues,
                pred,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                trues,
                pred,
                zero_division=0,
            ),
            4,
        ),
        "f1": round(
            f1_score(
                trues,
                pred,
                zero_division=0,
            ),
            4,
        ),
        "pr_auc": round(
            average_precision_score(
                trues,
                probs,
            ),
            4,
        ),
        "confusion": confusion_matrix(
            trues,
            pred,
        ).tolist(),
    }

    for metric in (
        "precision",
        "recall",
        "f1",
        "pr_auc",
    ):
        print(f"  {metric:10}: {result[metric]}")

    print(f"  confusion : {result['confusion']}")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


results = []


# 1. Random stratified

Xtr, Xte, ytr, yte = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42,
)

results.append(
    train_eval(
        Xtr,
        ytr,
        Xte,
        yte,
        "1_random_stratified",
    )
)


# 2. Leave-one-corpus-out

train_mask = df["source_corpus"].isin(
    ["nazario", "enron"]
)

test_mask = df["source_corpus"].isin(
    ["phishing_pot", "spamassassin_ham"]
)

results.append(
    train_eval(
        df.loc[train_mask, "text"],
        df.loc[train_mask, "label"],
        df.loc[test_mask, "text"],
        df.loc[test_mask, "label"],
        "2_leave_one_corpus_out",
    )
)


# 3. Old -> recent

old_phishing = df[
    (df["label"] == 1)
    & (df["source_corpus"] == "nazario")
]

new_phishing = df[
    (df["label"] == 1)
    & (df["source_corpus"] == "phishing_pot")
]

legit = df[df["label"] == 0]

legit_train, legit_test = train_test_split(
    legit,
    test_size=0.5,
    random_state=42,
)

train = pd.concat(
    [old_phishing, legit_train]
)

test = pd.concat(
    [new_phishing, legit_test]
)

results.append(
    train_eval(
        train["text"],
        train["label"],
        test["text"],
        test["label"],
        "3_temporal_old_to_new",
    )
)


# 4. Provenance probes

for cls, name, sources in [
    (1, "malicious", ["nazario", "phishing_pot"]),
    (0, "legitimate", ["enron", "spamassassin_ham"]),
]:
    sub = df[
        (df["label"] == cls)
        & (df["source_corpus"].isin(sources))
    ].copy()

    sub["src_label"] = (
        sub["source_corpus"] == sources[0]
    ).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        sub["text"],
        sub["src_label"],
        test_size=0.2,
        stratify=sub["src_label"],
        random_state=42,
    )

    results.append(
        train_eval(
            Xtr,
            ytr,
            Xte,
            yte,
            f"4_provenance_probe_{name}",
        )
    )


tag = "smoke" if SMOKE_TEST else "full"

out = (
    ROOT
    / "results"
    / f"distilbert_{tag}_{datetime.now():%Y%m%d_%H%M}.json"
)

out.write_text(
    json.dumps(results, indent=2)
)

print(f"\nSaved -> {out}")


print("\n" + "=" * 60)
print(f"SUMMARY (F1) -- DistilBERT [{tag}]")
print("=" * 60)

for result in results:
    print(
        f"  {result['experiment']:32} "
        f"{result['f1']}"
    )