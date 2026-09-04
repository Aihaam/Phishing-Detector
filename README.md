# Phishing-Detector

An AI-based system for detecting phishing email, and an evaluation of whether the
high accuracy such systems report survives honest, corpus-controlled testing.
Two detectors are built (a TF-IDF + logistic-regression baseline and a fine-tuned
DistilBERT model) and evaluated under a random split, a leave-one-corpus-out split,
a temporal split, and a provenance probe. A three-tier triage application
(Allow / Warn / Quarantine) is built on the baseline.

This repository is the digital artefact for the MSc dissertation
*"Development of an AI-based System for Real-time Detection and Prevention of
Email Phishing Attacks"* (MMU, 7V0007).

## Repository layout

```
src/                 source code
  data.py            assembles the four corpora into data/emails.parquet
  baseline.py        TF-IDF + logistic-regression, five evaluation settings
  transformer.py     fine-tuned DistilBERT, same settings
  recall.py          threshold sweep / operating-point analysis
  latency.py         throughput and latency measurement
  train_app_model.py trains and saves the model used by the app
  app.py             Streamlit triage application
notebooks/
  01_data_exploration.ipynb   data-quality analysis (Chapter 3 evidence)
results/             experiment outputs (JSON metrics + run logs)
models/              the trained application model (app_model.joblib)
dissertation/        appendices (ethics approval, consent form, PIS, questionnaire)
requirements.txt     pinned dependencies
```

## Data (not included)

The email corpora are **not** stored in this repository. They contain real
personal data, so they are excluded for data-protection (UK GDPR) reasons, and
they are large. Reproducibility is preserved through `src/data.py` plus the
download locations below.

Download each corpus and place it under `data/raw/` as follows, then build the
dataset:

| Corpus | Place under | Source |
| --- | --- | --- |
| Enron (legitimate) | `data/raw/enron/enron_mail_*.tar.gz` | https://www.cs.cmu.edu/~enron/ |
| SpamAssassin (ham + spam) | `data/raw/spamassassin/` (the extracted category folders) | https://spamassassin.apache.org/old/publiccorpus/ |
| Nazario (phishing) | `data/raw/nazario/` (the mbox files) | https://monkey.org/~jose/phishing/ |
| Phishing Pot (phishing) | `data/raw/phishing_pot/` (the `.eml` files) | https://github.com/rf-peixoto/phishing_pot |

```bash
python src/data.py          # writes data/emails.parquet (34,930 emails)
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the experiments

All seeds are fixed (`random_state=42`), so counts and scores reproduce on re-run.

```bash
python src/baseline.py      # -> results/baseline_*.json
python src/transformer.py   # -> results/distilbert_*.json   (GPU recommended)
python src/recall.py        # -> results/recall_log.txt
python src/latency.py       # throughput / latency figures
```

The data-quality figures in Chapter 3 are produced by
`notebooks/01_data_exploration.ipynb` (Restart & Run All).

## The application

```bash
python src/train_app_model.py     # writes models/app_model.joblib
streamlit run src/app.py
```

Paste an email or upload an `.eml` file. The app reads the subject and body only
(headers are ignored), returns a probability, buckets it as Allow (< 0.3),
Warn (0.3–0.7) or Quarantine (>= 0.7), and lists the terms that drove the score.
The trained `models/app_model.joblib` is included, so the app runs without
rebuilding the dataset.

## Notes

- The datasets are cited in the dissertation; see the References section.
