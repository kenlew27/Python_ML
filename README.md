# Assignment 3 – Headline Sentiment Scorer (Streamlit GUI)

## File Locations

| File | Path |
| Streamlit GUI (Assignment 3) | `/home/kenlew/ml_eng_class/assignment3/score_headlines_gui.py` |
| FastAPI web service (Assignment 2) | `/home/kenlew/Python_ML/score_headlines_api.py` |
| SVM model | `/home/kenlew/Python_ML/svm.joblib` |

## Overview

This Streamlit application provides a web interface for the Headline Scoring
API built in Assignment 2. Users can enter, edit, and delete news headlines
through an interactive GUI, submit them to the FastAPI scoring service, and
view sentiment results (Optimistic, Neutral, or Pessimistic).

## How to Run

Two servers must be running simultaneously, each on its own port.

### 1. Start the FastAPI Scoring Service (port 8084)

```bash
cd /home/kenlew/Python_ML
python score_headlines_api.py
```

### 2. Start the Streamlit GUI (port 9084)

```bash
cd /home/kenlew/ml_eng_class/assignment3
streamlit run score_headlines_gui.py --server.port 9084
```

## Features

- Three input modes: paste multiple headlines, add one-by-one, or upload a text file
- Inline editing of individual headlines before scoring
- Delete individual headlines from the queue
- Summary statistics showing counts per sentiment category
- Results displayed in a sortable table
- Download results as CSV or TXT
- Configurable API URL in the sidebar

## Dependencies

```bash
pip install streamlit requests pandas
```

The FastAPI server (Assignment 2) requires:

```bash
pip install fastapi uvicorn joblib sentence-transformers
```
