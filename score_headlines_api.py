# score_headlines_api.py
import logging
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score_headlines_api")

app = FastAPI()

MODEL = None
EMBEDDER = None


class ScoreReq(BaseModel):
    headlines: List[str]


def load_once():
    global MODEL, EMBEDDER
    if MODEL is None:
        p = Path("svm.joblib")
        if not p.exists():
            p = Path(__file__).resolve().parent / "svm.joblib"
        if not p.exists():
            logger.error("svm.joblib not found")
            raise FileNotFoundError("svm.joblib not found")
        MODEL = joblib.load(p)
        logger.info("Loaded svm.joblib from %s", p)

    if EMBEDDER is None:
        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedder all-MiniLM-L6-v2")


@app.on_event("startup")
def startup():
    load_once()


@app.get("/status")
def status():
    return {"status": "OK"}


@app.post("/score_headlines")
def score_headlines(req: ScoreReq):
    if not req.headlines:
        logger.warning("empty headlines list")
        raise HTTPException(status_code=400, detail="headlines must be non-empty")
    try:
        load_once()
        X = EMBEDDER.encode(req.headlines)
        labels = MODEL.predict(X)
        return {"labels": [str(x) for x in labels]}
    except Exception:
        logger.error("scoring failed", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("score_headlines_api:app", host="0.0.0.0", port=8084)
