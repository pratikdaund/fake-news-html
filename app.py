"""
app.py - Fake News Detection (Flask + HTML/CSS/JS)
===================================================
Run locally:   python app.py
Deploy:        gunicorn app:app  (configured in render.yaml)

Routes:
    GET  /                  -> serves the HTML UI
    GET  /health            -> health check
    POST /api/predict       -> JSON prediction endpoint
    GET  /static/<path>     -> CSS / JS
"""
import os
import re
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fnd")

# ---------- Config ----------
MODELS_DIR = "models"
MAX_LEN = 300   # Must match train.py

app = Flask(__name__, static_folder="static", template_folder="templates")


# ---------- NLTK setup ----------
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


setup_nltk()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------- Preprocessing (IDENTICAL to train.py) ----------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


# ---------- Model loaders (loaded once at startup) ----------
LOGREG_MODEL = None
TFIDF_VECTORIZER = None
LSTM_MODEL = None
LSTM_TOKENIZER = None


def load_logreg():
    global LOGREG_MODEL, TFIDF_VECTORIZER
    try:
        with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
            LOGREG_MODEL = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            TFIDF_VECTORIZER = pickle.load(f)
        log.info("Loaded LogReg + TF-IDF.")
    except FileNotFoundError as e:
        log.warning(f"LogReg not loaded: {e}")


def load_lstm():
    global LSTM_MODEL, LSTM_TOKENIZER
    try:
        from tensorflow.keras.models import load_model
        path = os.path.join(MODELS_DIR, "lstm_model.keras")
        if not os.path.exists(path):
            log.warning("LSTM model file missing.")
            return
        LSTM_MODEL = load_model(path)
        with open(os.path.join(MODELS_DIR, "lstm_tokenizer.pkl"), "rb") as f:
            LSTM_TOKENIZER = pickle.load(f)
        log.info("Loaded LSTM + tokenizer.")
    except Exception as e:
        log.warning(f"LSTM not loaded: {e}")


load_logreg()
load_lstm()


# ---------- Prediction functions ----------
def predict_logreg(text):
    if LOGREG_MODEL is None or TFIDF_VECTORIZER is None:
        return None
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0,
                "model": "TF-IDF + Logistic Regression",
                "tokens": 0, "real_prob": 0.5, "cleaned": cleaned}
    x = TFIDF_VECTORIZER.transform([cleaned])
    real_prob = float(LOGREG_MODEL.predict_proba(x)[0][1])
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    conf = real_prob if label == "REAL" else 1 - real_prob
    return {
        "label": label, "confidence": round(conf, 4),
        "model": "TF-IDF + Logistic Regression",
        "tokens": len(cleaned.split()),
        "real_prob": round(real_prob, 4),
        "cleaned": cleaned,
    }


def predict_lstm(text):
    if LSTM_MODEL is None or LSTM_TOKENIZER is None:
        return None
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0,
                "model": "Bidirectional LSTM",
                "tokens": 0, "real_prob": 0.5, "cleaned": cleaned}
    seq = LSTM_TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    real_prob = float(LSTM_MODEL.predict(padded, verbose=0).ravel()[0])
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    conf = real_prob if label == "REAL" else 1 - real_prob
    return {
        "label": label, "confidence": round(conf, 4),
        "model": "Bidirectional LSTM",
        "tokens": len(cleaned.split()),
        "real_prob": round(real_prob, 4),
        "cleaned": cleaned,
    }


def available_models():
    avail = []
    if LOGREG_MODEL is not None:
        avail.append("logreg")
    if LSTM_MODEL is not None:
        avail.append("lstm")
    return avail


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html",
                           models=available_models())


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_available": available_models(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    model = (data.get("model") or "logreg").lower()

    if len(text) < 20:
        return jsonify({"error": "Text must be at least 20 characters."}), 400

    avail = available_models()
    if model not in avail:
        return jsonify({
            "error": f"Model '{model}' not available. "
                     f"Available: {avail or 'none — run train.py first'}"
        }), 503

    try:
        if model == "logreg":
            result = predict_logreg(text)
        else:
            result = predict_lstm(text)

        if result is None:
            return jsonify({"error": "Prediction failed - model not loaded."}), 500
        return jsonify(result)
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


# ---------- Entry point ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
