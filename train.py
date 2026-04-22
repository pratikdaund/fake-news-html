"""
train.py - Fake News Detection Model Training
==============================================
Trains two models on the ISOT Fake News dataset:
  1. TF-IDF + Logistic Regression  (~94% accuracy, fast, small)
  2. Bidirectional LSTM            (~96% accuracy, deep learning)

USAGE:
  1. Download dataset from:
     https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  2. Place Fake.csv and True.csv inside the data/ folder.
  3. Run:   python train.py
  4. Saved artefacts go into models/

Run `python train.py --skip-lstm` if you only want the logistic regression
(much faster, smaller file, perfectly fine for the Render free tier).
"""

import os
import re
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "data"
MODELS_DIR = "models"
RANDOM_STATE = 42
MAX_WORDS = 10000
MAX_LEN = 300
EMBED_DIM = 64
EPOCHS = 4
BATCH_SIZE = 64

os.makedirs(MODELS_DIR, exist_ok=True)

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


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


def load_data():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            f"\nPlace Fake.csv and True.csv inside {DATA_DIR}/\n"
            "Download: https://www.kaggle.com/datasets/clmentbisaillon/"
            "fake-and-real-news-dataset\n"
        )

    print("[*] Loading data...")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # IMPORTANT: remove Reuters prefix so the model doesn't cheat.
    # This is a well-known artifact of the ISOT dataset. Mention it in
    # your thesis limitations section.
    true["text"] = true["text"].astype(str).str.replace(
        r"^.*?\(Reuters\)\s*-\s*", "", regex=True
    )

    fake["label"] = 0   # FAKE
    true["label"] = 1   # REAL

    df = pd.concat([fake, true], ignore_index=True)
    df = df[["title", "text", "label"]].dropna()
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"[+] Total samples: {len(df)}")
    print(f"[+] Fake: {(df.label==0).sum()}  Real: {(df.label==1).sum()}")
    return df


def train_logreg(X_train, X_test, y_train, y_test):
    print("\n[*] Training TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    print(f"\n===== Logistic Regression =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    print(f"[OK] Saved logreg_model.pkl ({os.path.getsize(os.path.join(MODELS_DIR, 'logreg_model.pkl'))/1024:.1f} KB)")
    print(f"[OK] Saved tfidf_vectorizer.pkl")


def train_lstm(X_train, X_test, y_train, y_test):
    print("\n[*] Training Bidirectional LSTM...")
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
        )
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("[!] TensorFlow not installed. Skipping LSTM.")
        print("    Install with: pip install tensorflow")
        return

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train),
                        maxlen=MAX_LEN, padding="post", truncating="post")
    Xte = pad_sequences(tokenizer.texts_to_sequences(X_test),
                        maxlen=MAX_LEN, padding="post", truncating="post")

    vocab = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = Sequential([
        Embedding(vocab, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(48, dropout=0.3, recurrent_dropout=0.0)),
        Dense(24, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    cb = [EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss")]
    model.fit(Xtr, y_train, validation_split=0.1,
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)

    proba = model.predict(Xte, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"\n===== LSTM =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    with open(os.path.join(MODELS_DIR, "lstm_tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    size_mb = os.path.getsize(os.path.join(MODELS_DIR, "lstm_model.keras"))/1024/1024
    print(f"[OK] Saved lstm_model.keras ({size_mb:.1f} MB)")
    print(f"[OK] Saved lstm_tokenizer.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Train only Logistic Regression (faster, smaller)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only N samples for quick testing")
    args = parser.parse_args()

    df = load_data()
    if args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(df)} rows")

    print("[*] Cleaning text (this may take a minute)...")
    df["cleaned"] = df["content"].apply(clean_text)
    df = df[df["cleaned"].str.len() > 0].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"].values, df["label"].values,
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"[+] Train: {len(X_train)}  Test: {len(X_test)}")

    train_logreg(X_train, X_test, y_train, y_test)
    if not args.skip_lstm:
        train_lstm(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("DONE. Artefacts in:", os.path.abspath(MODELS_DIR))
    print("Next step:  python app.py")
    print("="*50)


if __name__ == "__main__":
    main()
