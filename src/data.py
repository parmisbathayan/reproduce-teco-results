import os
import re
import pickle

import numpy as np
import pandas as pd

# (pickle name, code used in the result tables), in the usual participant order
PARTICIPANTS = [
    ("Amin", "AMB"), ("Arya", "ARM"), ("Eisa", "EIA"), ("Elahe", "ELD"),
    ("Farzam", "FAS"), ("Javad", "JAY"), ("Mehrnaz", "MEB"), ("Nastaran", "NAD"),
    ("Razieh", "RAS"), ("Sajad", "SAS"), ("Shahrzad", "SHU"), ("Yaser", "AYB"),
]

N_SENTENCES = 165
N_CHANNELS = 126


def _norm(text):
    # the label CSV uses a different sentence id than the pickles, so we match on
    # the sentence text instead; collapse whitespace and drop the zero-width
    # non-joiner so the two sources line up exactly
    return re.sub(r"\s+", "", str(text)).replace("‌", "")


def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    df["key"] = df["sentence"].map(_norm)
    return df[["key", "sentiment_label", "control"]]


def _read_subject(trt_dir, name):
    with open(os.path.join(trt_dir, f"{name}_trt_total.pickle"), "rb") as f:
        return pickle.load(f)


def load_eeg(trt_dir):
    """EEG as (subjects, sentences, words, channels) plus the per-sentence text."""
    subjects = [_read_subject(trt_dir, name) for name, _ in PARTICIPANTS]
    max_words = max(max(len(s[t]["word"]) for t in s) for s in subjects)

    eeg = np.zeros((len(subjects), N_SENTENCES, max_words, N_CHANNELS), dtype=np.float32)
    for k, sub in enumerate(subjects):
        for t in sub:
            sid = sub[t]["sentenceId"]                    # 1-based sentence position
            for wi, word in sub[t]["word"].items():
                vec = np.asarray(word["TRT_total"])
                # un-fixated words are stored as a length-2 stub; leave those at zero
                if vec.shape == (N_CHANNELS,):
                    eeg[k, sid - 1, wi] = vec

    # same stimuli for everyone, so the first subject is enough to get the text
    first = subjects[0]
    text = [None] * N_SENTENCES
    for t in first:
        text[first[t]["sentenceId"] - 1] = _norm(first[t]["persian_sentence"])
    return eeg, text


def build_binary_dataset(trt_dir, labels_csv, return_sentence_ids=False):
    """Positive vs negative sentences (neutral dropped -> 110 sentences)."""
    labels = load_labels(labels_csv)
    label_of = dict(zip(labels["key"], labels["sentiment_label"]))

    eeg, text = load_eeg(trt_dir)
    missing = [i for i, k in enumerate(text) if k not in label_of]
    if missing:
        raise ValueError(f"{len(missing)} sentences have no label in {labels_csv}")

    sentiment = np.array([label_of[k] for k in text])     # in sentence order, {-1,0,1}
    keep = np.where(sentiment != 0)[0]

    X = eeg[:, keep]
    y = (sentiment[keep] == 1).astype(int)                # negative -> 0, positive -> 1
    codes = [c for _, c in PARTICIPANTS]
    if return_sentence_ids:
        return X, y, codes, keep + 1                     # TeCo sentenceId is 1-based
    return X, y, codes
