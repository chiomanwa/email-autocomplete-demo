import json, re
import numpy as np
import streamlit as st
import tensorflow as tf

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(s: str):
    s = clean_text(s)
    return s.split() if s else []

@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("email_autocomplete_model.keras")
    with open("artifacts.json","r") as f:
        a = json.load(f)
    seq_len = int(a["sequence_length"])
    word_to_int = a["word_to_int"]
    int_to_word = {int(k): v for k, v in a["int_to_word"].items()}
    return model, seq_len, word_to_int, int_to_word

def encode(words, seq_len, word_to_int):
    ids = [word_to_int[w] for w in words if w in word_to_int]
    if len(ids) == 0:
        return None
    if len(ids) < seq_len:
        pad_id = ids[0]
        ids = [pad_id]*(seq_len-len(ids)) + ids
    else:
        ids = ids[-seq_len:]
    return np.array(ids).reshape(1, -1)

def sample(probs, temperature=0.8, top_k=20):
    p = np.asarray(probs).astype(float)
    if temperature != 1.0:
        p = np.log(p + 1e-12) / temperature
        p = np.exp(p)
    p = p / p.sum()

    if top_k and top_k > 0:
        idx = np.argpartition(p, -top_k)[-top_k:]
        pp = p[idx] / p[idx].sum()
        return int(np.random.choice(idx, p=pp))

    return int(np.random.choice(len(p), p=p))

def next_word(model, seed_text, seq_len, word_to_int, int_to_word, temperature, top_k):
    words = tokenize_words(seed_text)
    X = encode(words, seq_len, word_to_int)
    if X is None:
        return None, "No words from your input are in the model vocab. Try simpler/common words from your dataset."
    probs = model.predict(X, verbose=0)[0]
    wid = sample(probs, temperature, top_k)
    return int_to_word.get(wid, None), None

def complete(model, seed_text, n_words, seq_len, word_to_int, int_to_word, temperature, top_k):
    text = seed_text.strip()
    for _ in range(n_words):
        w, err = next_word(model, text, seq_len, word_to_int, int_to_word, temperature, top_k)
        if err or not w:
            break
        text = (text + " " + w).strip()
    return text

st.set_page_config(page_title="Email Autocomplete Demo", page_icon="✉️")
st.title("✉️ Email Autocomplete Demo (Left-to-Right)")

model, seq_len, word_to_int, int_to_word = load_all()

seed = st.text_area("Type an email draft:", height=160,
                    placeholder="Hi Professor, I hope you're doing well. I wanted to follow up about...")

c1, c2, c3 = st.columns(3)
temperature = c1.slider("Temperature", 0.2, 1.5, 0.8, 0.1)
top_k = c2.slider("Top-k", 0, 100, 20, 5)
n_words = c3.slider("Complete length (words)", 5, 120, 40, 5)

colA, colB = st.columns(2)

if colA.button("➡️ Suggest next word"):
    w, err = next_word(model, seed, seq_len, word_to_int, int_to_word, temperature, top_k)
    if err: st.warning(err)
    else: st.success(f"Next word: **{w}**")

if colB.button("🧠 Complete email"):
    if not seed.strip():
        st.warning("Type something first.")
    else:
        st.text_area("Completed email:", value=complete(model, seed, n_words, seq_len, word_to_int, int_to_word, temperature, top_k), height=240)