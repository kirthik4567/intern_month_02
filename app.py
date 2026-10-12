import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Spam Checker", page_icon="📧", layout="centered")

DATA_PATH = Path(__file__).parent / "spam.csv"
stemmer = PorterStemmer()

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 760px; padding-top: 3.5rem; padding-bottom: 4rem; }

    .page-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
    .page-sub   { opacity: .65; margin: .35rem 0 1.8rem 0; font-size: 1rem; }

    .stTextArea textarea { font-size: 1rem; line-height: 1.55; border-radius: 12px; }
    .stButton > button {
        border-radius: 10px; padding: .6rem 1.6rem; font-weight: 600; border: none;
    }

    .verdict {
        border-radius: 14px; padding: 1.3rem 1.5rem; margin: 1.4rem 0 .6rem 0;
        border: 1px solid; display: flex; justify-content: space-between; align-items: center;
    }
    .verdict.spam { background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.45); }
    .verdict.ham  { background: rgba(34,197,94,.10);  border-color: rgba(34,197,94,.45); }
    .verdict .label { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; }
    .verdict.spam .label { color: #f87171; }
    .verdict.ham  .label { color: #4ade80; }
    .verdict .hint { opacity: .7; font-size: .92rem; margin-top: .15rem; }
    .verdict .pct  { font-size: 2.4rem; font-weight: 700; font-variant-numeric: tabular-nums; }

    .bar { height: 8px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden; margin: .4rem 0 1.4rem 0; }
    .bar > div { height: 100%; border-radius: 99px; }

    .row { display: flex; align-items: center; gap: 1rem; padding: .55rem 0;
           border-bottom: 1px solid rgba(128,128,128,.2); font-size: .95rem; }
    .row .name { flex: 0 0 11rem; font-weight: 500; }
    .row .track { flex: 1; height: 6px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden; }
    .row .track > div { height: 100%; border-radius: 99px; }
    .row .val { flex: 0 0 4.5rem; text-align: right; font-variant-numeric: tabular-nums; opacity: .85; }
    .row .tag { flex: 0 0 3.2rem; text-align: right; font-weight: 600; font-size: .85rem; }

    .section-title { font-size: 1.05rem; font-weight: 600; margin: 1.6rem 0 .4rem 0; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [stemmer.stem(w) for w in text.split()
             if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(words)


@st.cache_resource(show_spinner="Training models. This only happens the first time.")
def train():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]].dropna().drop_duplicates()
    df.columns = ["label", "text"]
    df["y"] = (df["label"].str.lower().str.strip() == "spam").astype(int)
    df["clean"] = df["text"].apply(clean)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
    }
    rows = []
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        p = m.predict(Xte)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_te, p),
                     "Precision": precision_score(y_te, p),
                     "Recall": recall_score(y_te, p),
                     "F1": f1_score(y_te, p)})
    metrics = pd.DataFrame(rows).set_index("Model").round(4)

    lr = models["Logistic Regression"]
    feats = np.array(vec.get_feature_names_out())
    top = np.argsort(lr.coef_[0])[-15:]
    keywords = pd.Series(lr.coef_[0][top], index=feats[top], name="Spam weight")
    return vec, models, metrics, keywords, len(df)


vec, models, metrics, keywords, n_rows = train()

# ---------------------------------------------------------------- header
st.markdown('<p class="page-title">Spam checker</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="page-sub">Paste an email or SMS to see whether it looks like spam. '
    f'Trained on {n_rows:,} labelled messages.</p>',
    unsafe_allow_html=True,
)

tab_check, tab_perf = st.tabs(["Check a message", "Model performance"])

# ---------------------------------------------------------------- check tab
with tab_check:
    msg = st.text_area(
        "Message",
        height=170,
        placeholder="Paste the message you want to check…",
        label_visibility="collapsed",
    )
    col_btn, col_model = st.columns([1, 2], vertical_alignment="center")
    run = col_btn.button("Check message", type="primary")
    model_name = col_model.selectbox(
        "Model", list(models.keys()), index=1, label_visibility="collapsed")

    if run:
        if not msg.strip():
            st.warning("Enter a message to check.")
        else:
            X = vec.transform([clean(msg)])
            probs = {n: float(m.predict_proba(X)[0][1]) for n, m in models.items()}
            p = probs[model_name]
            is_spam = p >= 0.5
            kind = "spam" if is_spam else "ham"
            conf = p if is_spam else 1 - p
            color = "#ef4444" if is_spam else "#22c55e"
            hint = ("This message looks like spam." if is_spam
                    else "This message looks legitimate.")

            st.markdown(
                f"""
                <div class="verdict {kind}">
                  <div>
                    <div class="label">{'Spam' if is_spam else 'Not spam'}</div>
                    <div class="hint">{hint} Based on {model_name}.</div>
                  </div>
                  <div class="pct">{conf:.0%}</div>
                </div>
                <div class="bar"><div style="width:{p*100:.1f}%; background:{color};"></div></div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">What each model says</div>',
                        unsafe_allow_html=True)
            rows_html = ""
            for n, sp in probs.items():
                c = "#ef4444" if sp >= 0.5 else "#22c55e"
                t = "Spam" if sp >= 0.5 else "Ham"
                rows_html += (
                    f'<div class="row"><div class="name">{n}</div>'
                    f'<div class="track"><div style="width:{sp*100:.1f}%; background:{c};"></div></div>'
                    f'<div class="val">{sp:.1%}</div>'
                    f'<div class="tag" style="color:{c};">{t}</div></div>'
                )
            st.markdown(rows_html, unsafe_allow_html=True)
            st.caption("Percentages show the probability that the message is spam.")

# ---------------------------------------------------------------- performance tab
with tab_perf:
    st.markdown('<div class="section-title">Accuracy on held-out test data</div>',
                unsafe_allow_html=True)
    st.dataframe(metrics.style.format("{:.1%}"), use_container_width=True)

    st.markdown('<div class="section-title">Words that most strongly signal spam</div>',
                unsafe_allow_html=True)
    st.bar_chart(keywords, horizontal=True)