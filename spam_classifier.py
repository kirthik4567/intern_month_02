"""
Email / SMS Spam Classifier
Models: Naive Bayes, Logistic Regression, SVM
Run:  python spam_classifier.py
"""
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save plots to files (no blocking windows)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix, classification_report)

# ------------------------------------------------------------------
# 1. CONFIG  -> unga dataset-ku ethavadhu maaranum na inga maathunga
# ------------------------------------------------------------------
DATA_PATH = "spam.csv"     # unga CSV file name / path
LABEL_COL = None           # None = auto detect (example: "v1" or "Category")
TEXT_COL = None            # None = auto detect (example: "v2" or "Message")

# ------------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------------
try:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="ignore")

print("Columns in dataset:", list(df.columns))

def find_col(candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

if LABEL_COL is None:
    LABEL_COL = find_col(["v1", "label", "category", "class", "target", "spam", "type"])
if TEXT_COL is None:
    TEXT_COL = find_col(["v2", "text", "message", "email", "body", "content", "sms"])
if LABEL_COL is None or TEXT_COL is None:
    raise SystemExit("Column auto-detect fail aaiduchu. LABEL_COL & TEXT_COL ah manually set pannunga.")

df = df[[LABEL_COL, TEXT_COL]].dropna().drop_duplicates()
df.columns = ["label", "text"]

# label -> 0 (ham) / 1 (spam)
if pd.api.types.is_numeric_dtype(df["label"]):
    df["y"] = df["label"].astype(int)
else:
    df["y"] = df["label"].astype(str).str.lower().str.strip().map(
        lambda v: 1 if v in ("spam", "1", "junk") else 0)

print(f"\nTotal rows: {len(df)}")
print(df["y"].value_counts().rename({0: "ham", 1: "spam"}))

# ------------------------------------------------------------------
# 3. PREPROCESS TEXT
# ------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [stemmer.stem(w) for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(words)

df["clean"] = df["text"].apply(clean)

# ------------------------------------------------------------------
# 3b. EDA (notebook-la irundha analysis) -> png files-ah save aagum
# ------------------------------------------------------------------
df["length"] = df["text"].apply(len)

# Spam vs Ham pie chart
plt.figure(figsize=(5, 5))
df["y"].value_counts().rename({0: "Ham", 1: "Spam"}).plot(
    kind="pie", autopct="%1.1f%%", explode=[0, 0.1], shadow=True)
plt.ylabel(""); plt.title("Spam vs Ham")
plt.tight_layout(); plt.savefig("eda_spam_vs_ham.png", dpi=150); plt.close()

# Message length distribution
plt.figure(figsize=(8, 4))
sns.histplot(df[df["y"] == 0]["length"], bins=30, color="green", label="Ham", alpha=0.5)
sns.histplot(df[df["y"] == 1]["length"], bins=30, color="red", label="Spam", alpha=0.5)
plt.xlabel("Message length (characters)"); plt.legend(); plt.title("Message Length: Spam vs Ham")
plt.tight_layout(); plt.savefig("eda_message_length.png", dpi=150); plt.close()

# Word clouds (optional: pip install wordcloud)
try:
    from wordcloud import WordCloud
    for lbl, name in [(1, "spam"), (0, "ham")]:
        wc = WordCloud(width=800, height=450, background_color="white").generate(
            " ".join(df[df["y"] == lbl]["text"].str.lower()))
        plt.figure(figsize=(9, 5)); plt.imshow(wc); plt.axis("off")
        plt.title(f"{name.upper()} Word Cloud")
        plt.tight_layout(); plt.savefig(f"eda_wordcloud_{name}.png", dpi=150); plt.close()
except ImportError:
    print("wordcloud install illa -> word clouds skip pannuren (pip install wordcloud)")

print("Saved EDA plots: eda_*.png")

# ------------------------------------------------------------------
# 4. SPLIT + TF-IDF
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# ------------------------------------------------------------------
# 5. TRAIN 3 MODELS
# ------------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM (Linear)": LinearSVC(class_weight="balanced"),
}

results, fitted = [], {}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    fitted[name] = (model, pred)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "F0.5": fbeta_score(y_test, pred, beta=0.5, zero_division=0),
    })

res_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n=== MODEL COMPARISON ===")
print(res_df)

# F0.5 = precision-ku adhiga importance (ham-ah spam-nu thappa kandupidikka koodathu)
best_name = res_df["F0.5"].idxmax()
best_model, best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=["ham", "spam"]))

# ------------------------------------------------------------------
# 6. VISUALIZATIONS (png files-ah save aagum)
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# top spam-indicating keywords (Logistic Regression coefficients)
lr = fitted["Logistic Regression"][0]
features = np.array(vectorizer.get_feature_names_out())
top_idx = np.argsort(lr.coef_[0])[-20:]
plt.figure(figsize=(7, 6))
plt.barh(features[top_idx], lr.coef_[0][top_idx], color="crimson")
plt.title("Top 20 Spam-Indicating Keywords")
plt.tight_layout(); plt.savefig("top_spam_keywords.png", dpi=150); plt.close()
print("\nSaved: confusion_matrix.png, top_spam_keywords.png")

# ------------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------------
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved: spam_model.pkl, vectorizer.pkl")

# ------------------------------------------------------------------
# 8. REAL-TIME PREDICTION
# ------------------------------------------------------------------
def predict_email(text):
    vec = vectorizer.transform([clean(text)])
    return "SPAM" if best_model.predict(vec)[0] == 1 else "HAM (legit)"

print("\n=== Try your own message (blank enter = exit) ===")
while True:
    msg = input("Message: ").strip()
    if not msg:
        break
    print("->", predict_email(msg), "\n")
