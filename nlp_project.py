import re
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from lime.lime_text import LimeTextExplainer

nltk.download("stopwords")

from nltk.corpus import stopwords

# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "BBC News Train.csv"

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

# ============================================================
# CREATE FOLDERS
# ============================================================

import os

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# STOPWORDS
# ============================================================

stop_words = set(stopwords.words("english"))

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(TRAIN_FILE)

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Category"].value_counts())

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("\nCleaning text...")

df["clean_text"] = df["Text"].apply(clean_text)

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["Category"])

class_names = label_encoder.classes_

print("\nClasses:")
print(class_names)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = df["clean_text"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
)

model.fit(X_train_tfidf, y_train)

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:")
print(report)

# ============================================================
# SAVE REPORT
# ============================================================

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

plt.show()

# ============================================================
# LABEL DISTRIBUTION
# ============================================================

df["Category"].value_counts().plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Dataset Distribution")
plt.xlabel("Topic")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/dataset_distribution.png")

plt.show()

# ============================================================
# KEYWORD EXTRACTION
# ============================================================

print("\nExtracting keywords per topic...")

feature_names = np.array(vectorizer.get_feature_names_out())

topic_keywords = {}

for index, topic in enumerate(class_names):

    coefficients = model.coef_[index]

    top_indices = coefficients.argsort()[-10:][::-1]

    keywords = feature_names[top_indices]

    topic_keywords[topic] = keywords.tolist()

    print(f"\n{topic}:")
    print(keywords)

# ============================================================
# SAVE KEYWORDS
# ============================================================

with open(f"{OUTPUT_DIR}/topic_keywords.txt", "w") as file:

    for topic, keywords in topic_keywords.items():
        file.write(f"{topic}: {', '.join(keywords)}\n")

# ============================================================
# WORD CLOUDS
# ============================================================
print("\nGenerating word clouds...")

for topic in class_names:

    text = " ".join(
        df[df["Category"] == topic]["clean_text"]
    )

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(f"Word Cloud - {topic}")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/wordcloud_{topic}.png")

    plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(model, f"{MODEL_DIR}/topic_model.pkl")

joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")

# ============================================================
# LIME EXPLAINABILITY
# ============================================================

print("\nGenerating explainability output...")

def predict_proba(texts):

    cleaned_texts = [clean_text(text) for text in texts]

    transformed = vectorizer.transform(cleaned_texts)

    return model.predict_proba(transformed)

explainer = LimeTextExplainer(
    class_names=class_names
)

sample_text = X_test.iloc[0]

explanation = explainer.explain_instance(
    sample_text,
    predict_proba,
    num_features=10
)

explanation.save_to_file(
    f"{OUTPUT_DIR}/lime_explanation.html"
)

print("\nLIME explanation saved.")

# ============================================================
# CUSTOM PREDICTION
# ============================================================

print("\nTesting custom prediction...")

custom_text = """
Artificial intelligence and software companies are using machine learning
to improve modern technology and data analysis systems.
"""

cleaned = clean_text(custom_text)

vector = vectorizer.transform([cleaned])

prediction = model.predict(vector)[0]

predicted_topic = label_encoder.inverse_transform([prediction])[0]

print("\nCustom Text:")
print(custom_text)

print("\nPredicted Topic:")
print(predicted_topic)

# ============================================================
# FINISHED
# ============================================================

print("\nProject completed successfully.")
print("Check outputs folder.")