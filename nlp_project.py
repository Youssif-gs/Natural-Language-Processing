import os
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


DATASET_PATH = "data/bbc"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))


def load_dataset(dataset_path):
    texts = []
    labels = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)

        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)

                try:
                    with open(file_path, "r", encoding="latin-1") as file:
                        text = file.read()

                    texts.append(text)
                    labels.append(category)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def plot_label_distribution(df):
    counts = df["label"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Dataset Label Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Number of Documents")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.show()


def extract_keywords(vectorizer, model, class_names, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())

    keywords = {}

    for class_index, class_name in enumerate(class_names):
        coefficients = model.coef_[class_index]
        top_indices = coefficients.argsort()[-top_n:][::-1]
        top_words = feature_names[top_indices]
        keywords[class_name] = list(top_words)

    return keywords


def generate_wordclouds(df):
    for label in df["label"].unique():
        text = " ".join(df[df["label"] == label]["clean_text"])

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {label}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/wordcloud_{label}.png")
        plt.show()


def explain_prediction(text, pipeline_predict_function, class_names):
    explainer = LimeTextExplainer(class_names=class_names)

    explanation = explainer.explain_instance(
        text,
        pipeline_predict_function,
        num_features=10
    )

    explanation.save_to_file(f"{OUTPUT_DIR}/lime_explanation.html")

    print("\nTop words contributing to the prediction:")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("\nDataset preview:")
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    plot_label_distribution(df)

    print("\nCleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    X = df["clean_text"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

    model.fit(X_train_tfidf, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    class_names = label_encoder.classes_

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    print("\nClassification Report:")
    print(report)

    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    plot_confusion_matrix(y_test, y_pred, class_names)

    print("\nExtracting keywords per topic...")
    keywords = extract_keywords(vectorizer, model, class_names)

    for topic, words in keywords.items():
        print(f"{topic}: {', '.join(words)}")

    with open(f"{OUTPUT_DIR}/topic_keywords.txt", "w") as file:
        for topic, words in keywords.items():
            file.write(f"{topic}: {', '.join(words)}\n")

    print("\nGenerating word clouds...")
    generate_wordclouds(df)

    print("\nSaving model files...")
    joblib.dump(model, f"{MODEL_DIR}/topic_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")

    def predict_proba_for_lime(texts):
        cleaned_texts = [clean_text(text) for text in texts]
        transformed_texts = vectorizer.transform(cleaned_texts)
        return model.predict_proba(transformed_texts)

    sample_text = X_test.iloc[0]

    print("\nExplaining one prediction using LIME...")
    explain_prediction(sample_text, predict_proba_for_lime, class_names)

    print("\nTesting custom document prediction...")

    custom_document = """
    Artificial intelligence and machine learning are changing the software industry.
    Companies are using data, algorithms, and automation to improve technology products.
    """

    cleaned_document = clean_text(custom_document)
    document_vector = vectorizer.transform([cleaned_document])
    prediction = model.predict(document_vector)[0]
    predicted_topic = label_encoder.inverse_transform([prediction])[0]

    print("\nCustom Document:")
    print(custom_document)
    print(f"Predicted Topic: {predicted_topic}")

    print("\nProject finished successfully.")
    print(f"Check the '{OUTPUT_DIR}' folder for results.")


if __name__ == "__main__":
    main()