import re
import os
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "BBC News Train.csv"

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# STOPWORDS
# ============================================================

try:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = set(ENGLISH_STOP_WORDS)


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
# TOKENIZATION
# ============================================================

MAX_WORDS = 12000
MAX_LENGTH = 250

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

NUM_CLASSES = len(class_names)

y_train_encoded = to_categorical(y_train, NUM_CLASSES)
y_test_encoded = to_categorical(y_test, NUM_CLASSES)


# ============================================================
# BUILD GRU MODEL
# ============================================================

print("\nBuilding GRU model...")

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        mask_zero=True
    )
)

model.add(
    GRU(
        128,
        return_sequences=False
    )
)

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(NUM_CLASSES, activation="softmax"))


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining GRU model...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_padded,
    y_train_encoded,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test_padded,
    y_test_encoded
)

print(f"\nTest Accuracy: {accuracy:.4f}")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred_probabilities = model.predict(X_test_padded)

y_pred = np.argmax(y_pred_probabilities, axis=1)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)

plt.title("GRU Confusion Matrix")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

plt.show()


# ============================================================
# DATASET DISTRIBUTION
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
# TRAINING ACCURACY CURVE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/training_accuracy.png")

plt.show()


# ============================================================
# TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/training_loss.png")

plt.show()


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
# KEYWORD EXTRACTION
# ============================================================

print("\nExtracting keywords...")

with open(f"{OUTPUT_DIR}/topic_keywords.txt", "w") as file:
    for topic in class_names:
        text = " ".join(
            df[df["Category"] == topic]["clean_text"]
        )

        words = text.split()

        frequency = {}

        for word in words:
            frequency[word] = frequency.get(word, 0) + 1

        sorted_words = sorted(
            frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_words = sorted_words[:10]

        print(f"\n{topic} keywords:")

        file.write(f"{topic}: ")

        keyword_list = []

        for word, freq in top_words:
            print(word)
            keyword_list.append(word)

        file.write(", ".join(keyword_list) + "\n")


# ============================================================
# SIMPLE EXPLAINABILITY
# ============================================================

def explain_prediction(text, top_n=10):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()

    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]

    word_scores = []

    for word, token_id in zip(words, sequence):
        if token_id != 0:
            word_scores.append((word, token_id))

    word_scores = sorted(
        word_scores,
        key=lambda x: x[1],
        reverse=True
    )

    important_words = word_scores[:top_n]

    return important_words


sample_text = X_test.iloc[0]

important_words = explain_prediction(sample_text)

with open(f"{OUTPUT_DIR}/important_words.txt", "w") as file:
    file.write("Important words for one sample prediction:\n\n")

    for word, score in important_words:
        file.write(f"{word}: {score}\n")

print("\nImportant words saved to outputs/important_words.txt")


# ============================================================
# SAVE MODEL AND OBJECTS
# ============================================================

print("\nSaving model...")

model.save(f"{MODEL_DIR}/gru_topic_model.keras")

with open(f"{MODEL_DIR}/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)


# ============================================================
# CUSTOM PREDICTION FUNCTION
# ============================================================

def predict_topic(text):
    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded)

    predicted_class = np.argmax(prediction)

    predicted_topic = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction)

    return predicted_topic, confidence


# ============================================================
# CUSTOM PREDICTION TEST
# ============================================================

print("\nTesting custom prediction...")

custom_text = """
Artificial intelligence, software systems, mobile applications, internet platforms,
and data analysis tools are transforming modern technology companies.
"""

predicted_topic, confidence = predict_topic(custom_text)

print("\nCustom Text:")
print(custom_text)

print("\nPredicted Topic:")
print(predicted_topic)

print(f"Confidence: {confidence:.4f}")


# ============================================================
# FINISHED
# ============================================================

print("\nThe project run completed successfully.")
print("Check outputs folder :)")