
import re
import os
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import fitz

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    GRU,
    Dense,
    Dropout,
    Layer,
    Input,
    SpatialDropout1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))
print("=" * 60)



# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE_DIR, "BBC News Train.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# DOWNLOAD STOPWORDS
# ============================================================

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLOADING DATASET...")

df = pd.read_csv(TRAIN_FILE)

# ============================================================
# FIX COLUMN NAMES
# ============================================================

df.columns = [c.strip() for c in df.columns]

rename_map = {}

for col in df.columns:

    if col.lower() == "category":
        rename_map[col] = "Category"

    if col.lower() == "text":
        rename_map[col] = "Text"

df.rename(columns=rename_map, inplace=True)

print(df.head())

print("\nDataset Shape:", df.shape)

# ============================================================
# TEXT preprocessing
# ============================================================


def clean_text(text):

    text = str(text).lower()

    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():

        # keep meaningful words
        if len(word) > 2 and word not in stop_words:
            words.append(word)

    return " ".join(words)

print("\nCleaning text...")

df["clean_text"] = df["Text"].apply(clean_text)

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["Category"])

class_names = label_encoder.classes_

NUM_CLASSES = len(class_names)

print("\nClasses:")
print(class_names)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = df["clean_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ============================================================
# TOKENIZATION
# ============================================================

MAX_WORDS = 15000
MAX_LENGTH = 250

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",  #out of vocabulary used for unknown words
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)

# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, x):

        # Alignment scores
        e = tf.keras.backend.tanh(
            tf.keras.backend.dot(x, self.W) + self.b
        )

        # Attention weights
        a = tf.keras.backend.softmax(e, axis=1)

        # Context vector
        output = x * a

        output = tf.keras.backend.sum(output, axis=1)

        return output

# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

inputs = Input(shape=(MAX_LENGTH,))

# Embedding
x = Embedding(
    input_dim=MAX_WORDS,
    output_dim=128,
    mask_zero=True,
)(inputs)

# Spatial dropout
x = SpatialDropout1D(0.3)(x)

# BiGRU
x = Bidirectional(
    GRU(
        64,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.0,
    )
)(x)

# Attention
x = AttentionLayer()(x)

# Dense layers
x = Dropout(0.4)(x)

x = Dense(64, activation="relu")(x)

x = Dropout(0.3)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax",
)(x)

model = Model(inputs, outputs)

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test,
    verbose=0,
)

y_pred_prob = model.predict(X_test_pad, verbose=0)

y_pred = np.argmax(y_pred_prob, axis=1)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"F1 Macro      : {f1_macro:.4f}")
print(f"F1 Weighted   : {f1_weighted:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
).plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
    colorbar=False,
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png",
    ),
    dpi=150,
)

plt.show()

# ============================================================
# TRAINING CURVES
# ============================================================

epochs_range = range(
    1,
    len(history.history["accuracy"]) + 1,
)

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    history.history["accuracy"],
    label="Train Accuracy",
)

plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_accuracy.png",
    ),
    dpi=150,
)

plt.show()

# ============================================================
# WORD CLOUDS
# ============================================================

print("\nGenerating word clouds...")

for topic in class_names:

    text = " ".join(
        df[df["Category"] == topic]["clean_text"]
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,
    ).generate(text)

    plt.figure(figsize=(10, 5))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.title(f"Word Cloud - {topic}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"wordcloud_{topic}.png",
        ),
        dpi=150,
    )

    plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

model.save(
    os.path.join(
        MODEL_DIR,
        "bbc_bigru_attention.keras",
    )
)

with open(
    os.path.join(
        MODEL_DIR,
        "tokenizer.pkl",
    ),
    "wb",
) as f:

    pickle.dump(tokenizer, f)

with open(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl",
    ),
    "wb",
) as f:

    pickle.dump(label_encoder, f)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_topic(text):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    pred = model.predict(padded, verbose=0)

    idx = np.argmax(pred)

    topic = label_encoder.inverse_transform([idx])[0]

    confidence = float(np.max(pred))

    probs = {
        class_names[i]: float(pred[0][i])
        for i in range(NUM_CLASSES)
    }

    return topic, confidence, probs

# ============================================================
# DEMO PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOM PREDICTIONS")
print("=" * 60)

samples = [
    (
        "Tech",
        "Artificial intelligence and machine learning "
        "software are transforming technology companies."
    ),

    (
        "Politics",
        "The prime minister announced new economic "
        "policies during parliament."
    ),

    (
        "Sport",
        "The striker scored a hat trick in the "
        "football league final."
    ),

    (
        "Entertainment",
        "The movie achieved huge box office success "
        "and streaming growth."
    ),

    (
        "Business",
        "Retail sales increased as consumer confidence "
        "improved last quarter."
    ),
]

for expected, text in samples:

    topic, confidence, probs = predict_topic(text)

    print(f"\n[Expected: {expected}]")
    print("Predicted :", topic)
    print(f"Confidence: {confidence:.2%}")

    sorted_probs = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print("All probs :", {
        k: f"{v:.2%}"
        for k, v in sorted_probs
    })

# ============================================================
# extracting text from document
# ============================================================

import fitz 
from docx import Document


def extract_text_from_file(file_path):

    ext = os.path.splitext(file_path)[1].lower()


    if ext == ".txt":

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


    elif ext == ".pdf":

        text = ""

        pdf = fitz.open(file_path)

        for page in pdf:

            text += page.get_text()

        pdf.close()

        return text


    elif ext == ".docx":

        doc = Document(file_path)

        text = "\n".join(
            [para.text for para in doc.paragraphs]
        )
        return text


    else:

        raise ValueError(
            "Unsupported file type. "
            "Use TXT, PDF, or DOCX."
        )


# ============================================================
# PREDICT DOCUMENT
# ============================================================
def predict_document(document_path):

    document_path = document_path.strip()

    document_path = document_path.replace('"', "")
    document_path = document_path.replace("'", "")

    document_path = os.path.normpath(document_path)

    print("\nChecking file...")
    print("Path:", document_path)


    if not os.path.isfile(document_path):

        print("\nERROR: File not found.")
        return

    try:

        # ----------------------------------------------------
        # extract text from the given file
        # ----------------------------------------------------

        text = extract_text_from_file(document_path)

        # ----------------------------------------------------
        # empty file checking
        # ----------------------------------------------------

        if len(text.strip()) == 0:

            print("\nERROR: Empty document.")
            return

        # ----------------------------------------------------
        # topic prediction
        # ----------------------------------------------------

        topic, confidence, probs = predict_topic(text)

        # ----------------------------------------------------
        # results 
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DOCUMENT PREDICTION")
        print("=" * 60)

        print(f"\nDocument : {document_path}")

        print(f"Predicted Category : {topic}")

        print(f"Confidence : {confidence:.2%}")

        print("\nClass Probabilities:")

        sorted_probs = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for cls, prob in sorted_probs:

            print(f"{cls:<15} : {prob:.2%}")

    except Exception as e:

        print("\nERROR:")
        print(str(e))


# ============================================================
# extra input not from the dataset
# ============================================================

print("\n" + "=" * 60)
print("DOCUMENT CLASSIFIER")
print("=" * 60)

document_path = input(
    "\nEnter document path: "
)

predict_document(document_path)

# ============================================================
# finallllyyy
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY :)")
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Models saved to : {MODEL_DIR}")
print("=" * 60)