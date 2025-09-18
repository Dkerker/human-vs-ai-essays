import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px

# ---------------- Load data ----------------
@st.cache_data
def load_data(path="data/AI_Human.csv", nrows=100000):
    df = pd.read_csv(path, nrows=nrows)
    df.dropna(subset=['text'], inplace=True)
    df['clean'] = df['text'].astype(str).str.lower()
    return df

df = load_data()

st.subheader("Sanity check: shuffled labels")
if st.checkbox("Run shuffled-label sanity check"):
    # Split normally
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['generated'], train_size=10000, test_size=5000, random_state=42
    )

    # Shuffle training labels
    y_train_shuffled = np.random.permutation(y_train)

    # Vectorize text
    vectorizer_sanity = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf_sanity = vectorizer_sanity.fit_transform(X_train)
    X_test_tfidf_sanity = vectorizer_sanity.transform(X_test)

    # Train model
    model_sanity = LogisticRegression(max_iter=1000)
    model_sanity.fit(X_train_tfidf_sanity, y_train_shuffled)

    # Predict on test set
    y_pred_sanity = model_sanity.predict(X_test_tfidf_sanity)
    accuracy_sanity = accuracy_score(y_test, y_pred_sanity)

    st.write(f"Accuracy with shuffled labels: {accuracy_sanity:.4f}")


X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['generated'],
    train_size=10000,
    test_size=5000,
    stratify=df['generated'],
    random_state=42
)

st.write(f"Training on {len(X_train)} essays")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train_tfidf, y_train)



y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.4f}")
st.subheader("Classification report")
st.text(classification_report(y_test, y_pred))

st.subheader("Predict a new essay")
user_input = st.text_area("Enter essay text here:")
if st.button("Predict"):
    if user_input.strip():
        clean_input = user_input.strip().lower()
        X_new_tfidf = vectorizer.transform([clean_input])
        prediction = model.predict(X_new_tfidf)[0]
        readable_prediction = {0.0: "Human", 1.0: "AI"}[prediction]
        st.write(f"{readable_prediction}")
    else:
        st.write("Please enter some text to predict")
