import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load and preprocess
@st.cache_data
def load_data():
    df = pd.read_csv("data/balanced_ai_human_prompts.csv")
    df.dropna(subset=['text'], inplace=True)
    df['clean'] = df['text'].str.lower()
    return df

def get_top_features(vectorizer, model, text, top_n=5):
    vector = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    contributions = vector.toarray()[0] * coefs
    top_indicies = np.argsotr(np.abs(contributions))[::-1][:top_n]
    return [(feature_names[i], contributions[i]) for i in top_indicies]

df = load_data()

st.title("Human vs AI Essay Classifier")
st.write("Exploring different ML models on a Kaggle dataset.")

# Feature extraction
vectorizer = TfidfVectorizer(stop_words= 'english', max_features=5000)
X = vectorizer.fit_transform(df['clean'])
y = df['generated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Model training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append({"Model": name, "Accuracy": acc})

results_df = pd.DataFrame(results)

# Display
st.subheader("Model Accuracy Comparison")
st.dataframe(results_df)

best_model = results_df.loc[results_df['Accuracy'].idxmax()]
st.write(f"**Best model:** {best_model['Model']} "
          f"with accuracy {best_model['Accuracy']: .2%}")

# Custom prediction
st.subheader("Try your own text")
user_text = st.text_area("Paste an essay here:")
if user_text:
    X_user = vectorizer.transform([user_text.lower()])
    # Use best model for prediction
    final_model = models[best_model['Model']]
    pred = final_model.predict(X_user)[0]
    label = "AI-Generated" if pred == 1 else "Human-Written"
    st.success(f"Prediction: **{'label'}**")

    st.write("Top Influencial Words")
    top_feats = get_top_features(vectorizer, final_model, user_text.lower())
    exp_df = pd.DataFrame(top_feats, columns=['Word', 'Contribution'])
    exp_df['Sign'] = exp_df['Contribution'].apply(lambda x: 'AI signal' if x > 0 else 'Human signal')
    st.dataframe(exp_df[['Word', 'Sign']])
    st.caption("Positive contributions lean towards AI, negative ones towards Human.")