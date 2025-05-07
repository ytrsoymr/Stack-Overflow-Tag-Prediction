import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('punkt')
nltk.download('stopwords')

# --- Load artifacts ---
model=joblib.load(r"E:\Stack-Overflow\models\model.pkl")
vectorizer = joblib.load(r"E:\Stack-Overflow\models\vectorizer.pkl")  # TF-IDF vectorizer
mlb = joblib.load(r"E:\Stack-Overflow\models\mlb.pkl")             # MultiLabelBinarizer
# MultiLabelBinarizer

# --- Preprocessing function ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# --- Streamlit UI ---
st.set_page_config(page_title="Stack Overflow Tag Predictor", layout="centered")

st.title("💬 Stack Overflow Tag Predictor")
st.markdown("Enter a question (title + body) and get predicted tags.")

user_input = st.text_area("✍️ Question Title + Body", height=200)

top_k = st.slider("Number of tags to show", min_value=1, max_value=10, value=5)

if st.button("Predict Tags") and user_input.strip():
    cleaned = preprocess(user_input)
    X_vec = vectorizer.transform([cleaned])
    y_pred_proba = model.predict_proba(X_vec)

    # Get top-k tag predictions
    top_indices = y_pred_proba[0].argsort()[-top_k:][::-1]
    predicted_tags = [mlb.classes_[i] for i in top_indices]
    confidence = [y_pred_proba[0][i] for i in top_indices]

    st.markdown("### 🏷️ Predicted Tags:")
    for tag, conf in zip(predicted_tags, confidence):
        st.markdown(f"- **{tag}** (confidence: {conf:.2f})")
