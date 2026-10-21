
# Your Streamlit app code will go here

%%writefile app.py

import re
import streamlit as st
import torch
import torch.nn as nn
import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Streamlit page setup
st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review and let the trained RNN model "
    "predict whether the sentiment is positive or negative."
)

# RNN architecture — same as training
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out


# Load trained model and fitted preprocessing objects
@st.cache_resource
def load_artifacts():
    tf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")

    model = RNN(
        input_size=5000,
        hidden_size=128,
        num_layers=1
    )

    model.load_state_dict(
        torch.load("rnn_sentiment.pth", map_location="cpu")
    )

    model.eval()

    return model, tf, le


# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - {
    "not", "no", "nor", "never"
}

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords while preserving negation words
    tokens = [
        word for word in tokens
        if word.lower() not in stop_words
    ]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# User input
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: I did not enjoy this movie.",
    height=160
)

if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        try:
            model, tf, le = load_artifacts()

            # Apply preprocessing
            cleaned_review = preprocess_text(review)

            # Convert text into TF-IDF features
            features = tf.transform([cleaned_review]).toarray()

            # RNN input shape: batch, sequence length, features
            x = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(1)

            # Predict
            with torch.no_grad():
                score = model(x)
                probability = torch.sigmoid(score).item()

            # Convert probability into predicted class
            predicted_class = int(probability >= 0.5)

            sentiment = le.inverse_transform(
                [predicted_class]
            )[0]

            st.subheader(
                f"Predicted sentiment: {str(sentiment).upper()}"
            )

            st.metric(
                "Model probability for class 1",
                f"{probability:.2%}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
