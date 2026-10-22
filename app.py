import re
import streamlit as st
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


LEAK_PATTERNS = [
    r"\breuters\b", r"\bbreitbart\b", r"\bgetty images?\b",
    r"\bfeatured image\b", r"\btwitter com\b", r"\bpic twitter com\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(19|20)\d{2}\b",   # years
]


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)    # URLs
    s = re.sub(r"<.*?>", " ", s)             # HTML tags
    s = re.sub(r"[^a-z0-9\s]", " ", s)       # special characters
    for p in LEAK_PATTERNS:
        s = re.sub(p, " ", s)
    s = re.sub(r"\s+", " ", s).strip()       # extra spaces
    return s


# Mapping used in training: 0 = Real, 1 = Fake
LABEL_NAMES = {0: "Real News", 1: "Fake News"}

st.title("📰 Fake News Detector")
st.write("Enter a news headline and article to classify the news.")


MODEL_PATH = Path(__file__).parent.parent / "final_model_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Check that final_model_pipeline.pkl exists at the path set in MODEL_PATH.")
    st.stop()


# 5. User Inputs

headline = st.text_input("News Headline")
article = st.text_area("News Article", height=200)


# 6. Prediction

if st.button("Check News", type="primary"):

    if not headline.strip() and not article.strip():
        st.warning("Please enter a headline or article.")

    else:
        
        text = clean_text(f"{headline} {article}")

        if not text:
            st.warning("The text has no usable words after cleaning.")
            st.stop()

        try:
            prediction = int(model.predict([text])[0])
            probabilities = model.predict_proba([text])[0]
            prob_by_class = dict(zip(model.classes_, probabilities))

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)
            col1.metric(LABEL_NAMES[0], f"{prob_by_class[0]:.2%}")
            col2.metric(LABEL_NAMES[1], f"{prob_by_class[1]:.2%}")

            if prediction == 0:
                st.success("✅ Prediction: Real News")
            else:
                st.error("⚠️ Prediction: Fake News")

            if max(probabilities) < 0.65:
                st.info(
                    "Low confidence: the model is unsure about this article. "
                    "Treat this result with extra caution."
                )

            st.caption(
                "This is an ML prediction of writing style, not a fact-check. "
                "Verify important news with reliable sources."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
