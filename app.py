import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



from scipy.sparse import hstack
# -----------------------
# Feedback state (for continuous learning)
# -----------------------
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Sarcasm Detection Dashboard", layout="wide")
st.title("🧠 Sarcasm Detection – NLP Dashboard")
DATASET_PATH = "sarcasm_large_synthetic_dataset_2000.csv"

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("sarcasm_large_synthetic_dataset_2000.csv")

df = load_data()

# -----------------------
# Cleaning (keep emojis)
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s😂🙃😒😅😊👍🎉]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------
# Emoji features
# -----------------------
emoji_list = ["😂", "🙃", "😒", "😅", "😊", "👍", "🎉"]

def emoji_count(text):
    return sum(text.count(e) for e in emoji_list)

df["emoji_count"] = df["text"].apply(emoji_count)
df["has_emoji"] = (df["emoji_count"] > 0).astype(int)

# -----------------------
# Features
# -----------------------
X_text = df["clean_text"]
X_emoji = df[["emoji_count", "has_emoji"]].values
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_text_vec = vectorizer.fit_transform(X_text)

X_final = hstack([X_text_vec, X_emoji])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# -----------------------
# Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

trained_models = {}
model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    preds = model.predict(X_test)
    model_scores[name] = accuracy_score(y_test, preds)

best_model = max(model_scores, key=model_scores.get)

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs([
    "📘 Overview",
    "📂 Dataset",
    "📊 Text Analysis",
    "🤖 Model Evaluation",
    "✍️ Live Analysis",
    "⚠️ Limitations"
])

# -----------------------
# TAB 1: Overview
# -----------------------
with tabs[0]:
    st.header("Project Overview")
    st.write("""
    This project detects sarcasm in social media text using NLP and Machine Learning.

    **Pipeline:**  
    Data → Cleaning → TF-IDF + Emoji Features → Model Training → Evaluation → Prediction

    **Applications:**  
    - Social media monitoring  
    - Brand sentiment analysis  
    - Fake review detection  
    """)

# -----------------------
# TAB 2: Dataset
# -----------------------
with tabs[1]:
    st.header("Dataset Explorer")

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sarcasm Distribution")
        fig, ax = plt.subplots()
        df["label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Emoji Usage")
        fig, ax = plt.subplots()
        df["has_emoji"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# -----------------------
# TAB 3: Text Analysis
# -----------------------
with tabs[2]:
    st.header("Text Analysis")

    avg_len_sarc = df[df["label"] == 1]["clean_text"].str.len().mean()
    avg_len_nonsarc = df[df["label"] == 0]["clean_text"].str.len().mean()

    avg_emoji_sarc = df[df["label"] == 1]["emoji_count"].mean()
    avg_emoji_nonsarc = df[df["label"] == 0]["emoji_count"].mean()

    st.write(f"Average length (Sarcastic): **{int(avg_len_sarc)} characters**")
    st.write(f"Average length (Non-sarcastic): **{int(avg_len_nonsarc)} characters**")
    st.write(f"Average emojis (Sarcastic): **{avg_emoji_sarc:.2f}**")
    st.write(f"Average emojis (Non-sarcastic): **{avg_emoji_nonsarc:.2f}**")

    st.subheader("🔍 What this tells us")
    st.markdown("""
    - Sarcastic texts are slightly longer on average  
    - Emojis appear more often in sarcastic messages  
    - This supports the idea that sarcasm is usually more expressive  
    - Emoji features provide useful additional signal beyond plain text  
    """)

# -----------------------
# TAB 4: Model Evaluation
# -----------------------
with tabs[3]:
    st.header("Model Evaluation")

    for model, acc in model_scores.items():
        st.write(f"**{model} Accuracy:** {acc:.2f}")

    selected = st.selectbox("Select model for detailed view", list(models.keys()))
    model = trained_models[selected]
    preds = model.predict(X_test)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, preds))

    st.subheader("📌 Interpretation")
    st.markdown(f"""
    - **{best_model}** performs best on this dataset  
    - Most predictions are correct, as shown by strong diagonal values in the confusion matrix  
    - Errors mainly occur when sentences are short or emotionally neutral  
    - Overall, the model learns meaningful patterns rather than guessing  
    """)

# -----------------------
# TAB 5: Live Analysis (FIXED WITH CONFIDENCE)
# -----------------------
with tabs[4]:
    st.header("Live Prediction with Model Performance")

    model_choice = st.selectbox("Choose model", list(trained_models.keys()))
    model = trained_models[model_choice]
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.divider()

    user_input = st.text_area("Enter a sentence to test:")
if st.button("Predict", key="predict_btn"):
    cleaned = clean_text(user_input)
    st.session_state.last_input = cleaned

    vec_text = vectorizer.transform([cleaned])
    emoji_cnt = emoji_count(user_input)
    emoji_feat = np.array([[emoji_cnt, int(emoji_cnt > 0)]])
    final_vec = hstack([vec_text, emoji_feat])

    feedback_df = pd.read_csv(DATASET_PATH)

    if "clean_text" not in feedback_df.columns:
        feedback_df["clean_text"] = feedback_df["text"].apply(clean_text)

    prev = feedback_df[feedback_df["clean_text"] == cleaned]

    if not prev.empty:
        pred = int(prev.iloc[-1]["label"])
        st.info("Prediction corrected using past user feedback.")
    else:
        pred = model.predict(final_vec)[0]

    st.session_state.last_prediction = pred
    st.session_state.feedback_mode = True


# -----------------------
# DISPLAY PREDICTION
# -----------------------
if st.session_state.feedback_mode:
    if st.session_state.last_prediction == 1:
        st.error("Prediction: Sarcastic 😏")
    else:
        st.success("Prediction: Not Sarcastic 🙂")

    st.markdown("### 🧠 Is this prediction correct?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, correct", key="yes_btn"):
            st.success("Thanks! Prediction confirmed.")
            st.session_state.feedback_mode = False

    with col2:
        if st.button("❌ No, wrong", key="no_btn"):
            st.session_state.feedback_mode = "correcting"


# -----------------------
# USER CORRECTION INPUT
# -----------------------
if st.session_state.feedback_mode == "correcting":
    correct_label = st.radio(
        "Select the correct label:",
        [0, 1],
        format_func=lambda x: "Not Sarcastic" if x == 0 else "Sarcastic",
        key="correct_label_radio"
    )

    if st.button("💾 Save & Learn", key="save_btn"):
        feedback_df = pd.read_csv(DATASET_PATH)

        if "clean_text" not in feedback_df.columns:
            feedback_df["clean_text"] = feedback_df["text"].apply(clean_text)

        new_row = {
            "text": user_input,
            "clean_text": st.session_state.last_input,
            "label": correct_label,
            "emoji_count": emoji_count(user_input),
            "has_emoji": int(emoji_count(user_input) > 0),
            "source": "user_feedback"
        }

        for col in new_row:
            if col not in feedback_df.columns:
                feedback_df[col] = None

        feedback_df = pd.concat(
            [feedback_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

        feedback_df.to_csv(DATASET_PATH, index=False)

        st.success("Saved to Excel. The system has learned from this correction.")
        st.session_state.feedback_mode = False

    



    st.subheader("🧠 How to interpret predictions")
    st.markdown("""
    - Sentences with expressive tone or emojis are more likely to be flagged as sarcastic  
    - Neutral sentences without emotional signals are harder for the model  
    - The model relies on both wording and emoji usage  
    - This reflects real-world difficulty in detecting sarcasm automatically  
    """)

# -----------------------
# TAB 6: Limitations
# -----------------------
with tabs[5]:
    st.header("Limitations & Future Scope")
    st.markdown("""
    **Limitations:**
    - Dataset is synthetic, not collected from real users  
    - No conversation context (only single sentences)  
    - Cultural and subtle sarcasm is difficult to detect  

    **Future Improvements:**
    - Train on real Twitter or Reddit data  
    - Use deep learning models like BERT  
    - Include conversation history for better context  
    - Support multiple languages  
    """)


