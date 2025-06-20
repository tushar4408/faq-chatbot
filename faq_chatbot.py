import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data from CSV ---
@st.cache_data
def load_faq():
    df = pd.read_csv("faq_data.csv")
    return df['Question'].tolist(), df['Answer'].tolist()

questions, answers = load_faq()
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# --- Streamlit Setup ---
st.set_page_config(page_title="FAQ ChatBot", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>🤖 FAQ ChatBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask me anything from the available FAQs.</p>", unsafe_allow_html=True)

# --- Chat Style CSS ---
st.markdown("""
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
        }
        .message {
            padding: 10px 15px;
            margin: 10px;
            border-radius: 15px;
            width: fit-content;
            max-width: 80%;
            clear: both;
        }
        .user {
            background-color: #dcf8c6;
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background-color: #f1f0f0;
            margin-right: auto;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# --- Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input ---
user_input = st.text_input("Type your question below 👇", key="user_input")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))

            # Find best match
            user_vector = vectorizer.transform([user_input])
            similarity_scores = cosine_similarity(user_vector, question_vectors)
            best_match_idx = similarity_scores.argmax()
            best_score = similarity_scores[0][best_match_idx]

            threshold = 0.4
            if best_score >= threshold:
                reply = answers[best_match_idx]
            else:
                reply = "I'm not sure about that. Please try asking something from the available FAQs."

            st.session_state.chat_history.append(("bot", reply))

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

# --- Display Messages ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for sender, msg in st.session_state.chat_history:
    style = "user" if sender == "user" else "bot"
    st.markdown(f"<div class='message {style}'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
