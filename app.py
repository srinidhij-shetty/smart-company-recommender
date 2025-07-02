import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import torch

st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.05);
    }

    h1 {
        color: #222831;
        font-family: 'Segoe UI', sans-serif;
    }

    .stTextInput > label, .stButton {
        font-weight: 500;
        color: #393e46;
    }

    .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
        color: #444;
    }
    </style>
""", unsafe_allow_html=True)


# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load company data
@st.cache_data
def load_data():
    with open("companies.json", "r", encoding="utf-8") as file:
        return json.load(file)

companies = load_data()

# Extract descriptions
company_descriptions = [company["description"] for company in companies]
company_embeddings = model.encode(company_descriptions, convert_to_tensor=True)

# --- UI ---
st.markdown("<h1 style='text-align: center;'>🤖 Smart Company Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Find the best-fit company for your need — powered by AI</p>", unsafe_allow_html=True)
st.write("")

user_input = st.text_input("📝 **What do you need a company for?**")

if st.button("🔍 Find Best Matches") and user_input:
    with st.spinner("Analyzing your request..."):
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, company_embeddings)

        top_k = 3
        top_results = torch.topk(cosine_scores, k=top_k)

        st.subheader("🎯 Top AI Matches")
        for score, idx in zip(top_results.values[0], top_results.indices[0]):
            match = companies[idx]

            # Keyword explanation
            description_words = set(match['description'].lower().split())
            query_words = set(user_input.lower().split())
            common_words = description_words & query_words

            if common_words:
                explanation = "🔍 Matched based on: " + ", ".join([f"`{word}`" for word in common_words])
            else:
                explanation = "🧠 Semantic match based on meaning — no exact word overlap."

            # Display as styled card
            st.markdown("---")
            st.markdown(f"### 🏢 {match['name']}")
            st.markdown(f"📄 _{match['description']}_")
            st.markdown(f"🌐 [Visit Website]({match['website']})")
            st.markdown(f"📈 **Similarity Score**: `{score:.2f}`")
            st.info(explanation)

        st.success("✅ Done! Refine your query for even better matches.")
