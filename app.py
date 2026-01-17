import streamlit as st
import pickle
import re

st.set_page_config(page_title="Skill Gap Prediction", page_icon="📄")

# Load model & vectorizer
@st.cache_resource
def load_objects():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_objects()

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    return text

# UI
st.title("📊 Skill Gap Prediction from Resume Data")
st.write("Paste resume text to identify skill gaps")

resume_text = st.text_area("Paste Resume Text Here", height=250)

if st.button("🔍 Predict Skill Gap"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text")
    else:
        cleaned_text = clean_text(resume_text)
        vectorized_text = vectorizer.transform([cleaned_text])  # list!

        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("❌ Skill Gap Detected")
            st.write("Suggestions:")
            st.write("- Improve missing skills")
            st.write("- Take relevant courses")
            st.write("- Work on projects")
        else:
            st.success("✅ No Skill Gap Detected")
            st.write("Resume matches required skills")

st.markdown("---")
st.caption("ML Project | Streamlit Deployment")