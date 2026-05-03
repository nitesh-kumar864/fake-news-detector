
import streamlit as st
import torch
from datetime import datetime
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------- Page Config --------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- Custom CSS----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
/* ---------------- GLOBAL ---------------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #1e1b4b, #0ea5e9);
    color: white;
}

/* Layout */
.block-container {
    padding-top: 2rem;
}

/* ---------------- HERO ---------------- */
.hero-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 8px;
    color: white;
}

.gradient-text {
    background: linear-gradient(135deg, #22d3ee, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    font-size: 16px;
    margin-bottom: 30px;
}

/* ---------------- MAIN CARD ---------------- */
.main-card {
    max-width: 900px;
    margin: auto;
    padding: 30px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(20px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}

/* ---------------- TEXTAREA ---------------- */
div[data-baseweb="textarea"] textarea {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
    padding: 15px !important;
    font-size: 15px;
}

/* Placeholder fix */
div[data-baseweb="textarea"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ---------------- BUTTONS ---------------- */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
    border: none;
}

/* Primary button (Analyze) */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
}

/* Normal buttons (Real + Fake) */
button[kind="secondary"] {
    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    color: white !important;
}

/* Hover */
div.stButton > button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}
            
/* ---------------- RESULT CARD ---------------- */
.result-card {
    margin-top: 25px;
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a, #334155);
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* Title */
.result-title {
    font-size: 26px;
    font-weight: 700;
    color: #22d3ee;
    margin-bottom: 10px;
}

/* Text */
.result-text {
    font-size: 18px;
    color: #e2e8f0;
}

/* Status colors */
.fake { color: #ef4444; font-weight: 700; }
.real { color: #22c55e; font-weight: 700; }
.uncertain { color: #f59e0b; font-weight: 700; }

/* Confidence */
.conf {
    margin-top: 10px;
    font-size: 15px;
    color: #cbd5f5;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 25px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)
# ---------------- Header ----------------
st.markdown(
    """
    <div class="hero-title">
        Detect Misinformation<br>
        <span class="gradient-text">Instantly & Accurately</span>
    </div>
    <div class="subtitle">
        Our AI analyzes it to predict whether the news is real or fake.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    MODEL_NAME = "nitesh-kumar864/FAKE-NEWS-DETECTOR"

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# ---------------- Input Card ----------------
st.markdown('<div class="btn-center">', unsafe_allow_html=True)

# Initialize state
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

def load_real_example():
    st.session_state.news_text = (
        "The Ministry of Education has launched a new digital learning "
        "initiative aimed at improving access to online courses for students in rural areas, officials said on Tuesday."
    )

def load_fake_example():
    st.session_state.news_text = (
        "Scientists have proven that sleeping for only two hours "
        "a day can make humans live up to 150 years without any health problems."
    )

news_text = st.text_area(
    "📰 Enter News Content",
    height=200,
    placeholder="Paste the headline or full news article here...",
    key="news_text"
)

col1, col2 = st.columns(2)

with col1:
    st.button("Try Real Example", on_click=load_real_example)

with col2:
    st.button("Try Fake Example", on_click=load_fake_example)

detect = st.button("Analyze", use_container_width=True, type="primary")

# ---------------- Prediction ----------------
if detect:
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                news_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=256
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)

        conf = confidence.item() * 100
        pred_label = pred.item()

        if pred_label == 1:
            label = "REAL"
            cls = "real"
        else:
            label = "FAKE"
            cls = "fake"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Prediction Result</div>
            <div class="result-text">
                This news appears to be: <span class="{cls}">{label}</span>
            </div>
            <div class="conf">Confidence: {conf:.2f}%</div>
            <div class="conf">(This result is generated by  Fake News Detection model)</div>
        </div>
        """, unsafe_allow_html=True)
# ---------------- Footer ----------------
year = datetime.now().year

st.markdown(
    f"<div class='footer'>© {year}  · Fake News Detection System</div>",
    unsafe_allow_html=True
)