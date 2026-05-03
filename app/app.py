
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------- Page Config --------------
st.set_page_config(
    page_title="Veritas AI - Fake News Detector",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- Custom CSS----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

.gradient-text {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 35px;
}


.btn {
    background: #2563eb;
    color: white;
    padding: 12px 26px;
    border-radius: 999px;
    font-weight: 600;
    border: none;
}

.btn:hover {
    background: #1d4ed8;
}

.result-real {
    background: #ecfdf5;
    color: #065f46;
    padding: 18px;
    border-radius: 14px;
    font-weight: 600;
    text-align: center;
}

.result-fake {
    background: #fee2e2;
    color: #7f1d1d;
    padding: 18px;
    border-radius: 14px;
    font-weight: 600;
    text-align: center;
}

.result-uncertain {
    background: #fff7ed;
    color: #92400e;
    padding: 18px;
    border-radius: 14px;
    font-weight: 600;
    text-align: center;
}


.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 40px;
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
        Paste your article, tweet, or message below.
        Our AI analyzes it to predict whether the news is real or fake.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("../model/saved_model")
    model = DistilBertForSequenceClassification.from_pretrained(
        "../model/saved_model",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

tokenizer, model = load_model()


# ---------------- Input Card ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

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

detect = st.button("🔍 Analyze Veracity", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction ----------------
if detect:
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing content using AI..."):
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
        word_count = len(news_text.split())

        st.markdown("## 📊 Analysis Result")

        # ✅ FIXED LOGIC
        if word_count < 8:
            st.markdown(
                f"<div class='result-uncertain'>⚠️ Too Short heading<br>Confidence: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

        elif conf < 50:
            st.markdown(
                f"<div class='result-uncertain'>⚠️ Low Confidence<br>Confidence: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

        elif pred_label == 1:
            st.markdown(
                f"<div class='result-real'>🟢 Likely REAL<br>Confidence: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"<div class='result-fake'>🔴 Likely FAKE<br>Confidence: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(int(conf))
# ---------------- Footer ----------------
st.markdown(
    "<div class='footer'>© 2025 Veritas AI · Fake News Detection System</div>",
    unsafe_allow_html=True
)