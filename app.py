import streamlit as st
import math
from pathlib import Path

import joblib

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI – Analisis Sentimen Indonesia",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #080b14;
    color: #dde2f0;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1629 0%, #131a2e 60%, #0e1422 100%);
    border: 1px solid rgba(100,120,255,0.15);
    border-radius: 20px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner h1 {
    font-size: 1.95rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 70%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    line-height: 1.2;
}
.hero-banner p {
    color: #8892b0;
    font-size: 0.92rem;
    margin: 0;
    line-height: 1.65;
}

/* ── Model cards (top row) ── */
.model-card {
    background: #0f1423;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
    margin-bottom: 6px;
}
.model-card .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 8px;
}
.model-card .model-name {
    font-size: 1rem;
    font-weight: 700;
    color: #e0e5f5;
    margin-bottom: 4px;
}
.model-card .model-repo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4b5875;
    word-break: break-all;
}

/* ── Input area ── */
.stTextArea label {
    color: #a5b4fc !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea {
    background: #0d1120 !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    color: #dde2f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f52c8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px 28px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.35) !important;
}

/* ── Result cards ── */
.result-card {
    background: #0d1221;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.07);
    transition: transform 0.2s;
}
.result-card:hover { transform: translateY(-2px); }

.result-card .rc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 10px;
}
.result-card .rc-model {
    font-weight: 700;
    font-size: 0.95rem;
    color: #c7d0f0;
}
.result-card .rc-repo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4b5875;
    margin-top: 2px;
}

/* sentiment badges */
.badge-pos {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.3);
    padding: 5px 14px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
}
.badge-neg {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
    padding: 5px 14px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
}
.badge-neu {
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.3);
    padding: 5px 14px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
}

/* confidence bar row */
.bar-row { margin-bottom: 10px; }
.bar-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8892b0;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
}
.bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 22px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    color: white;
    min-width: 38px;
    transition: width 0.6s ease;
}
.bar-pos { background: linear-gradient(90deg, rgba(34,197,94,0.25), rgba(34,197,94,0.75)); }
.bar-neg { background: linear-gradient(90deg, rgba(239,68,68,0.25), rgba(239,68,68,0.75)); }
.bar-neu { background: linear-gradient(90deg, rgba(59,130,246,0.25), rgba(59,130,246,0.75)); }

/* consensus banner */
.consensus-box {
    background: linear-gradient(135deg, #0f1629, #141c34);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
}
.consensus-box::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
}
.consensus-box .c-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6b7280;
    margin-bottom: 10px;
}
.consensus-box .c-result {
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.consensus-box .c-detail {
    font-size: 0.82rem;
    color: #6b7280;
}
.c-pos { color: #22c55e; }
.c-neg { color: #ef4444; }
.c-neu { color: #60a5fa; }

/* result card header */
.result-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    background: #0d1221;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px 14px 0 0;
    padding: 18px 20px 14px;
    margin-top: 6px;
}

/* Streamlit progress bar color overrides */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* Spinner override */
.stSpinner { color: #818cf8 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Hide Streamlit footer/header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "save_models"
VECTOR_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODELS = {
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
}

LABEL_MAP = {
    "positive": ("Positif", "😊", "pos"),
    "negative": ("Negatif", "😞", "neg"),
    "neutral":  ("Netral",  "😐", "neu"),
    "pos":      ("Positif", "😊", "pos"),
    "neg":      ("Negatif", "😞", "neg"),
    "neu":      ("Netral",  "😐", "neu"),
}

def normalize_label(raw, classes=None) -> str:
    label = str(raw).lower().strip()
    if "pos" in label:   return "positive"
    if "neg" in label:   return "negative"
    if "neu" in label:   return "neutral"
    if "positif" in label: return "positive"
    if "negatif" in label: return "negative"
    if "netral" in label: return "neutral"

    numeric_label = label.removeprefix("label_")
    if numeric_label in ("0", "1", "2"):
        numeric_classes = {str(value).lower() for value in classes or []}
        if numeric_classes == {"0", "1"}:
            return "negative" if numeric_label == "0" else "positive"
        if numeric_label == "0": return "negative"
        if numeric_label == "1": return "neutral"
        if numeric_label == "2": return "positive"
    return "neutral"

# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorizer():
    return joblib.load(VECTOR_PATH)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return joblib.load(MODEL_DIR / MODELS[model_name])

# ─── Inference ────────────────────────────────────────────────────────────────
def decision_distribution(model, features, classes):
    decisions = model.decision_function(features)
    if getattr(decisions, "ndim", 1) == 2:
        decisions = decisions[0]
    elif len(classes) == 2:
        score = float(decisions[0])
        decisions = [-score, score]

    decisions = [float(score) for score in decisions]
    peak = max(decisions)
    exp_scores = [math.exp(score - peak) for score in decisions]
    total = sum(exp_scores)
    probabilities = [score / total for score in exp_scores]
    return decisions, probabilities


def predict(vectorizer, model, text: str) -> dict:
    features = vectorizer.transform([text])
    classes = list(getattr(model, "classes_", []))
    is_svc = model.__class__.__name__ == "SVC"

    if is_svc and hasattr(model, "decision_function"):
        decisions, probabilities = decision_distribution(model, features, classes)
        predicted_label = classes[max(range(len(decisions)), key=decisions.__getitem__)]
    else:
        predicted_label = model.predict(features)[0]
        if not classes:
            classes = [predicted_label]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
        elif hasattr(model, "decision_function"):
            _, probabilities = decision_distribution(model, features, classes)
        else:
            probabilities = [1.0 if label == predicted_label else 0.0 for label in classes]

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for label, probability in zip(classes, probabilities):
        canonical = normalize_label(label, classes)
        scores[canonical] += float(probability)

    predicted = normalize_label(predicted_label, classes)
    return {"scores": scores, "predicted": predicted}

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🧠 Analisis Sentimen Indonesia</h1>
    <p>Bandingkan prediksi enam model machine learning secara bersamaan.<br>
    Masukkan teks, klik Analisis, dan lihat konsensus seluruh model.</p>
</div>
""", unsafe_allow_html=True)

# ── Model info row ────────────────────────────────────────────────────────────
model_items = list(MODELS.items())
for start in range(0, len(model_items), 3):
    cols = st.columns(3)
    for offset, (col, (name, filename)) in enumerate(
        zip(cols, model_items[start:start + 3])
    ):
        with col:
            st.markdown(f"""
            <div class="model-card">
              <div class="badge">Model {start + offset + 1}</div>
              <div class="model-name">{name}</div>
              <div class="model-repo">{filename}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Text input ────────────────────────────────────────────────────────────────
user_text = st.text_area(
    "📝 Masukkan teks untuk dianalisis",
    placeholder="Ketik atau paste teks bahasa Indonesia di sini…",
    height=140,
)

model_options = ["Semua model", *MODELS.keys()]
selected_model = st.selectbox("Pilih model yang akan diuji", model_options)
selected_models = list(MODELS) if selected_model == "Semua model" else [selected_model]

c1, c2 = st.columns([3, 1])
with c1:
    model_label = "6 Model" if selected_model == "Semua model" else selected_model
    run_btn = st.button(
        f"🔍 Analisis Sentimen dengan {model_label}", use_container_width=True
    )
with c2:
    clear_btn = st.button("🗑️ Bersihkan", use_container_width=True)

if clear_btn:
    st.rerun()

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn:
    if not user_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        results = {}
        progress = st.progress(0, text="Memuat model dan menganalisis…")

        for i, model_name in enumerate(selected_models):
            with st.spinner(f"Memuat {model_name}…"):
                try:
                    vectorizer = load_vectorizer()
                    model = load_model(model_name)
                    results[model_name] = predict(vectorizer, model, user_text.strip())
                except Exception as e:
                    results[model_name] = {"error": str(e)}
            progress.progress(
                (i + 1) / len(selected_models), text=f"Selesai: {model_name}"
            )

        progress.empty()

        # ── Consensus ────────────────────────────────────────────────────────
        valid = {k: v for k, v in results.items() if "error" not in v}
        if len(valid) > 1:
            counts = {"positive": 0, "negative": 0, "neutral": 0}
            for v in valid.values():
                counts[v["predicted"]] += 1
            consensus = max(counts, key=counts.get)
            avg_conf = sum(v["scores"][v["predicted"]] for v in valid.values()) / len(valid)
            agree_n = counts[consensus]
            total_n = len(valid)

            name_c, emoji_c, css_c = LABEL_MAP.get(consensus, (consensus, "🤔", "neu"))
            all_agree = agree_n == total_n

            st.markdown(f"""
            <div class="consensus-box">
              <div class="c-label">Kesepakatan Model</div>
              <div class="c-result c-{css_c}">{emoji_c} {name_c}</div>
              <div class="c-detail">
                {"Semua " + str(total_n) + " model sepakat" if all_agree
                 else str(agree_n) + " dari " + str(total_n) + " model memilih " + name_c}
                 &nbsp;•&nbsp; Rata-rata kepercayaan: {avg_conf*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"### 📊 Hasil pengujian: {', '.join(selected_models)}")

        # ── Individual cards ──────────────────────────────────────────────────
        for model_name, res in results.items():
            repo = MODELS[model_name]
            short_repo = ("…" + repo[-38:]) if len(repo) > 40 else repo

            with st.container():
                st.markdown("""<div class="result-card-wrap">""", unsafe_allow_html=True)

                if "error" in res:
                    st.markdown(f"""
                    <div class="result-card" style="border-color:rgba(239,68,68,0.3)">
                      <div class="rc-header">
                        <div><div class="rc-model">{model_name}</div>
                        <div class="rc-repo">{short_repo}</div></div>
                        <span class="badge-neg">⚠ Error</span>
                      </div>
                      <p style="color:#ef4444;font-size:0.82rem;margin:0">{res['error']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    continue

                pred = res["predicted"]
                s = res["scores"]
                lname, lemoji, lcss = LABEL_MAP.get(pred, (pred, "🤔", "neu"))

                # Card header (HTML — no inline width needed here)
                st.markdown(f"""
                <div class="result-card-header">
                  <div>
                    <div class="rc-model">{model_name}</div>
                    <div class="rc-repo">{short_repo}</div>
                  </div>
                  <span class="badge-{lcss}">{lemoji} {lname}</span>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars — use st.progress (native, always renders)
                bar_data = [
                    ("😊 Positif", s["positive"], "#22c55e"),
                    ("😞 Negatif", s["negative"], "#ef4444"),
                    ("😐 Netral",  s["neutral"],  "#60a5fa"),
                ]
                for blabel, bval, _ in bar_data:
                    pct = bval * 100
                    col_label, col_bar, col_pct = st.columns([1.2, 6, 1])
                    with col_label:
                        st.markdown(
                            f"<div style='color:#8892b0;font-size:0.78rem;"
                            f"font-weight:600;padding-top:6px'>{blabel}</div>",
                            unsafe_allow_html=True
                        )
                    with col_bar:
                        st.progress(bval)
                    with col_pct:
                        st.markdown(
                            f"<div style='color:#d1d5db;font-size:0.82rem;"
                            f"font-weight:700;padding-top:6px;text-align:right'>{pct:.1f}%</div>",
                            unsafe_allow_html=True
                        )

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;color:#3a4460;font-size:0.75rem'>"
            "SentimentAI — Decision Tree · KNN · Logistic Regression · Naive Bayes · Random Forest · SVM</p>",
            unsafe_allow_html=True
        )
