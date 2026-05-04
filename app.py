import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LightweightSwin3D
from utils import attention_to_heatmap


st.set_page_config(
    page_title="Lung Nodule Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PREMIUM CLINICAL CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(14,165,233,0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(34,197,94,0.10), transparent 30%),
        linear-gradient(135deg, #020617 0%, #07111f 42%, #0f172a 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.premium-shell {
    border: 1px solid rgba(148, 163, 184, 0.20);
    background: rgba(15, 23, 42, 0.66);
    backdrop-filter: blur(18px);
    border-radius: 34px;
    padding: 34px;
    box-shadow:
        0 35px 100px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: #67e8f9;
    background: rgba(8,145,178,0.12);
    border: 1px solid rgba(103,232,249,0.22);
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 800;
}

.hero-title {
    margin-top: 22px;
    font-size: 58px;
    line-height: 1.02;
    font-weight: 950;
    letter-spacing: -0.05em;
    color: #ffffff;
}

.hero-gradient {
    background: linear-gradient(90deg, #ffffff, #93c5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 18px;
    max-width: 850px;
    font-size: 18px;
    line-height: 1.8;
    color: #cbd5e1;
}

.status-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 22px;
}

.status-pill {
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.20);
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
}

.panel {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 0 22px 60px rgba(0,0,0,0.32);
}

.panel-title {
    font-size: 19px;
    font-weight: 850;
    color: #ffffff;
    margin-bottom: 8px;
}

.panel-sub {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 18px;
}

.metric-card {
    background:
        linear-gradient(145deg, rgba(30,41,59,0.95), rgba(2,6,23,0.95));
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 24px;
    padding: 22px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

.metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 900;
}

.metric-value {
    margin-top: 10px;
    font-size: 34px;
    font-weight: 950;
    color: #ffffff;
    letter-spacing: -0.04em;
}

.risk-card-high {
    background:
        radial-gradient(circle at top left, rgba(248,113,113,0.22), transparent 30%),
        linear-gradient(135deg, rgba(127,29,29,0.90), rgba(69,10,10,0.94));
    border: 1px solid rgba(248,113,113,0.55);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 0 0 70px rgba(239,68,68,0.18);
}

.risk-card-low {
    background:
        radial-gradient(circle at top left, rgba(52,211,153,0.22), transparent 30%),
        linear-gradient(135deg, rgba(6,78,59,0.90), rgba(2,44,34,0.94));
    border: 1px solid rgba(52,211,153,0.55);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 0 0 70px rgba(16,185,129,0.18);
}

.risk-title {
    font-size: 24px;
    font-weight: 950;
    color: white;
    margin-bottom: 6px;
}

.risk-text {
    color: rgba(255,255,255,0.84);
    line-height: 1.7;
    font-size: 14px;
}

.viewer-frame {
    margin-top: 18px;
    background: rgba(248,250,252,0.98);
    border-radius: 28px;
    padding: 14px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.40);
    border: 1px solid rgba(255,255,255,0.15);
}

.footer-premium {
    margin-top: 54px;
    padding: 30px 24px;
    text-align: center;
    border-top: 1px solid rgba(148,163,184,0.18);
    background: linear-gradient(90deg, transparent, rgba(15,23,42,0.55), transparent);
}

.footer-name {
    font-size: 19px;
    font-weight: 850;
    color: #f8fafc;
}

.footer-sub {
    margin-top: 7px;
    color: #94a3b8;
    font-size: 13px;
}

.footer-links {
    margin-top: 17px;
}

.footer-links a {
    display: inline-block;
    margin: 6px 7px;
    padding: 10px 16px;
    border-radius: 999px;
    text-decoration: none;
    color: #e0f2fe !important;
    font-size: 13px;
    font-weight: 800;
    background: rgba(14,165,233,0.10);
    border: 1px solid rgba(125,211,252,0.28);
}

.footer-links a:hover {
    background: rgba(14,165,233,0.22);
    border: 1px solid rgba(125,211,252,0.55);
}

.disclaimer {
    margin-top: 16px;
    color: #64748b;
    font-size: 12px;
}

.stFileUploader {
    background: rgba(2,6,23,0.35);
    border-radius: 18px;
}

.stSlider label {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

button[kind="primary"] {
    border-radius: 999px !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = LightweightSwin3D(
        in_channels=1,
        embed_dim=32,
        num_heads=(2, 4, 8),
        window_size=4
    )
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()


def predict(volume):
    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, attn_maps = model(tensor, return_attention=True)
        prob = torch.sigmoid(logits).item()

    heatmap = attention_to_heatmap(
        attn_maps[-1],
        batch_index=0,
        target_size=(64, 64, 64)
    )

    return prob, heatmap


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="premium-shell">
    <div class="hero-eyebrow"> AI-Powered Clinical Imaging Intelligence</div>
    <div class="hero-title">
        Lung Nodule <span class="hero-gradient">Malignancy Intelligence</span>
    </div>
    <div class="hero-subtitle">
        A premium clinical-grade AI dashboard powered by a lightweight 3D Swin Transformer
        for nodule-level CT analysis, malignancy probability prediction, and explainable
        attention-based visualization.
    </div>
    <div class="status-row">
        <div class="status-pill">3D CT Nodule Input</div>
        <div class="status-pill">Transformer Attention</div>
        <div class="status-pill">LIDC-IDRI Prototype</div>
        <div class="status-pill">Research Demonstration</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# MAIN UI
# =========================================================
left, right = st.columns([0.88, 1.55], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Case Upload Console</div>
        <div class="panel-sub">
            Upload a preprocessed 3D nodule patch generated from the XML-based nodule extraction pipeline.
            Expected input shape: <b>(1, 64, 64, 64)</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload .npy nodule volume",
        type=["npy"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    threshold = st.slider(
        "Clinical decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.01
    )

    slice_idx = st.slider(
        "Axial slice selector",
        min_value=0,
        max_value=63,
        value=32,
        step=1
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Model Configuration</div>
        <div class="panel-sub">
            Architecture: <b>Lightweight 3D Swin Transformer</b><br>
            Loss: <b>Weighted BCE</b><br>
            Explainability: <b>Window attention heatmap</b><br>
            Output: <b>Benign / Malignant probability</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


with right:
    if uploaded_file is None:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">No Case Loaded</div>
            <div class="panel-sub">
                Upload a nodule volume to activate malignancy prediction, confidence scoring,
                and attention visualization.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        volume = np.load(uploaded_file)

        if volume.shape != (1, 64, 64, 64):
            st.error(f"Invalid input shape: {volume.shape}. Expected (1, 64, 64, 64).")

        else:
            prob, heatmap = predict(volume)

            prediction = "Malignant" if prob >= threshold else "Benign"
            confidence = prob if prediction == "Malignant" else 1 - prob

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">AI Classification</div>
                    <div class="metric-value">{prediction}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Malignancy Score</div>
                    <div class="metric-value">{prob:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if prediction == "Malignant":
                st.markdown(f"""
                <div class="risk-card-high">
                    <div class="risk-title">High-Risk AI Flag</div>
                    <div class="risk-text">
                        The model classified this nodule as <b>malignant</b> using a threshold of <b>{threshold:.2f}</b>.
                        This result should be interpreted as an AI research output and not as a final clinical diagnosis.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card-low">
                    <div class="risk-title">Low-Risk AI Flag</div>
                    <div class="risk-text">
                        The model classified this nodule as <b>benign</b> using a threshold of <b>{threshold:.2f}</b>.
                        This result should be interpreted as an AI research output and not as a final clinical diagnosis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            img = volume[0, slice_idx]
            hm = heatmap[slice_idx]

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            ax[0].imshow(img, cmap="gray")
            ax[0].set_title("CT Nodule Slice", fontsize=13, fontweight="bold")
            ax[0].axis("off")

            ax[1].imshow(img, cmap="gray")
            ax[1].imshow(hm, cmap="jet", alpha=0.48)
            ax[1].set_title("Transformer Attention Overlay", fontsize=13, fontweight="bold")
            ax[1].axis("off")

            st.markdown('<div class="viewer-frame">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-premium">
    <div class="footer-name">Developed by Gershom Richard Bruno</div>
    <div class="footer-sub">
        Biomedical AI • Medical Imaging • Deep Learning Systems • Clinical Decision Support Prototype
    </div>
    <div class="footer-links">
        <a href="https://www.linkedin.com/in/gershom-richard-bruno-17ab3327a/" target="_blank">LinkedIn</a>
        <a href="https://github.com/gershomrichardbruno" target="_blank">GitHub</a>
        <a href="https://gershomrichardbruno.github.io/gershomrichardbruno.com/" target="_blank">Portfolio</a>
    </div>
    <div class="disclaimer">
        LIDC-IDRI based research prototype • Not intended for clinical diagnosis • For academic and demonstration use only
    </div>
</div>
""", unsafe_allow_html=True)