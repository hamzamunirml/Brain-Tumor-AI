"""
Brain Tumor Detection - Streamlit App
CT & MRI image classification (Healthy vs Tumor)

Run with:  streamlit run app.py
"""

import os

import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_MODEL, MODEL_CONFIGS, MODELS_DIR
from src.model_utils import get_model_path, load_model, predict_image
from src.preprocessing import detect_scan_type, load_image, preprocess_image

st.set_page_config(
    page_title="BrainTumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GITHUB_URL = "https://github.com/hamzamunirml"
LINKEDIN_URL = "https://www.linkedin.com/in/hamza-munir-8b2159367/"


# =============================================================================
# Cached model loading
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_model(model_name: str):
    return load_model(model_name)


def available_models():
    """Only list models whose .keras file actually exists in models/."""
    return [name for name in MODEL_CONFIGS if os.path.exists(get_model_path(name))]


# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #05070d;
        --card-bg: #0d1424;
        --card-bg-soft: #0a0f1d;
        --card-border: #1e293b;
        --accent: #3b82f6;
        --accent-light: #60a5fa;
        --text: #e2e8f0;
        --text-dim: #94a3b8;
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
    }

    .stApp { background-color: var(--bg); }
    #MainMenu, header, footer { visibility: hidden; }
    .block-container { padding-top: 1.2rem; max-width: 1200px; }

    /* ---- Navbar ---- */
    .navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 22px; background: #070a13; border: 1px solid var(--card-border);
        border-radius: 14px; margin-bottom: 28px;
    }
    .navbar .brand { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700; color: var(--text); }
    .navbar .brand span.accent { color: var(--accent-light); }
    .navbar .links { display: flex; gap: 26px; }
    .navbar .links a { color: var(--text-dim); text-decoration: none; font-size: 14.5px; }
    .navbar .links a.active { color: var(--accent-light); border-bottom: 2px solid var(--accent-light); padding-bottom: 4px; }
    .navbar .links a:hover { color: var(--accent-light); }
    .navbar .badge-pill { border: 1px solid var(--card-border); border-radius: 20px; padding: 6px 14px; font-size: 13px; color: var(--text-dim); text-decoration: none; }
    .navbar .badge-pill:hover { color: var(--accent-light); border-color: var(--accent-light); }

    /* ---- Hero ---- */
    .hero-title { font-size: 46px; font-weight: 800; line-height: 1.15; color: var(--text); margin-bottom: 6px; }
    .hero-title .accent { color: var(--accent-light); }
    .hero-subtitle { font-size: 20px; font-weight: 600; color: var(--accent-light); margin-bottom: 10px; }
    .hero-desc { font-size: 15.5px; color: var(--text-dim); margin-bottom: 22px; line-height: 1.55; }
    .btn-primary-link {
        display: inline-block; background: var(--accent); color: white !important; text-decoration: none;
        padding: 11px 22px; border-radius: 8px; font-weight: 600; font-size: 14.5px; margin-right: 10px;
    }
    .badges-row { display: flex; gap: 18px; margin-top: 20px; color: var(--text-dim); font-size: 13.5px; flex-wrap: wrap; }
    .badges-row span { display: flex; align-items: center; gap: 6px; }

    .hero-visual {
        border: 1px solid var(--card-border); border-radius: 16px; background: radial-gradient(circle at 30% 20%, #13213d, #070a13);
        display: flex; align-items: center; justify-content: center; flex-direction: column;
        height: 340px; gap: 10px;
    }
    .hero-visual .icon { font-size: 74px; }
    .hero-visual .cap { color: var(--text-dim); font-size: 13px; }

    /* ---- Cards / containers ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg) !important; border-color: var(--card-border) !important;
        border-radius: 14px !important;
    }
    .card-title { font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .section-title { font-size: 30px; font-weight: 800; color: var(--text); margin-bottom: 6px; }
    .section-subtitle { font-size: 15.5px; color: var(--text-dim); margin-bottom: 28px; line-height: 1.6; max-width: 780px; }

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: #070a13 !important; border: 1.5px dashed #2d3a52 !important; border-radius: 12px !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: var(--accent); color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 0;
    }
    .stButton > button:hover { background: var(--accent-light); color: white; }
    .stButton > button:disabled { background: #1e293b; color: var(--text-dim); }

    /* ---- Result banners ---- */
    .result-banner {
        border-radius: 10px; padding: 14px 18px; font-size: 19px; font-weight: 700; margin: 18px 0 20px 0;
    }
    .result-banner.tumor { background: rgba(239,68,68,0.12); color: var(--red); border: 1px solid rgba(239,68,68,0.35); }
    .result-banner.healthy { background: rgba(34,197,94,0.12); color: var(--green); border: 1px solid rgba(34,197,94,0.35); }

    .scan-badge { display: inline-block; background: #101a2e; border: 1px solid var(--card-border); border-radius: 8px; padding: 8px 12px; font-size: 13.5px; color: var(--text); margin: 14px 0 4px 0; }
    .placeholder-box { color: var(--text-dim); font-size: 14px; text-align: center; padding: 50px 10px; }

    /* ---- About page ---- */
    .stat-grid { display: flex; gap: 16px; flex-wrap: wrap; margin: 22px 0 8px 0; }
    .stat-chip { flex: 1; min-width: 150px; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; text-align: center; }
    .stat-chip .num { font-size: 26px; font-weight: 800; color: var(--accent-light); }
    .stat-chip .lbl { font-size: 12.5px; color: var(--text-dim); margin-top: 4px; }
    .about-text { color: var(--text-dim); font-size: 15px; line-height: 1.75; }
    .about-text b { color: var(--text); }
    .disclaimer-box { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 16px 18px; color: #fbbf24; font-size: 14px; margin-top: 20px; }

    /* ---- How it works (timeline) ---- */
    .timeline { position: relative; margin-top: 10px; padding-left: 6px; }
    .tl-item { display: flex; gap: 18px; padding-bottom: 30px; position: relative; }
    .tl-item:last-child { padding-bottom: 0; }
    .tl-line { position: absolute; left: 20px; top: 44px; bottom: -6px; width: 2px; background: var(--card-border); }
    .tl-item:last-child .tl-line { display: none; }
    .tl-num { flex-shrink: 0; width: 42px; height: 42px; border-radius: 50%; background: var(--accent); color: white;
              display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px; z-index: 1; }
    .tl-content { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px 20px; flex: 1; }
    .tl-content .tl-title { font-size: 16px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
    .tl-content .tl-desc { font-size: 14px; color: var(--text-dim); line-height: 1.65; }
    .tl-content code { background: #101a2e; color: var(--accent-light); padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }

    /* ---- Features page ---- */
    .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 18px; }
    .feature-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
    .feature-card .f-icon { font-size: 26px; margin-bottom: 10px; }
    .feature-card .f-title { font-size: 15.5px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
    .feature-card .f-desc { font-size: 13.5px; color: var(--text-dim); line-height: 1.55; }

    .imgfeat-group { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 22px; margin-bottom: 16px; }
    .imgfeat-group .ig-title { font-size: 16px; font-weight: 700; color: var(--accent-light); margin-bottom: 12px; }
    .imgfeat-group ul { margin: 0; padding-left: 20px; color: var(--text-dim); font-size: 14px; line-height: 1.9; }
    .imgfeat-group ul li b { color: var(--text); }

    /* ---- Contact page ---- */
    .contact-grid { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 20px; }
    .contact-card {
        flex: 1; min-width: 260px; background: var(--card-bg); border: 1px solid var(--card-border);
        border-radius: 14px; padding: 26px; text-align: center;
    }
    .contact-card .c-icon { font-size: 32px; margin-bottom: 12px; }
    .contact-card .c-title { font-size: 16px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
    .contact-card .c-desc { font-size: 13.5px; color: var(--text-dim); margin-bottom: 16px; }
    .contact-card a.c-link {
        display: inline-block; background: var(--accent); color: white !important; text-decoration: none;
        padding: 9px 20px; border-radius: 8px; font-weight: 600; font-size: 13.5px;
    }
    .contact-card a.c-link:hover { background: var(--accent-light); }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { color: var(--text-dim); }
    .stTabs [aria-selected="true"] { color: var(--accent-light) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Routing (query-param based "pages", single-file app)
# =============================================================================
VALID_PAGES = {"home", "about", "how-it-works", "features", "contact"}
page = st.query_params.get("page", "home")
if page not in VALID_PAGES:
    page = "home"


def render_navbar(active: str):
    def link(key, label):
        cls = "active" if key == active else ""
        return f'<a class="{cls}" href="?page={key}" target="_self">{label}</a>'

    st.markdown(
        f"""
        <div class="navbar">
            <a href="?page=home" target="_self" style="text-decoration:none;">
                <div class="brand">🧠 BrainTumor <span class="accent">AI</span></div>
            </a>
            <div class="links">
                {link("home", "Home")}
                {link("about", "About")}
                {link("how-it-works", "How It Works")}
                {link("features", "Features")}
                {link("contact", "Contact")}
            </div>
            <a class="badge-pill" href="?page=about" target="_self">ℹ️ About Project</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_navbar(page)


# =============================================================================
# PAGE: HOME
# =============================================================================
def render_home():
    hero_col1, hero_col2 = st.columns([1.1, 1])

    with hero_col1:
        st.markdown(
            """
            <div class="hero-title">BrainTumor <span class="accent">AI</span></div>
            <div class="hero-subtitle">AI-Powered Brain Scan Analysis</div>
            <div class="hero-desc">
                Detects Tumors from Brain CT &amp; MRI Scan Images<br/>
                with High Accuracy using Deep Learning
            </div>
            <a class="btn-primary-link" href="#analyze-anchor" target="_self">⬆️ Upload Scan</a>
            """,
            unsafe_allow_html=True,
        )
        demo_clicked = st.button("▶️ View Demo", key="demo_btn")
        st.markdown(
            """
            <div class="badges-row">
                <span>✅ AI Powered</span><span>•</span>
                <span>🎯 Accurate</span><span>•</span>
                <span>⚡ Fast</span><span>•</span>
                <span>🔒 Reliable</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if demo_clicked:
            st.info(
                "**Demo:** upload any brain CT or MRI scan below and click "
                "**Analyze Image** — the model will classify it as Healthy or "
                "Tumor with a confidence score and a visual breakdown."
            )

    with hero_col2:
        st.markdown(
            """
            <div class="hero-visual">
                <div class="icon">🧠</div>
                <div class="cap">Brain CT / MRI Scan Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div id='analyze-anchor'></div>", unsafe_allow_html=True)
    st.write("")

    found_models = available_models()

    _, center_col, _ = st.columns([0.15, 1.7, 0.15])
    with center_col:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">🔍 Analyze a Brain Scan</div>',
                unsafe_allow_html=True,
            )

            if not found_models:
                st.error(
                    "No model files found in `models/`.\n\n"
                    "Copy these `.keras` files from the notebook's output into "
                    "the `models/` folder:\n\n"
                    + "\n".join(
                        f"- `{cfg['filename']}`" for cfg in MODEL_CONFIGS.values()
                    )
                )
                selected_model_name = None
            else:
                default_index = (
                    found_models.index(DEFAULT_MODEL)
                    if DEFAULT_MODEL in found_models
                    else 0
                )
                selected_model_name = st.selectbox(
                    "AI Model", options=found_models, index=default_index
                )
                st.caption(MODEL_CONFIGS[selected_model_name]["description"])

            uploaded_file = st.file_uploader(
                "Drag & drop your scan here, or click to browse",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )
            st.caption("Supported formats: JPG, PNG, JPEG")

            analyze_clicked = st.button(
                "🔍 Analyze Image",
                use_container_width=True,
                disabled=(uploaded_file is None or not found_models),
            )

            if (
                analyze_clicked
                and uploaded_file is not None
                and selected_model_name is not None
            ):
                img = load_image(uploaded_file)
                with st.spinner("Analyzing scan..."):
                    scan_info = detect_scan_type(img)
                    model = get_cached_model(selected_model_name)
                    processed = preprocess_image(img)
                    result = predict_image(model, processed)

                st.session_state["bt_image"] = img
                st.session_state["bt_scan_info"] = scan_info
                st.session_state["bt_result"] = result
                st.session_state["bt_model_used"] = selected_model_name

            # ---- Result flow: image -> scan type -> verdict -> confidence -> chart ----
            if "bt_result" in st.session_state:
                st.markdown("---")

                img = st.session_state["bt_image"]
                scan_info = st.session_state["bt_scan_info"]
                result = st.session_state["bt_result"]

                # 1. Uploaded image
                st.image(img, use_container_width=True)

                # 2. Scan type (CT or MRI)
                badge = (
                    "🩻 CT Scan" if scan_info["scan_type"] == "CT" else "🧲 MRI Scan"
                )
                st.markdown(
                    f'<div class="scan-badge">{badge} (estimated, {scan_info["confidence"]*100:.0f}% confidence)</div>',
                    unsafe_allow_html=True,
                )

                # 3. Tumor / Healthy verdict
                label = result["label"]
                confidence = result["confidence"]
                banner_class = "tumor" if label == "Tumor" else "healthy"
                banner_icon = "⚠️" if label == "Tumor" else "✅"
                banner_text = (
                    "Tumor Detected"
                    if label == "Tumor"
                    else "No Tumor Detected — Healthy"
                )
                st.markdown(
                    f'<div class="result-banner {banner_class}">{banner_icon} {banner_text}</div>',
                    unsafe_allow_html=True,
                )

                # 4. Confidence score — gauge chart
                gauge_color = "#ef4444" if label == "Tumor" else "#22c55e"
                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={
                            "suffix": "%",
                            "font": {"color": "#e2e8f0", "size": 34},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": "#94a3b8",
                                "tickfont": {"color": "#94a3b8"},
                            },
                            "bar": {"color": gauge_color},
                            "bgcolor": "#0d1424",
                            "borderwidth": 1,
                            "bordercolor": "#1e293b",
                            "steps": [
                                {"range": [0, 50], "color": "#101a2e"},
                                {"range": [50, 100], "color": "#131f36"},
                            ],
                        },
                        title={
                            "text": "Confidence Score",
                            "font": {"color": "#94a3b8", "size": 14},
                        },
                    )
                )
                gauge_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(l=25, r=25, t=40, b=10),
                )
                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                # 5. Tumor vs Healthy percentage breakdown — bar chart
                probs = sorted(
                    result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
                )
                bar_colors = {"Healthy": "#22c55e", "Tumor": "#ef4444"}
                bar_fig = go.Figure(
                    go.Bar(
                        x=[p * 100 for _, p in probs],
                        y=[name for name, _ in probs],
                        orientation="h",
                        marker_color=[bar_colors[name] for name, _ in probs],
                        text=[f"{p*100:.2f}%" for _, p in probs],
                        textposition="outside",
                    )
                )
                bar_fig.update_layout(
                    title={
                        "text": "All Predictions",
                        "font": {"color": "#94a3b8", "size": 14},
                    },
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=180,
                    margin=dict(l=10, r=30, t=40, b=20),
                    xaxis=dict(range=[0, 105], showgrid=False, showticklabels=False),
                    yaxis=dict(color="#e2e8f0"),
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(
                    bar_fig, use_container_width=True, config={"displayModeBar": False}
                )

                st.caption(f"Model used: {st.session_state['bt_model_used']}")
            else:
                st.markdown(
                    '<div class="placeholder-box">Upload a scan and click '
                    "<b>Analyze Image</b> to see the prediction here.</div>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# PAGE: ABOUT
# =============================================================================
def render_about():
    st.markdown(
        '<div class="section-title">About This Project</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">BrainTumor AI is a deep-learning based web app that '
        "classifies brain CT and MRI scans as Healthy or showing signs of a Tumor.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-chip"><div class="num">3</div><div class="lbl">Deep Learning Models</div></div>
            <div class="stat-chip"><div class="num">2</div><div class="lbl">Scan Modalities (CT &amp; MRI)</div></div>
            <div class="stat-chip"><div class="num">96.6%</div><div class="lbl">Best Model Test Accuracy</div></div>
            <div class="stat-chip"><div class="num">0.993</div><div class="lbl">Best Model AUC</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div class="about-text">
            <p><b>What it does:</b> the app takes an uploaded brain scan image, estimates whether it's
            a CT or MRI scan, and runs it through a trained convolutional neural network to predict
            <b>Healthy</b> or <b>Tumor</b>, along with a confidence score and full probability breakdown.</p>

            <p><b>How it was built:</b> three deep learning architectures were trained and compared on a
            labeled brain-scan dataset (CT &amp; MRI, Healthy vs Tumor) — a <b>Custom CNN</b> built from
            scratch, and two transfer-learning models, <b>MobileNetV2</b> and <b>ResNet50</b>, both
            pretrained on ImageNet and fine-tuned for this task. All three were evaluated on a held-out
            test set using accuracy, precision, recall, F1, confusion matrices, and ROC/AUC curves.
            <b>ResNet50</b> came out on top and is the default model in this app.</p>

            <p><b>Tech stack:</b> TensorFlow / Keras for model training, OpenCV &amp; Pillow for image
            handling, scikit-learn for evaluation metrics, and Streamlit for this interactive web app.</p>

            <p><b>Why it matters:</b> this project demonstrates an end-to-end deep learning workflow —
            EDA, preprocessing, model training &amp; comparison, evaluation, and deployment — applied to a
            medical imaging classification problem.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer-box">
        ⚠️ <b>Disclaimer:</b> this is an educational / portfolio project. It is <b>not</b> a certified
        medical device and must not be used for real diagnosis or clinical decision-making. Always
        consult a qualified radiologist or physician for medical concerns.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: HOW IT WORKS
# =============================================================================
def render_how_it_works():
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">From the moment you upload a scan to the moment you see a '
        "result, here's exactly what happens under the hood.</div>",
        unsafe_allow_html=True,
    )

    steps = [
        (
            "Image Upload",
            "You upload a brain CT or MRI scan (JPG / PNG / JPEG) through the drag-and-drop "
            "uploader on the Home page.",
        ),
        (
            "Image Loading &amp; Validation",
            "The image is opened with <code>Pillow (PIL)</code> and converted to RGB — this drops "
            "any alpha channel and normalizes grayscale exports, which are common in CT/MRI files.",
        ),
        (
            "Scan-Type Detection (CT vs MRI)",
            "A rule-based heuristic estimates whether the scan is CT or MRI by analyzing "
            "<b>bright-pixel ratio</b> (CT's bright bone ring shows up as a spike in "
            "near-white pixels, since bone has very high signal on CT but is usually dark "
            "on MRI) and <b>background darkness ratio</b> (CT backgrounds tend to be a very "
            "uniform pure black). This is a heuristic, not a trained classifier — it's shown "
            "as a hint alongside the diagnosis, not a certainty.",
        ),
        (
            "Resizing &amp; Preprocessing",
            "The image is resized to <code>224×224</code> pixels (the input size every model was "
            "trained on) and converted to a float32 array. Unlike a classic ML pipeline, the raw "
            "0–255 pixel values are passed through unchanged — normalization happens inside the model.",
        ),
        (
            "Model-Specific Normalization",
            "Each architecture normalizes pixels differently, and this is baked directly into the "
            "model graph so the app doesn't need to know which one to apply: <b>Custom CNN</b> uses a "
            "<code>Rescaling(1/255)</code> layer, <b>MobileNetV2</b> uses its own "
            "<code>preprocess_input</code> ([-1, 1] scaling), and <b>ResNet50</b> uses its "
            "<code>preprocess_input</code> (channel-wise mean subtraction).",
        ),
        (
            "Deep Learning Inference",
            "The selected model (Custom CNN / MobileNetV2 / ResNet50) runs a forward pass. "
            "Convolutional layers automatically learn a hierarchy of features — edges &amp; gradients "
            "first, then textures &amp; shapes, then tumor-specific patterns — with no manual feature "
            "engineering required.",
        ),
        (
            "Postprocessing",
            "The model outputs a single sigmoid value between 0 and 1 (probability of "
            "<b>Tumor</b>). A threshold of <code>0.5</code> converts this into a Healthy/Tumor label, "
            "and the confidence score is the probability of whichever class was predicted.",
        ),
        (
            "Result Rendering",
            "The scan-type badge, diagnosis banner, a confidence gauge, and a Tumor-vs-Healthy "
            "probability bar chart are rendered back to you — all within a second or two.",
        ),
    ]

    html = '<div class="timeline">'
    for i, (title, desc) in enumerate(steps, start=1):
        html += f"""
        <div class="tl-item">
            <div class="tl-line"></div>
            <div class="tl-num">{i}</div>
            <div class="tl-content">
                <div class="tl-title">{title}</div>
                <div class="tl-desc">{desc}</div>
            </div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# PAGE: FEATURES
# =============================================================================
def render_features():
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">What this app can do, and what\'s going on underneath '
        "the images it analyzes.</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🚀 Project Features", "🖼️ Image Features"])

    with tab1:
        features = [
            (
                "🧠",
                "3 Deep Learning Models",
                "Choose between a Custom CNN, MobileNetV2, and ResNet50 — trained and benchmarked side by side.",
            ),
            (
                "🩻",
                "CT / MRI Detection",
                "A built-in heuristic estimates whether an uploaded scan is CT or MRI before diagnosis.",
            ),
            (
                "📊",
                "Confidence Gauge &amp; Charts",
                "Every prediction comes with a visual confidence gauge and a Tumor-vs-Healthy probability chart.",
            ),
            (
                "⚡",
                "Fast Inference",
                "Predictions typically return in under a couple of seconds per image.",
            ),
            (
                "🎯",
                "High Accuracy",
                "The best model (ResNet50) reaches 96.6% test accuracy and 0.993 AUC.",
            ),
            (
                "🔍",
                "Transparent Pipeline",
                "The How It Works page documents every preprocessing and inference step in detail.",
            ),
            (
                "🎨",
                "Modern Dark UI",
                "A clean, professional, responsive dark-themed interface.",
            ),
            (
                "🧪",
                "Model Comparison Ready",
                "Switch models from the dropdown to compare predictions across architectures.",
            ),
            (
                "📚",
                "Educational Focus",
                "Built as a portfolio / learning project demonstrating a full deep learning workflow.",
            ),
        ]
        grid_html = '<div class="feature-grid">'
        for icon, title, desc in features:
            grid_html += f"""
            <div class="feature-card">
                <div class="f-icon">{icon}</div>
                <div class="f-title">{title}</div>
                <div class="f-desc">{desc}</div>
            </div>
            """
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    with tab2:
        st.markdown(
            """
            <div class="imgfeat-group">
                <div class="ig-title">🧠 Deep-Learning Features (used by the deployed models)</div>
                <ul>
                    <li><b>Low-level features</b> — edges, gradients, and corners, learned by the first
                    convolutional layers.</li>
                    <li><b>Mid-level features</b> — textures and shapes, built up from combinations of
                    low-level patterns.</li>
                    <li><b>High-level features</b> — tumor-specific spatial patterns, learned in the
                    deepest layers just before the final classification.</li>
                    <li>No manual feature engineering is required — MobileNetV2 and ResNet50 also bring
                    ImageNet-pretrained feature extractors, fine-tuned for this task.</li>
                </ul>
            </div>

            <div class="imgfeat-group">
                <div class="ig-title">🩻 Heuristic Features (used for CT vs MRI detection)</div>
                <ul>
                    <li><b>Bright-pixel ratio</b> — fraction of near-white pixels; CT's bright bone
                    ring pushes this much higher than MRI, where bone is usually dark.</li>
                    <li><b>Background darkness ratio</b> — fraction of near-black pixels (air/background
                    outside the skull).</li>
                </ul>
            </div>

            <div class="imgfeat-group">
                <div class="ig-title">📐 Handcrafted Features (companion classical-ML notebook)</div>
                <ul>
                    <li><b>Intensity statistics</b> — mean, std, min, max, median pixel intensity.</li>
                    <li><b>Histogram shape</b> — entropy, skewness, kurtosis of the pixel distribution.</li>
                    <li><b>Edge density</b> — Sobel edge magnitude (mean &amp; std).</li>
                    <li><b>GLCM texture</b> — contrast, homogeneity, energy, correlation.</li>
                    <li><b>HOG descriptor</b> — Histogram of Oriented Gradients, summarized via
                    mean/std.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: CONTACT
# =============================================================================
def render_contact():
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Built by Hamza Munir. Feel free to check out the code or '
        "connect.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="contact-grid">
            <div class="contact-card">
                <div class="c-icon">🐙</div>
                <div class="c-title">GitHub</div>
                <div class="c-desc">Source code, notebooks, and other projects</div>
                <a class="c-link" href="{GITHUB_URL}" target="_blank">View Profile</a>
            </div>
            <div class="contact-card">
                <div class="c-icon">💼</div>
                <div class="c-title">LinkedIn</div>
                <div class="c-desc">Professional profile &amp; experience</div>
                <a class="c-link" href="{LINKEDIN_URL}" target="_blank">Connect</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Dispatch
# =============================================================================
if page == "home":
    render_home()
elif page == "about":
    render_about()
elif page == "how-it-works":
    render_how_it_works()
elif page == "features":
    render_features()
elif page == "contact":
    render_contact()

st.markdown("---")
st.caption(
    "⚠️ This tool is for educational/portfolio purposes only and is not a "
    "substitute for professional medical diagnosis."
)
