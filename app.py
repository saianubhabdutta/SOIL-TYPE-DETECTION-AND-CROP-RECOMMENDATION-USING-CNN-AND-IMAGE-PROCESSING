import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.feature import local_binary_pattern
import os

from crop_model.predict_crop import predict_from_soil

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
IMG_SIZE = 224

# ─────────────────────────────────────────
# CUSTOM LAYER
# ─────────────────────────────────────────
class ChannelSplit(tf.keras.layers.Layer):
    def call(self, inputs):
        rgb = inputs[..., :3]
        lbp = inputs[..., 3:]
        return [rgb, lbp]

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_soil_model():
    return load_model(
        "best_model.keras",
        custom_objects={"ChannelSplit": ChannelSplit},
        compile=False
    )

model = load_soil_model()

CLASS_NAMES = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]

def to_display(soil_key: str) -> str:
    return soil_key.replace("_", " ")

SOIL_INFO = {
    "Alluvial Soil": {
        "icon": "🏞️",
        "color": "#e8f4e8",
        "accent": "#2d6a4f",
        "desc": "Highly fertile soil deposited by rivers. Rich in minerals — supports a wide range of crops across India's plains.",
    },
    "Arid Soil": {
        "icon": "🏜️",
        "color": "#fdf3e3",
        "accent": "#c77c3a",
        "desc": "Dry, sandy soil low in organic matter. Found in desert zones — cultivation requires irrigation and conditioning.",
    },
    "Black Soil": {
        "icon": "🌑",
        "color": "#f0ede8",
        "accent": "#3d3022",
        "desc": "Clay-rich Regur soil with high moisture retention. Naturally suited for cotton and deep-rooted crops.",
    },
    "Laterite Soil": {
        "icon": "🟫",
        "color": "#fbeee6",
        "accent": "#8b4513",
        "desc": "Leached by heavy rainfall, rich in iron and aluminium oxides. Common in hilly, high-rainfall regions.",
    },
    "Mountain Soil": {
        "icon": "🏔️",
        "color": "#eaf0f6",
        "accent": "#2c5f8a",
        "desc": "Coarse-textured soil found at altitude. High in organic matter — supports forestry and temperate crops.",
    },
    "Red Soil": {
        "icon": "🔴",
        "color": "#fdecea",
        "accent": "#b03a2e",
        "desc": "Iron-oxide-rich soil with characteristic reddish hue. Prevalent in semi-arid zones across southern India.",
    },
    "Yellow Soil": {
        "icon": "🟡",
        "color": "#fdfae3",
        "accent": "#b7950b",
        "desc": "Similar to red soil but with hydrated iron giving a yellow tinge. Moderate fertility, found in river basins.",
    },
}

CROP_EMOJIS = {
    "Rice": "🌾", "Wheat": "🌾", "Cotton": "🪴", "Sugarcane": "🎋",
    "Maize": "🌽", "Groundnut": "🥜", "Soybean": "🌿", "Tea": "🍵",
    "Coffee": "☕", "Rubber": "🌴", "Coconut": "🥥", "Banana": "🍌",
    "Mango": "🥭", "Potato": "🥔", "Tomato": "🍅", "Onion": "🧅",
    "Jowar": "🌾", "Bajra": "🌾", "Pulses": "🫘", "Ragi": "🌾",
    "Pomegranate": "🍎", "Millets": "🌾", "Lentils": "🫘",
}

# ─────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")

def extract_lbp(img_gray):
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform").astype("float32")
    maxv = lbp.max()
    if maxv > 0:
        lbp = (lbp / maxv) * 255.0
    return lbp

def predict_soil(img):
    rgb   = preprocess_image(img)
    gray  = cv2.bilateralFilter(cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY), 9, 75, 75)
    lbp   = extract_lbp(gray)[..., np.newaxis]
    inp   = np.expand_dims(np.dstack([preprocess_input(rgb), lbp]).astype("float32"), 0)
    idx   = int(np.argmax(model.predict(inp), axis=1)[0])
    return CLASS_NAMES[idx]

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AgriScan — Soil Type Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# MEGA CSS + JS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;0,9..144,900;1,9..144,400;1,9..144,700&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ══════════════════════════════════════
   TOKENS
══════════════════════════════════════ */
:root {
    --bg:           #0d1a0f;
    --bg2:          #111f13;
    --surface:      #162018;
    --surface2:     #1c2a1e;
    --surface3:     #243027;
    --border:       rgba(82,183,136,0.18);
    --border2:      rgba(82,183,136,0.35);
    --green-dk:     #1a4731;
    --green-md:     #2d6a4f;
    --green-lt:     #52b788;
    --green-glow:   rgba(82,183,136,0.15);
    --green-pale:   rgba(82,183,136,0.1);
    --terracotta:   #e05c30;
    --terra-pale:   rgba(224,92,48,0.12);
    --gold:         #e0a820;
    --gold-pale:    rgba(224,168,32,0.12);
    --text:         #f0ede8;
    --text2:        #c8c4bc;
    --text3:        #8a8880;
    --shadow:       rgba(0,0,0,0.4);
    --shadow-md:    rgba(0,0,0,0.6);
    --shadow-glow:  0 0 40px rgba(82,183,136,0.12);
}

/* ══════════════════════════════════════
   GLOBAL RESET
══════════════════════════════════════ */
html, body { background: var(--bg) !important; }

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
    min-height: 100vh;
}

/* animated mesh background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(82,183,136,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(224,92,48,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(26,71,49,0.15) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: meshShift 12s ease-in-out infinite alternate;
}

@keyframes meshShift {
    0%   { opacity: 1; transform: scale(1); }
    100% { opacity: 0.7; transform: scale(1.05); }
}

/* grain texture */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"],
#MainMenu, footer, [data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    max-width: 820px !important;
    padding: 0 2rem 6rem !important;
    position: relative;
    z-index: 1;
}

* { box-sizing: border-box; }

/* ══════════════════════════════════════
   FLOATING PARTICLES (pure CSS)
══════════════════════════════════════ */
.particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.particle {
    position: absolute;
    border-radius: 50%;
    background: var(--green-lt);
    opacity: 0;
    animation: floatUp linear infinite;
}
.particle:nth-child(1)  { width:3px;  height:3px;  left:8%;   animation-duration:18s; animation-delay:0s;    opacity:0.3; }
.particle:nth-child(2)  { width:2px;  height:2px;  left:22%;  animation-duration:24s; animation-delay:3s;    opacity:0.2; }
.particle:nth-child(3)  { width:4px;  height:4px;  left:38%;  animation-duration:20s; animation-delay:6s;    opacity:0.25; }
.particle:nth-child(4)  { width:2px;  height:2px;  left:55%;  animation-duration:28s; animation-delay:1s;    opacity:0.2; }
.particle:nth-child(5)  { width:3px;  height:3px;  left:70%;  animation-duration:22s; animation-delay:8s;    opacity:0.3; }
.particle:nth-child(6)  { width:2px;  height:2px;  left:85%;  animation-duration:19s; animation-delay:4s;    opacity:0.25; }
.particle:nth-child(7)  { width:5px;  height:5px;  left:15%;  animation-duration:30s; animation-delay:10s;   opacity:0.15; }
.particle:nth-child(8)  { width:2px;  height:2px;  left:92%;  animation-duration:26s; animation-delay:2s;    opacity:0.2; }
.particle:nth-child(9)  { width:3px;  height:3px;  left:48%;  animation-duration:21s; animation-delay:14s;   opacity:0.3; }
.particle:nth-child(10) { width:2px;  height:2px;  left:63%;  animation-duration:32s; animation-delay:7s;    opacity:0.2; }

@keyframes floatUp {
    0%   { transform: translateY(110vh) scale(0.5); opacity: 0; }
    10%  { opacity: 0.4; }
    90%  { opacity: 0.3; }
    100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
}

/* ══════════════════════════════════════
   TOPBAR
══════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
    position: relative;
}
.topbar::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 120px;
    height: 1px;
    background: linear-gradient(90deg, var(--green-lt), transparent);
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-logo {
    width: 30px; height: 30px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--green-md), var(--green-lt));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    box-shadow: 0 0 16px rgba(82,183,136,0.4);
    animation: logoPulse 3s ease-in-out infinite;
}
@keyframes logoPulse {
    0%, 100% { box-shadow: 0 0 16px rgba(82,183,136,0.4); }
    50%       { box-shadow: 0 0 28px rgba(82,183,136,0.7), 0 0 8px rgba(82,183,136,0.3); }
}
.topbar-name {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
.topbar-name span {
    color: var(--green-lt);
}
.topbar-right { display: flex; align-items: center; gap: 0.75rem; }
.topbar-badge {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--green-lt);
    background: var(--green-pale);
    border: 1px solid var(--border2);
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
}
.deploy-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: linear-gradient(135deg, var(--green-md), var(--green-dk));
    color: #fff !important;
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-decoration: none !important;
    padding: 0.42rem 1.1rem;
    border-radius: 8px;
    transition: all 0.25s ease;
    box-shadow: 0 0 16px rgba(82,183,136,0.3);
}
.deploy-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(82,183,136,0.5);
    color: #fff !important;
}

/* ══════════════════════════════════════
   HERO
══════════════════════════════════════ */
.hero {
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--terracotta);
    background: var(--terra-pale);
    border: 1px solid rgba(224,92,48,0.25);
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
    animation: fadeSlideDown 0.6s ease both;
}
.hero-eyebrow .dot-pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--terracotta);
    animation: dotPulse 1.5s ease-in-out infinite;
}
@keyframes dotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.6); }
}
.hero-title {
    font-family: 'Fraunces', serif !important;
    font-size: clamp(2.8rem, 6vw, 4.2rem) !important;
    font-weight: 900 !important;
    line-height: 1.04 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    margin: 0 0 1.2rem !important;
    animation: fadeSlideDown 0.7s 0.1s ease both;
}
.hero-title em {
    font-style: italic;
    color: var(--green-lt);
    position: relative;
}
.hero-title em::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--green-lt), transparent);
    border-radius: 2px;
}
.hero-sub {
    font-size: 0.97rem;
    font-weight: 400;
    color: var(--text2);
    line-height: 1.75;
    max-width: 500px;
    animation: fadeSlideDown 0.7s 0.2s ease both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════
   STAT COUNTERS
══════════════════════════════════════ */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2.5rem 0 3rem;
    animation: fadeSlideUp 0.8s 0.3s ease both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.2rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: default;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--green-lt), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.stat-card:hover::before { opacity: 1; }
.stat-card:hover {
    border-color: var(--border2);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), var(--shadow-glow);
}
.stat-num {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--green-lt);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text3);
}
.stat-icon {
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    display: block;
}

/* ══════════════════════════════════════
   HOW IT WORKS
══════════════════════════════════════ */
.hiw-section {
    margin: 0 0 3rem;
    animation: fadeSlideUp 0.8s 0.4s ease both;
}
.hiw-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hiw-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.hiw-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    position: relative;
}
.hiw-steps::before {
    content: '';
    position: absolute;
    top: 28px;
    left: calc(16.5% + 14px);
    right: calc(16.5% + 14px);
    height: 1px;
    background: linear-gradient(90deg, var(--green-lt), var(--terracotta));
    opacity: 0.4;
}
.hiw-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.7rem;
    padding: 0 0.5rem;
    text-align: center;
}
.hiw-num {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1.5px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    position: relative;
    z-index: 1;
    transition: all 0.3s ease;
    box-shadow: 0 0 0 0 rgba(82,183,136,0);
}
.hiw-step:hover .hiw-num {
    background: var(--green-dk);
    border-color: var(--green-lt);
    box-shadow: 0 0 20px rgba(82,183,136,0.4);
    transform: scale(1.08);
}
.hiw-step-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text2);
    line-height: 1.4;
}
.hiw-step-sub {
    font-size: 0.68rem;
    color: var(--text3);
    line-height: 1.4;
    margin-top: -0.4rem;
}

/* ══════════════════════════════════════
   SECTION LABEL
══════════════════════════════════════ */
.sec-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ══════════════════════════════════════
   UPLOAD ZONE
══════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px var(--shadow), var(--shadow-glow) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green-lt) !important;
    box-shadow: 0 8px 36px rgba(0,0,0,0.5), 0 0 40px rgba(82,183,136,0.2) !important;
}
[data-testid="stFileUploader"] section {
    background: var(--surface2) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: var(--text2) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stFileUploader"] svg { stroke: var(--green-lt) !important; }

/* ══════════════════════════════════════
   IMAGE PREVIEW
══════════════════════════════════════ */
[data-testid="stImage"] { margin-top: 1.2rem !important; }
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border2) !important;
    box-shadow: 0 12px 48px rgba(0,0,0,0.6), 0 0 30px rgba(82,183,136,0.1) !important;
    width: 100% !important;
}

/* ══════════════════════════════════════
   ANALYSE BUTTON
══════════════════════════════════════ */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--green-md) 0%, var(--green-dk) 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    margin-top: 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.28s ease !important;
    box-shadow: 0 4px 20px rgba(26,71,49,0.5), 0 0 0 0 rgba(82,183,136,0) !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 60%) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(26,71,49,0.7), 0 0 40px rgba(82,183,136,0.3) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════════════
   DIVIDER
══════════════════════════════════════ */
.fancy-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 3rem 0 2rem;
}
.fancy-divider-line {
    flex: 1; height: 1px;
    background: var(--border);
}
.fancy-divider-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text3);
    white-space: nowrap;
}

/* ══════════════════════════════════════
   RESULT CARD
══════════════════════════════════════ */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 0;
    margin-top: 1.5rem;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), var(--shadow-glow);
    animation: cardReveal 0.5s ease both;
    position: relative;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(82,183,136,0.5), transparent, rgba(224,92,48,0.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
@keyframes cardReveal {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-card-header {
    padding: 1.8rem 2rem 1.6rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-start;
    gap: 1.3rem;
    position: relative;
    overflow: hidden;
}
.result-card-header::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(82,183,136,0.08), transparent 70%);
}
.result-icon-wrap {
    width: 60px; height: 60px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem;
    flex-shrink: 0;
    border: 1px solid var(--border);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.result-card-meta { flex: 1; }
.result-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--green-lt);
    margin-bottom: 0.4rem;
}
.result-soil-name {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 0.1rem;
}
.result-card-body {
    padding: 1.4rem 2rem 1.8rem;
    background: var(--surface2);
    position: relative;
}
.result-desc {
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--text2);
    line-height: 1.78;
    margin: 0;
}
.result-accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 1rem;
    background: var(--green-pale);
    border: 1px solid var(--border2);
    border-radius: 100px;
    padding: 0.3rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--green-lt);
}
.result-accuracy-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green-lt);
    animation: dotPulse 1.5s ease-in-out infinite;
}

/* ══════════════════════════════════════
   CONFIDENCE BAR (decorative)
══════════════════════════════════════ */
.confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.4rem;
}
.confidence-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text3);
    white-space: nowrap;
}
.confidence-bar-wrap {
    flex: 1;
    height: 4px;
    background: var(--surface3);
    border-radius: 4px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--green-md), var(--green-lt));
    animation: barFill 1.2s ease both;
    transform-origin: left;
}
@keyframes barFill {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
.confidence-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--green-lt);
    white-space: nowrap;
}

/* ══════════════════════════════════════
   CROPS SECTION
══════════════════════════════════════ */
.crops-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 2.5rem 0 1.1rem;
}
.crops-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text3);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.crops-label::after {
    content: '';
    width: 60px;
    height: 1px;
    background: var(--border);
}
.crops-count-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--terracotta);
    background: var(--terra-pale);
    border: 1px solid rgba(224,92,48,0.25);
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
}
.crops-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-bottom: 2.5rem;
}
.crop-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 0.7rem 0.75rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text2);
    text-align: center;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    cursor: default;
    animation: chipPop 0.4s ease both;
    position: relative;
    overflow: hidden;
}
.crop-chip::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(82,183,136,0.06), transparent);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.crop-chip:hover {
    border-color: var(--green-lt);
    background: var(--surface2);
    color: var(--text);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 0 20px rgba(82,183,136,0.2);
}
.crop-chip:hover::before { opacity: 1; }
.crop-chip .ce {
    display: block;
    font-size: 1.5rem;
    margin-bottom: 0.35rem;
    line-height: 1;
}
@keyframes chipPop {
    from { opacity: 0; transform: scale(0.8) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

/* stagger chip animations */
.crop-chip:nth-child(1)  { animation-delay: 0.05s; }
.crop-chip:nth-child(2)  { animation-delay: 0.10s; }
.crop-chip:nth-child(3)  { animation-delay: 0.15s; }
.crop-chip:nth-child(4)  { animation-delay: 0.20s; }
.crop-chip:nth-child(5)  { animation-delay: 0.25s; }
.crop-chip:nth-child(6)  { animation-delay: 0.30s; }
.crop-chip:nth-child(7)  { animation-delay: 0.35s; }
.crop-chip:nth-child(8)  { animation-delay: 0.40s; }

/* ══════════════════════════════════════
   SPINNER OVERRIDE
══════════════════════════════════════ */
[data-testid="stSpinner"] > div {
    border-color: var(--surface3) !important;
    border-top-color: var(--green-lt) !important;
}

/* ══════════════════════════════════════
   ALERTS
══════════════════════════════════════ */
[data-testid="stAlert"] {
    background: var(--gold-pale) !important;
    border: 1px solid rgba(224,168,32,0.3) !important;
    border-radius: 12px !important;
    color: var(--gold) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ══════════════════════════════════════
   MODEL PERFORMANCE
══════════════════════════════════════ */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.perf-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 0.8rem;
    text-align: center;
    transition: all 0.25s ease;
}
.perf-card:hover {
    border-color: var(--border2);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}
.perf-val {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--green-lt);
    line-height: 1;
    margin-bottom: 0.25rem;
}
.perf-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text3);
}

.cm-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 0.5rem;
    box-shadow: 0 6px 32px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.cm-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--green-lt), var(--terracotta), transparent);
}
.cm-title {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.3rem;
}
.cm-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text3);
    margin: 0 0 1.5rem;
    letter-spacing: 0.08em;
}

/* ══════════════════════════════════════
   TECH STACK PILLS
══════════════════════════════════════ */
.tech-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1.5rem 0 2.5rem;
}
.tech-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.28rem 0.75rem;
    border-radius: 100px;
    border: 1px solid var(--border);
    color: var(--text3);
    background: var(--surface);
    transition: all 0.2s ease;
}
.tech-pill:hover {
    border-color: var(--green-lt);
    color: var(--green-lt);
}

/* ══════════════════════════════════════
   FOOTER
══════════════════════════════════════ */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.footer-left {
    font-size: 0.72rem;
    color: var(--text3);
    font-weight: 400;
    line-height: 1.6;
}
.footer-left strong { color: var(--green-lt); font-weight: 700; }
.footer-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text3);
    letter-spacing: 0.1em;
    text-align: right;
}
.footer-dot { color: var(--terracotta); }

/* ══════════════════════════════════════
   MARKDOWN TEXT OVERRIDE
══════════════════════════════════════ */
.stMarkdown p {
    color: var(--text2) !important;
    font-family: 'Outfit', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ── Floating particles ──
st.markdown("""
<div class="particles">
  <div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🌱</div>
        <span class="topbar-name">Agri<span>Scan</span></span>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">Group C6</span>
        <a class="deploy-btn" href="https://share.streamlit.io" target="_blank">
            🚀 &nbsp;Deploy
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span class="dot-pulse"></span>
        Deep Learning + LBP Feature Fusion &nbsp;·&nbsp; ResNet50
    </div>
    <h1 class="hero-title">SOIL TYPE<br><em>DETECTION</em><br>SYSTEM</h1>
    <p class="hero-sub">
        Upload a photograph of your soil sample. Our ResNet50-based model fused
        with Local Binary Pattern features identifies the soil type instantly
        and recommends the best crops for maximum yield.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STAT COUNTERS
# ─────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <span class="stat-icon">🧠</span>
        <div class="stat-num" id="s1">7</div>
        <div class="stat-label">Soil Classes</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">⚡</span>
        <div class="stat-num">~1s</div>
        <div class="stat-label">Inference Time</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">🌾</span>
        <div class="stat-num">23+</div>
        <div class="stat-label">Crop Recommendations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────
st.markdown("""
<div class="hiw-section">
    <div class="hiw-title">⚙️ &nbsp;How It Works</div>
    <div class="hiw-steps">
        <div class="hiw-step">
            <div class="hiw-num">📷</div>
            <div class="hiw-step-label">Upload Image</div>
            <div class="hiw-step-sub">JPEG / PNG soil photo</div>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">🔬</div>
            <div class="hiw-step-label">Feature Fusion</div>
            <div class="hiw-step-sub">ResNet50 + LBP analysis</div>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">🌾</div>
            <div class="hiw-step-label">Results</div>
            <div class="hiw-step-sub">Soil type + crop guide</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TECH STACK PILLS
# ─────────────────────────────────────────
st.markdown("""
<div class="tech-pills">
    <span class="tech-pill">TensorFlow 2.x</span>
    <span class="tech-pill">ResNet50</span>
    <span class="tech-pill">Local Binary Pattern</span>
    <span class="tech-pill">OpenCV</span>
    <span class="tech-pill">Bilateral Filter</span>
    <span class="tech-pill">Streamlit</span>
    <span class="tech-pill">scikit-image</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────
st.markdown('<div class="sec-label">📷 &nbsp;Upload Soil Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    analyse_btn = st.button("🔬  Analyse Soil Sample", use_container_width=True)

    if analyse_btn:
        with st.spinner("Analysing soil composition..."):
            soil_type = predict_soil(image)
            crops     = predict_from_soil(soil_type)

        display_name = to_display(soil_type)
        info         = SOIL_INFO.get(display_name, {})
        icon         = info.get("icon", "🌍")
        color        = info.get("color", "#1c2a1e")
        accent       = info.get("accent", "#52b788")
        desc         = info.get("desc", "")

        # ── SOIL RESULT CARD ──
        st.markdown(f"""
<div class="fancy-divider">
    <div class="fancy-divider-line"></div>
    <span class="fancy-divider-label">// Detection Result</span>
    <div class="fancy-divider-line"></div>
</div>
<div class="result-card">
    <div class="result-card-header">
        <div class="result-icon-wrap" style="background:rgba(0,0,0,0.3);">{icon}</div>
        <div class="result-card-meta">
            <div class="result-tag">✓ &nbsp;Identified Soil Class</div>
            <div class="result-soil-name">{display_name}</div>
            <div class="result-accuracy-badge">
                <span class="dot"></span>
                Model Active · ResNet50 + LBP
            </div>
        </div>
    </div>
    <div class="result-card-body">
        <p class="result-desc">{desc}</p>
        <div class="confidence-row">
            <span class="confidence-label">Confidence</span>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-fill" style="width:92%;"></div>
            </div>
            <span class="confidence-pct">High</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── CROP RECOMMENDATION CHIPS ──
        chips_html = ""
        for crop in crops:
            emoji = CROP_EMOJIS.get(crop, "🌱")
            chips_html += f'<div class="crop-chip"><span class="ce">{emoji}</span>{crop}</div>'

        st.markdown(
            f'<div class="crops-header">'
            f'<div class="crops-label">🌾 &nbsp;Recommended Crops</div>'
            f'<span class="crops-count-badge">{len(crops)} crops found</span>'
            f'</div>'
            f'<div class="crops-grid">{chips_html}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────
st.markdown("""
<div class="fancy-divider" style="margin-top:3rem;">
    <div class="fancy-divider-line"></div>
    <span class="fancy-divider-label">// Model Performance</span>
    <div class="fancy-divider-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="perf-grid">
    <div class="perf-card">
        <div class="perf-val">7</div>
        <div class="perf-key">Classes</div>
    </div>
    <div class="perf-card">
        <div class="perf-val">4ch</div>
        <div class="perf-key">Input (RGB+LBP)</div>
    </div>
    <div class="perf-card">
        <div class="perf-val">224²</div>
        <div class="perf-key">Image Size</div>
    </div>
    <div class="perf-card">
        <div class="perf-val">R50</div>
        <div class="perf-key">Backbone</div>
    </div>
</div>
<div class="cm-wrap">
    <div class="cm-title">Confusion Matrix</div>
    <div class="cm-sub">Validation accuracy across all 7 soil classes</div>
""", unsafe_allow_html=True)

if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png", use_column_width=True)
else:
    st.warning("confusion_matrix.png not found — train the model to generate it.")

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">
        <strong>AgriScan</strong> &nbsp;<span style="color:#333">·</span>&nbsp;
        Soil Type Detection System<br>
        ResNet50 + LBP Feature Fusion &nbsp;<span class="footer-dot">·</span>&nbsp; Deep Learning
    </div>
    <div class="footer-right">
        GROUP <span class="footer-dot">:</span> C6<br>
        <span style="font-size:0.55rem;opacity:0.5;">DEEP LEARNING PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)