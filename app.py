import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io
import base64
import streamlit.components.v1 as components
import cv2
import tempfile

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FishVision | Creative Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("runs/detect/fish_project_v2/weights/best.pt")

model = load_model()

# ─────────────────────────────────────────────
# SPECIES DATABASE & FACT SHEETS
# ─────────────────────────────────────────────
FISH_METADATA = {
    'AngelFish': {
        'scientific': 'Pterophyllum scalare',
        'habitat': 'Slow Amazon basin streams & tropical reefs',
        'fact': 'They form lifelong monogamous pairs and cooperate to protect their eggs.'
    },
    'BlueTang': {
        'scientific': 'Paracanthurus hepatus',
        'habitat': 'Indo-Pacific coral reefs',
        'fact': 'They can sleep on their sides in reef crevices to hide from predators.'
    },
    'ButterflyFish': {
        'scientific': 'Chaetodontidae',
        'habitat': 'Shallow tropical coral reefs',
        'fact': 'They have fake eyespots near their tails to confuse and trick predators.'
    },
    'ClownFish': {
        'scientific': 'Amphiprioninae',
        'habitat': 'Warm Indian & Pacific Oceans',
        'fact': 'All clownfish are born male; the most dominant one turns female.'
    },
    'GoldFish': {
        'scientific': 'Carassius auratus',
        'habitat': 'Freshwater ponds & slow-moving rivers',
        'fact': 'They have a memory span of months and can recognize their owners\' faces.'
    },
    'Gourami': {
        'scientific': 'Osphronemidae',
        'habitat': 'Slow freshwater bodies across Asia',
        'fact': 'They use their long thread-like pelvic fins as feelers in murky waters.'
    },
    'MorishIdol': {
        'scientific': 'Zanclus cornutus',
        'habitat': 'Tropical Indo-Pacific reefs',
        'fact': 'Moors of Africa believed they brought good luck and happiness.'
    },
    'PlatyFish': {
        'scientific': 'Xiphophorus maculatus',
        'habitat': 'Central American rivers & estuaries',
        'fact': 'They give birth to live, free-swimming young rather than laying eggs.'
    },
    'RibbonedSweetlips': {
        'scientific': 'Plectorhinchus polytaenia',
        'habitat': 'Coral reefs of the Indo-West Pacific',
        'fact': 'Juveniles mimic toxic flatworms by swimming with a wild wiggly motion.'
    },
    'ThreeStripedDamselfish': {
        'scientific': 'Dascyllus aruanus',
        'habitat': 'Sheltered lagoons & shallow reefs',
        'fact': 'They live in coral branches and fiercely defend their home from intruders.'
    },
    'YellowCichlid': {
        'scientific': 'Labidochromis caeruleus',
        'habitat': 'Lake Malawi in East Africa',
        'fact': 'Females carry eggs and young in their mouths for up to three weeks.'
    },
    'YellowTang': {
        'scientific': 'Zebrasoma flavescens',
        'habitat': 'Shallow coral reefs of Hawaii & Pacific',
        'fact': 'They grow a razor-sharp white spine on their tail for self-defense.'
    },
    'ZebraFish': {
        'scientific': 'Danio rerio',
        'habitat': 'Freshwater streams of South Asia',
        'fact': 'They share 70% of their genes with humans and can fully regenerate organs.'
    }
}

def crop_and_encode(img, box):
    """
    Crops the image according to bounding box coordinates and returns base64 JPEG string.
    img: numpy array of the image.
    box: YOLO box coordinates [x1, y1, x2, y2].
    """
    try:
        h_orig, w_orig = img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)
        
        if x2 <= x1 or y2 <= y1:
            return ""
            
        crop_np = img[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop_np)
        
        # Resize crop if it's too large to save space in HTML
        target_height = 80
        h_crop, w_crop = crop_np.shape[:2]
        if h_crop > target_height:
            scale = target_height / h_crop
            new_w = int(w_crop * scale)
            pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""

def crop_bgr_and_encode(frame_bgr, box):
    """
    Crops the BGR video frame according to bounding box coordinates,
    converts it to RGB, and returns a base64 JPEG string.
    """
    try:
        h_orig, w_orig = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)
        
        if x2 <= x1 or y2 <= y1:
            return ""
            
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        target_height = 80
        h_crop, w_crop = crop_rgb.shape[:2]
        if h_crop > target_height:
            scale = target_height / h_crop
            new_w = int(w_crop * scale)
            pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""

if "run_count" not in st.session_state:
    st.session_state.run_count = 0

# ─────────────────────────────────────────────
# GLOBAL FRONTEND INJECTION (Liquid Theme Engine)
# ─────────────────────────────────────────────
# The user specifically requested using st.components.v1.html to inject global styles/scripts.
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;
    
    // Ensure we only inject once
    if (parentDoc.getElementById('liquid-theme-engine')) return;

    // ── 1. GLOBAL CSS ──
    const style = parentDoc.createElement('style');
    style.id = 'liquid-theme-engine';
    style.innerHTML = `
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');

        /* Liquid Theme Variables */
        :root, [data-theme="dark"] {
            --bg: #050505;
            --surface: rgba(20, 20, 22, 0.4);
            --text-main: #f0f0f0;
            --text-muted: #888888;
            --accent: #6600ff; /* Electric Indigo */
            --accent-glow: rgba(102, 0, 255, 0.2);
            --border: rgba(255, 255, 255, 0.05);
            --spotlight: rgba(255, 255, 255, 0.03);
        }
        
        [data-theme="light"] {
            --bg: #F9F9F9;
            --surface: rgba(255, 255, 255, 0.6);
            --text-main: #111111;
            --text-muted: #666666;
            --accent: #0044cc; /* Deep Cobalt */
            --accent-glow: rgba(0, 68, 204, 0.1);
            --border: rgba(0, 0, 0, 0.05);
            --spotlight: rgba(0, 0, 0, 0.02);
        }

        /* Streamlit Overrides */
        .stApp {
            background-color: var(--bg) !important;
            transition: background-color 0.8s cubic-bezier(0.22, 1, 0.36, 1);
            font-family: 'Instrument Sans', sans-serif !important;
            color: var(--text-main) !important;
            overflow-x: hidden !important;
            cursor: none !important; /* Hide default cursor for custom cursor */
        }
        
        /* Hide everything by default for custom cursor, except buttons */
        * { cursor: none !important; }
        
        header, footer, [data-testid="stSidebarNav"], [data-testid="stDecoration"] { display: none !important; }
        .block-container { max-width: 1200px !important; padding: 4rem 2rem 8rem !important; }

        /* Typography */
        h1, h2, h3, .serif-heading {
            font-family: 'Playfair Display', serif !important;
            letter-spacing: -0.04em !important;
            text-wrap: balance;
            color: var(--text-main) !important;
        }

        /* Custom Cursor */
        #custom-cursor {
            position: fixed; top: 0; left: 0; width: 12px; height: 12px;
            background: var(--accent); border-radius: 50%;
            pointer-events: none; z-index: 99999;
            transform: translate(-50%, -50%);
            transition: width 0.3s, height 0.3s, background 0.3s, mix-blend-mode 0.3s;
            will-change: transform;
        }
        #custom-cursor-follower {
            position: fixed; top: 0; left: 0; width: 40px; height: 40px;
            border: 1px solid var(--accent); border-radius: 50%;
            pointer-events: none; z-index: 99998;
            transform: translate(-50%, -50%);
            transition: width 0.3s, height 0.3s, transform 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            will-change: transform;
        }
        body.hovering-interactive #custom-cursor {
            width: 60px; height: 60px; background: var(--text-main); mix-blend-mode: difference;
        }
        body.hovering-interactive #custom-cursor-follower {
            width: 0; height: 0; opacity: 0;
        }

        /* Glass-Refraction Bento Grid */
        .bento-card {
            position: relative;
            background: var(--surface);
            backdrop-filter: blur(40px);
            -webkit-backdrop-filter: blur(40px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem;
            overflow: hidden;
            clip-path: polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%); /* Slightly irregular modern corners */
        }
        
        /* Spotlight Effect */
        .bento-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), var(--spotlight), transparent 40%);
            pointer-events: none; z-index: 0;
        }

        /* Cinematic Scanning Beam */
        .scan-container { position: relative; overflow: hidden; border-radius: 12px; }
        .scan-beam {
            position: absolute; top: 0; bottom: 0; left: -100%; width: 50%;
            background: linear-gradient(90deg, transparent, var(--accent-glow) 80%, var(--accent) 100%);
            box-shadow: 20px 0 40px var(--accent);
            animation: cinematicScan 2s cubic-bezier(0.65, 0, 0.35, 1) infinite;
            z-index: 10; pointer-events: none;
        }
        @keyframes cinematicScan {
            0% { left: -100%; }
            100% { left: 200%; }
        }

        /* Image Smooth Zoom */
        .smooth-zoom { transition: transform 0.8s cubic-bezier(0.22, 1, 0.36, 1); }
        .smooth-zoom:hover { transform: scale(1.03); }

        /* Floating Tooltips (Mocked as clean overlays for YOLO boxes) */
        .detection-tooltip {
            position: absolute; background: var(--surface); backdrop-filter: blur(10px);
            border: 1px solid var(--accent); padding: 4px 10px; border-radius: 20px;
            font-family: 'Instrument Sans', sans-serif; font-size: 11px; font-weight: 500;
            color: var(--text-main); transform: translate(-50%, -100%); margin-top: -8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3); pointer-events: none;
            display: flex; align-items: center; gap: 6px;
        }
        .detection-tooltip::after {
            content: ''; position: absolute; bottom: -4px; left: 50%; transform: translateX(-50%);
            width: 1px; height: 8px; background: var(--accent);
        }

        /* Vertical Timeline (Classification Log) */
        .timeline { position: relative; padding-left: 24px; margin-top: 1rem; }
        .timeline::before { content: ''; position: absolute; left: 4px; top: 0; bottom: 0; width: 1px; background: linear-gradient(to bottom, var(--accent), transparent); opacity: 0.3; }
        .timeline-item { position: relative; margin-bottom: 1.5rem; }
        .timeline-item::before {
            content: ''; position: absolute; left: -23px; top: 6px; width: 7px; height: 7px;
            background: var(--bg); border: 2px solid var(--accent); border-radius: 50%;
            box-shadow: 0 0 12px var(--accent-glow);
        }

        /* Entity Analysis Cards & Items */
        .entity-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.85rem;
            margin-top: 0.5rem;
        }
        .entity-item {
            display: flex;
            align-items: flex-start;
            gap: 1.25rem;
            padding: 1.1rem;
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid var(--border);
            border-radius: 16px;
            transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .entity-item:hover {
            transform: translateX(6px);
            border-color: var(--accent);
            background: linear-gradient(90deg, var(--accent-glow) 0%, rgba(255, 255, 255, 0.02) 100%);
        }
        .entity-thumb-container {
            width: 72px;
            height: 72px;
            flex-shrink: 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }
        .entity-thumb {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }
        .entity-item:hover .entity-thumb {
            transform: scale(1.1);
        }
        .entity-meta {
            flex: 1;
            min-width: 0;
        }
        .entity-row-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .entity-name {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-main);
            letter-spacing: -0.01em;
        }
        .entity-conf-badge {
            font-size: 0.65rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
            background: var(--accent-glow);
            border: 1px solid var(--accent);
            color: var(--text-main);
            white-space: nowrap;
            letter-spacing: 0.03em;
        }
        .entity-habitat {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin: 2px 0 6px 0;
            display: flex;
            align-items: center;
            gap: 4px;
            opacity: 0.9;
        }
        .entity-stats-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin: 6px 0;
            font-size: 0.74rem;
            color: var(--text-muted);
        }
        .entity-stat-item {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .entity-stat-item:not(:last-child)::after {
            content: "•";
            margin-left: 0.6rem;
            color: var(--border);
            font-weight: bold;
        }
        .entity-fact {
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.45;
            margin-top: 8px;
            padding: 6px 12px;
            background: var(--accent-glow);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
        }
        .entity-fact-label {
            font-weight: 600;
            color: var(--accent);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 6px;
        }

        /* Metrics summary grid for video */
        .metrics-summary-grid {
            display: grid;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-summary-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .metric-summary-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            background: var(--surface);
        }
        .metric-summary-val {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 0.2rem;
        }
        .metric-summary-lbl {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
        }


        
        /* System Header & Footer */
        .sys-header {
            position: fixed; top: 0; left: 0; right: 0; height: 60px;
            background: var(--surface); backdrop-filter: blur(20px); border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;
            z-index: 1000;
        }
        .header-brand { font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 600; color: var(--text-main); }
        .header-links { display: flex; gap: 2rem; font-family: 'Instrument Sans', sans-serif; font-size: 0.85rem; font-weight: 500; }
        .header-links a { color: var(--text-muted); text-decoration: none; transition: color 0.3s; }
        .header-links a:hover { color: var(--text-main); }
        
        .sys-footer {
            position: fixed; bottom: 0; left: 0; right: 0; height: 50px;
            background: var(--surface); backdrop-filter: blur(20px); border-top: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;
            font-family: 'Instrument Sans', sans-serif; font-size: 0.75rem; color: var(--text-muted);
            z-index: 1000;
        }
        
        /* Theme Toggle FAB */
        .theme-fab {
            position: fixed; bottom: 80px; right: 2rem; width: 48px; height: 48px;
            background: var(--surface); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem; color: var(--text-main); cursor: none; z-index: 1000;
            box-shadow: 0 12px 32px rgba(0,0,0,0.2); transition: transform 0.3s ease;
        }
        .theme-fab:hover { transform: scale(1.1); }

        /* Hide standard uploader UI & style */
        [data-testid="stFileUploader"] { background: transparent !important; }
        [data-testid="stFileUploader"] > div > section { background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 24px !important; backdrop-filter: blur(20px); transition: all 0.3s ease; }
        [data-testid="stFileUploader"] > div > section:hover { border-color: var(--accent) !important; }

        /* stButton Override */
        .stButton > button { background: transparent !important; border: 1px solid var(--border) !important; border-radius: 30px !important; color: var(--text-main) !important; transition: all 0.3s ease !important; font-family: 'Instrument Sans', sans-serif !important; }
        .stButton > button:hover { background: var(--text-main) !important; color: var(--bg) !important; }
    `;
    parentDoc.head.appendChild(style);

    // ── 2. CUSTOM CURSOR & SPOTLIGHT JS ──
    const cursor = parentDoc.createElement('div'); cursor.id = 'custom-cursor';
    const follower = parentDoc.createElement('div'); follower.id = 'custom-cursor-follower';
    parentDoc.body.appendChild(cursor);
    parentDoc.body.appendChild(follower);

    let mouseX = 0, mouseY = 0, cursorX = 0, cursorY = 0;
    parentDoc.addEventListener('mousemove', (e) => {
        mouseX = e.clientX; mouseY = e.clientY;
        cursor.style.left = mouseX + 'px';
        cursor.style.top = mouseY + 'px';
        
        // Spotlight effect for bento cards
        parentDoc.querySelectorAll('.bento-card').forEach(card => {
            const rect = card.getBoundingClientRect();
            card.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
            card.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
        });
    });

    // Spring follower animation
    function animate() {
        cursorX += (mouseX - cursorX) * 0.15;
        cursorY += (mouseY - cursorY) * 0.15;
        follower.style.transform = `translate(${cursorX}px, ${cursorY}px) translate(-50%, -50%)`;
        requestAnimationFrame(animate);
    }
    animate();

    // Interactive Hover State
    const interactiveSelectors = 'button, a, input, [data-testid="stFileUploader"] section, .bento-card, .theme-fab';
    parentDoc.addEventListener('mouseover', (e) => {
        if (e.target.closest(interactiveSelectors)) {
            parentDoc.body.classList.add('hovering-interactive');
        }
    });
    parentDoc.addEventListener('mouseout', (e) => {
        if (e.target.closest(interactiveSelectors)) {
            parentDoc.body.classList.remove('hovering-interactive');
        }
    });

    // ── 3. THEME TOGGLE ──
    const fab = parentDoc.createElement('div');
    fab.className = 'theme-fab interactive';
    fab.innerHTML = '◐';
    fab.onclick = () => {
        const current = parentDoc.documentElement.getAttribute('data-theme') || 'dark';
        parentDoc.documentElement.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
    };
    parentDoc.body.appendChild(fab);

})();
</script>
""", height=0, width=0)

st.markdown('''
<div class="sys-header">
  <div class="header-brand">AquaSight.</div>
  <div class="header-links">
    <a href="?page=home" target="_self">Home</a>
    <a href="?page=about" target="_self">About</a>
    <a href="?page=model" target="_self">Model</a>
  </div>
</div>
<div style="height: 60px;"></div>
''', unsafe_allow_html=True)

def render_footer(latency=0, load=0):
    st.markdown(f'''
    <div class="sys-footer">
      <div style="display: flex; gap: 1.5rem; color: var(--text-muted);">
        <span>© 2026 AquaSight Studio</span>
        <a href="#" style="color: var(--text-muted); text-decoration: none;">Privacy</a>
        <a href="#" style="color: var(--text-muted); text-decoration: none;">Terms</a>
        <a href="mailto:vikramsing3124k@gmail.com" style="color: var(--text-muted); text-decoration: none;">Contact</a>
      </div>
      <div style="display: flex; align-items: center; gap: 1rem;">
        <span style="display: flex; align-items: center; gap: 6px;">
          <div style="width: 6px; height: 6px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent);"></div>
          SYSTEM LIVE
        </span>
        <span>|</span>
        <span>LATENCY: {latency}MS</span>
        <span>|</span>
        <div style="display: flex; align-items: center; gap: 8px;">
          NEURAL LOAD
          <div style="width: 100px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden;">
            <div style="width: {min(100, load*10)}%; height: 100%; background: var(--accent); transition: width 1s ease;"></div>
          </div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

page = st.query_params.get("page", "home")

if page == "about":
    st.markdown("<h1 class='serif-heading' style='font-size: 3.5rem; margin-bottom: 0;'>About AquaSight.</h1>", unsafe_allow_html=True)
    st.markdown('''
<div class="bento-card" style="margin-top: 2rem; max-width: 800px;">
<h3 style="color: var(--accent); font-family: 'Playfair Display', serif;">The Author</h3>
<p style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem;">Developed by Vikram Singh. Passionate about leveraging computer vision and neural networks for real-world environmental and analytical applications.</p>

<h3 style="color: var(--accent); font-family: 'Playfair Display', serif; margin-top: 2.5rem;">The Dataset</h3>
<p style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem;">The model was trained on the <strong>Fish Detection v5</strong> dataset sourced from Roboflow Universe. It consists of <strong>8,242 meticulously annotated high-resolution images</strong> covering 13 distinct marine species (including AngelFish, ClownFish, BlueTang, and ZebraFish).</p>
<p style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem;">To ensure robust detection across varying water turbidities, the training pipeline applied advanced image augmentation techniques including random Gaussian blurring, exposure adjustments, and multi-directional rotations.</p>

<h3 style="color: var(--accent); font-family: 'Playfair Display', serif; margin-top: 2.5rem;">The Model</h3>
<p style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem;">Powered by the <strong>YOLOv8</strong> architecture, optimized for real-time edge inference. Following completion of the training cycle, the model achieved an impressive validation profile:</p>
<ul style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem; margin-top: 0.5rem; margin-left: 1rem;">
  <li><strong>mAP50 (Mean Average Precision):</strong> 89.8%</li>
  <li><strong>Precision:</strong> 83.7%</li>
  <li><strong>Recall:</strong> 86.7%</li>
</ul>
<p style="color: var(--text-muted); line-height: 1.6; font-size: 1.1rem; margin-top: 1rem;">This architecture ensures high accuracy classification while maintaining the necessary speed for live video stream analysis.</p>
</div>
''', unsafe_allow_html=True)
    render_footer()
    st.stop()

elif page == "model":
    st.markdown("<h1 class='serif-heading' style='font-size: 3.5rem; margin-bottom: 0;'>Model Architecture.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: var(--text-muted); max-width: 600px; margin-bottom: 2rem;'>Comprehensive training metrics and evaluation profiles generated during the YOLOv8 neural network compilation.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>Training Convergence</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/results.png", use_container_width=True)
    with col2:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>Normalized Confusion Matrix</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/confusion_matrix_normalized.png", use_container_width=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1], gap="large")
    with col3:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>F1-Confidence Curve</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/BoxF1_curve.png", use_container_width=True)
    with col4:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>Precision-Recall Curve</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/BoxPR_curve.png", use_container_width=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col5, col6 = st.columns([1, 1], gap="large")
    with col5:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>Precision-Confidence Curve</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/BoxP_curve.png", use_container_width=True)
    with col6:
        st.markdown("<h3 style='color: var(--accent); font-family: \"Playfair Display\", serif;'>Recall-Confidence Curve</h3>", unsafe_allow_html=True)
        st.image("runs/detect/fish_project_v2/BoxR_curve.png", use_container_width=True)
        
    render_footer()
    st.stop()

st.markdown("<h1 class='serif-heading' style='font-size: 3.5rem; margin-bottom: 0;'>AquaSight.</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: var(--text-muted); max-width: 500px; margin-bottom: 2rem;'>An intelligent, cinematic vision system designed for high-end marine analysis.</p>", unsafe_allow_html=True)

infer_ms = 0
num_boxes = 0


with st.expander("⚙️ Inference Controls"):
    c1, c2 = st.columns(2)
    with c1:
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.25, 0.05)
    with c2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        heatmap_mode = st.toggle("Enable Heatmap Mode", value=False)

# File Uploader
uploaded_file = st.file_uploader("Upload Media", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], label_visibility="collapsed")

# Top Metrics Row (Bento Grid)
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    is_video = file_ext in ['mp4', 'mov', 'avi']
    
    if is_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        scan_ph = st.empty()
        scan_ph.markdown("""
        <div class="bento-card scan-container" style="height: 400px; display: flex; align-items: center; justify-content: center;">
          <div class="scan-beam"></div>
          <div style="font-family: 'Instrument Sans'; color: var(--text-muted); font-size: 0.9rem; z-index: 20;">Initializing Video Stream Neural Pathways...</div>
        </div>
        """, unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(tfile.name)
        stop_btn = st.button("Stop Video Processing", key="stop_vid")
        
        st.markdown(f'''
        <div class="bento-card" style="padding: 1rem; margin-bottom: 24px;">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; padding-left: 0.5rem;">Live Detection Stream</div>
        </div>
        ''', unsafe_allow_html=True)
        col_vid, col_log = st.columns([1, 1], gap="small")
            
        with col_vid:
            stframe = st.empty()
        with col_log:
            timeline_ph = st.empty()
            csv_export_ph = st.empty()
        
        # Oceanic Census dynamic placeholder (below video stream and log)
        details_ph = st.empty()
        
        scan_ph.empty()
        
        global_classes = {}
        video_tracker = {}
        t0 = time.time()
        last_log_update = 0
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            res = results[0]
            plotted = res.plot()
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            
            needs_log_update = False
            frame_counts = {}
            for b in res.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                name = model.names[cls_id]
                
                # Global classification log updates
                if name not in global_classes or conf > global_classes[name]:
                    global_classes[name] = conf
                    needs_log_update = True
                num_boxes += 1
                
                # Track frame occurrence counts
                frame_counts[name] = frame_counts.get(name, 0) + 1
                
                # Track best crops, peak counts, etc.
                h_frame, w_frame = frame.shape[:2]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                coverage = ((x2 - x1) * (y2 - y1)) / (w_frame * h_frame) * 100
                
                if name not in video_tracker:
                    video_tracker[name] = {
                        'max_conf': conf,
                        'crop_b64': crop_bgr_and_encode(frame, b.xyxy[0]),
                        'peak_count': 1,
                        'total_detections': 1,
                        'coverage': coverage
                    }
                    needs_log_update = True
                else:
                    video_tracker[name]['total_detections'] += 1
                    if conf > video_tracker[name]['max_conf']:
                        video_tracker[name]['max_conf'] = conf
                        video_tracker[name]['coverage'] = coverage
                        new_crop = crop_bgr_and_encode(frame, b.xyxy[0])
                        if new_crop:
                            video_tracker[name]['crop_b64'] = new_crop
                        needs_log_update = True
            
            # Check for new peaks in simultaneous counts
            for sp, cnt in frame_counts.items():
                if sp in video_tracker:
                    if cnt > video_tracker[sp]['peak_count']:
                        video_tracker[sp]['peak_count'] = cnt
                        needs_log_update = True
                
            # Resize frame to prevent it from overwhelming the screen
            max_height = 450
            h, w = plotted_rgb.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                plotted_rgb = cv2.resize(plotted_rgb, (new_w, max_height))
                
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(plotted_rgb, cv2.COLOR_RGB2BGR))
            b64_img = base64.b64encode(buffer).decode()
            stframe.markdown(f'''
            <div class="bento-card" style="height: 450px; display: flex; justify-content: center; align-items: center; padding: 0; overflow: hidden; background: #000;">
                <img src="data:image/jpeg;base64,{b64_img}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 12px;" />
            </div>
            ''', unsafe_allow_html=True)
            
            current_time = time.time()
            if needs_log_update or (current_time - last_log_update > 0.5):
                classes_vid = list(global_classes.keys())
                confidences_vid = list(global_classes.values())
                
                timeline_html = '<div class="timeline">'
                for i, (name, conf) in enumerate(zip(classes_vid, confidences_vid)):
                    if heatmap_mode:
                        if conf >= 0.75: bar_color = "#16a34a"
                        elif conf >= 0.50: bar_color = "#ca8a04"
                        else: bar_color = "#dc2626"
                        shadow_col = f"{bar_color}80"
                    else:
                        bar_color = "var(--accent)"
                        shadow_col = "var(--accent-glow)"
                        
                    timeline_html += f'''
<div class="timeline-item">
<div style="font-size: 0.9rem; color: var(--text-main); font-weight: 500;">{name}</div>
<div style="font-size: 0.75rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px; margin-top: 4px;">
<span style="width: 100px;">CONFIDENCE: {conf:.0%}</span>
<div style="flex: 1; max-width: 120px; height: 3px; background: var(--surface); border-radius: 2px; overflow: hidden; border: 1px solid var(--border);">
<div style="height: 100%; width: {conf*100}%; background: {bar_color}; box-shadow: 0 0 8px {shadow_col}; transition: background 0.3s ease;"></div>
</div>
</div>
</div>
'''
                timeline_html += '</div>'
                if not classes_vid:
                    timeline_html = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 1rem 0;">Awaiting entities...</div>'

                # 1. Update live timeline
                timeline_ph.markdown(f'''
                <div class="bento-card" style="height: 450px; overflow-y: auto;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem;">Live Classification Log</div>
                {timeline_html}
                </div>
                ''', unsafe_allow_html=True)
                
                # 2. Update Live Oceanic Census Details Card
                unique_species_vid = len(video_tracker)
                if unique_species_vid == 0:
                    biodiversity_vid = "None"
                elif unique_species_vid <= 1:
                    biodiversity_vid = "Low"
                elif unique_species_vid <= 3:
                    biodiversity_vid = "Moderate"
                else:
                    biodiversity_vid = "Rich"
                    
                best_confs = [d['max_conf'] for d in video_tracker.values()]
                avg_conf_vid = sum(best_confs) / len(best_confs) if best_confs else 0
                
                video_entities_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; margin-top: 1rem;">'
                added_video_cards = 0
                for sp, data in video_tracker.items():
                    if data['max_conf'] < 0.8:
                        continue
                    added_video_cards += 1
                    meta = FISH_METADATA.get(sp, {'habitat': 'Oceanic reefs', 'fact': 'Marine ecosystem inhabitant.'})
                    img_src = f"data:image/jpeg;base64,{data['crop_b64']}" if data['crop_b64'] else ""
                    img_tag = f'<img src="{img_src}" class="entity-thumb" />' if img_src else '<div style="font-size:0.6rem; color:var(--text-muted); padding: 10px;">Capturing...</div>'
                    
                    video_entities_html += f'''
<div class="entity-item">
  <div class="entity-thumb-container">
    {img_tag}
  </div>
  <div class="entity-meta">
    <div class="entity-row-top">
      <span class="entity-name">{sp}</span>
      <span class="entity-conf-badge">MAX CONF: {data['max_conf']:.0%}</span>
    </div>
    <div class="entity-habitat">
      🌊 {meta.get('habitat', 'Oceanic reefs')}
    </div>
    <div class="entity-stats-row">
      <span class="entity-stat-item">📐 Max Area: {data.get('coverage', 0):.1f}%</span>
      <span class="entity-stat-item">👥 Peak: {data['peak_count']}</span>
      <span class="entity-stat-item">⏱️ Detections: {data['total_detections']}</span>
    </div>
    <div class="entity-fact"><span class="entity-fact-label">Fun Fact:</span>{meta['fact']}</div>
  </div>
</div>
'''
                video_entities_html += '</div>'
                if added_video_cards == 0:
                    video_entities_html = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 2rem 0; text-align: center;">Waiting for high-confidence entities (>80%)...</div>'
                
                details_ph.markdown(f'''
<div class="bento-card" style="margin-top: 24px;">
<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem;">✦ Live Oceanic Census & Entity Analysis</div>
<div class="metrics-summary-grid" style="grid-template-columns: repeat(4, 1fr);">
  <div class="metric-summary-card">
    <div class="metric-summary-val">{num_boxes}</div>
    <div class="metric-summary-lbl">Total Detections</div>
  </div>
  <div class="metric-summary-card">
    <div class="metric-summary-val">{unique_species_vid}</div>
    <div class="metric-summary-lbl">Unique Species</div>
  </div>
  <div class="metric-summary-card">
    <div class="metric-summary-val">{avg_conf_vid:.0%}</div>
    <div class="metric-summary-lbl">Avg Max Conf</div>
  </div>
  <div class="metric-summary-card">
    <div class="metric-summary-val">{biodiversity_vid}</div>
    <div class="metric-summary-lbl">Biodiversity</div>
  </div>
</div>
{video_entities_html}
</div>
''', unsafe_allow_html=True)

                last_log_update = current_time
                
                if len(classes_vid) > 0:
                    import pandas as pd
                    df = pd.DataFrame({
                        "ID": range(1, len(classes_vid)+1), 
                        "Species": classes_vid, 
                        "Max Confidence": [round(c, 4) for c in confidences_vid]
                    })
                    csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
                    csv_export_ph.markdown(f'''
<div style="margin-top: 16px; text-align: right;">
<a href="data:file/csv;base64,{csv_b64}" download="aquasight_export_video.csv" style="display:inline-flex; align-items:center; gap:8px; padding:10px 18px; background:var(--text-main); color:var(--bg); text-decoration:none; font-family:'Instrument Sans', sans-serif; font-size:0.75rem; font-weight:500; border-radius:30px; box-shadow:0 4px 12px rgba(0,0,0,0.15); transition:transform 0.2s ease;">
  ↓ Export Classification Matrix
</a>
</div>
''', unsafe_allow_html=True)
            
        infer_ms = int((time.time() - t0) * 1000)
        cap.release()
        render_footer(latency=infer_ms, load=num_boxes)
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    # Cinematic Scanning Trigger
    scan_ph = st.empty()
    scan_ph.markdown("""
    <div class="bento-card scan-container" style="height: 400px; display: flex; align-items: center; justify-content: center;">
      <div class="scan-beam"></div>
      <div style="font-family: 'Instrument Sans'; color: var(--text-muted); font-size: 0.9rem; z-index: 20;">Initializing Neural Pathways...</div>
    </div>
    """, unsafe_allow_html=True)
    
    t0 = time.time()
    results = model.predict(img_np, conf=conf_threshold)
    infer_ms = int((time.time() - t0) * 1000)
    result = results[0]
    plotted = result.plot()
    boxes = result.boxes
    num_boxes = len(boxes)
    st.session_state.run_count += 1
    
    scan_ph.empty()
    
    img_w, img_h = image.size
    
    confidences = []
    classes = []
    
    for b in boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        name = model.names[cls_id]
        confidences.append(conf)
        classes.append(name)
    
    # Base64 Original Image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64_orig = base64.b64encode(buf.getvalue()).decode()
    
    # Base64 Detection Image (Plotted YOLO)
    buf_det = io.BytesIO()
    Image.fromarray(plotted).save(buf_det, format="JPEG", quality=90)
    b64_det = base64.b64encode(buf_det.getvalue()).decode()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown(f'''
<div class="bento-card" style="padding: 1rem; margin-bottom: 24px;">
<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; padding-left: 0.5rem;">Original Frame</div>
<div style="position: relative; width: 100%; border-radius: 16px; overflow: hidden; background: #000;" class="smooth-zoom">
<img src="data:image/jpeg;base64,{b64_orig}" style="width: 100%; height: auto; display: block; opacity: 0.9;" />
</div>
</div>
''', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'''
<div class="bento-card" style="padding: 1rem; margin-bottom: 24px;">
<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; padding-left: 0.5rem;">Detection Frame</div>
<div style="position: relative; width: 100%; border-radius: 16px; overflow: hidden; background: #000;" class="smooth-zoom">
<img src="data:image/jpeg;base64,{b64_det}" style="width: 100%; height: auto; display: block; opacity: 0.9;" />
</div>
</div>
''', unsafe_allow_html=True)
        
    # ─────────────────────────────────────────────
    # ENTITY CENSUS PROCESSING (Image)
    # ─────────────────────────────────────────────
    entities_list = []
    species_counts = {}
    
    for i, b in enumerate(boxes):
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        name = model.names[cls_id]
        
        # Crop & base64 encode
        crop_b64 = crop_and_encode(img_np, b.xyxy[0])
        
        meta = FISH_METADATA.get(name, {
            'habitat': 'Oceanic reefs',
            'fact': 'Prevalent marine ecosystem species.'
        })
        
        # Calculate coverage
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        coverage = ((x2 - x1) * (y2 - y1)) / (img_w * img_h) * 100
        
        entities_list.append({
            'name': name,
            'fact': meta['fact'],
            'habitat': meta.get('habitat', 'Oceanic reefs'),
            'conf': conf,
            'b64': crop_b64,
            'coverage': coverage
        })
        
        species_counts[name] = species_counts.get(name, 0) + 1

    unique_species = len(species_counts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    # Species Distribution Badges (Modern Sleek Badges)
    badges_html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1.2rem;">'
    for sp, cnt in species_counts.items():
        badges_html += f'''
        <span style="font-size:0.75rem; background:var(--surface); border:1px solid var(--border); padding:4px 10px; border-radius:20px; color:var(--text-main); font-weight:500;">
          {sp} <span style="color:var(--accent); font-weight:600; margin-left:4px;">{cnt}</span>
        </span>
        '''
    badges_html += '</div>'
    if not species_counts:
        badges_html = ""
        
    # Minimalist entity items grid
    entity_items_html = '<div class="entity-grid">'
    added_image_cards = 0
    for ent in entities_list:
        if ent['conf'] < 0.8:
            continue
        added_image_cards += 1
        img_src = f"data:image/jpeg;base64,{ent['b64']}" if ent['b64'] else ""
        img_tag = f'<img src="{img_src}" class="entity-thumb" />' if img_src else '<div style="font-size:0.5rem; color:var(--text-muted);">No Thumb</div>'
        
        entity_items_html += f'''
<div class="entity-item">
  <div class="entity-thumb-container">
    {img_tag}
  </div>
  <div class="entity-meta">
    <div class="entity-row-top">
      <span class="entity-name">{ent['name']}</span>
      <span class="entity-conf-badge">CONF: {ent['conf']:.0%}</span>
    </div>
    <div class="entity-habitat">
      🌊 {ent['habitat']}
    </div>
    <div class="entity-stats-row">
      <span class="entity-stat-item">📐 Area: {ent['coverage']:.1f}%</span>
      <span class="entity-stat-item">🎯 Quality: {('High' if ent['conf'] >= 0.8 else 'Good' if ent['conf'] >= 0.5 else 'Fair')}</span>
    </div>
    <div class="entity-fact"><span class="entity-fact-label">Fun Fact:</span>{ent['fact']}</div>
  </div>
</div>
'''
    entity_items_html += '</div>'
    if added_image_cards == 0:
        entity_items_html = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 2rem 0; text-align: center;">No entities with confidence > 80% identified.</div>'

    # Second row layout: Left column = Details Card, Right column = Classification Log
    col_details, col_log = st.columns([1.2, 0.8], gap="large")
    
    with col_details:
        st.markdown(f'''
<div class="bento-card" style="height: 500px; overflow-y: auto;">
<div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1rem;">
  <div>
    <h4 style="margin: 0; font-family: 'Playfair Display', serif; font-size: 1.25rem;">Oceanic Census</h4>
    <span style="font-size: 0.75rem; color: var(--text-muted);">Species log and behavioral insights</span>
  </div>
  <div style="display: flex; gap: 1.5rem; text-align: right;">
    <div>
      <div style="font-size: 1.25rem; font-weight: 600; color: var(--accent); font-family: 'Playfair Display', serif;">{len(boxes)}</div>
      <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Total Detected</div>
    </div>
    <div>
      <div style="font-size: 1.25rem; font-weight: 600; color: var(--accent); font-family: 'Playfair Display', serif;">{unique_species}</div>
      <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Species</div>
    </div>
  </div>
</div>
{badges_html}
{entity_items_html}
</div>
''', unsafe_allow_html=True)
        
    with col_log:
        # Micro-Layout Vertical Timeline
        timeline_html = '<div class="timeline">'
        for i, (name, conf) in enumerate(zip(classes, confidences)):
            if heatmap_mode:
                if conf >= 0.75:
                    bar_color = "#16a34a" # Green for high confidence
                elif conf >= 0.50:
                    bar_color = "#ca8a04" # Yellow for medium
                else:
                    bar_color = "#dc2626" # Red for low
                shadow_col = f"{bar_color}80"
            else:
                bar_color = "var(--accent)"
                shadow_col = "var(--accent-glow)"
                
            timeline_html += f'''
<div class="timeline-item">
<div style="font-size: 0.9rem; color: var(--text-main); font-weight: 500;">{name}</div>
<div style="font-size: 0.75rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px; margin-top: 4px;">
<span style="width: 100px;">CONFIDENCE: {conf:.0%}</span>
<div style="flex: 1; max-width: 120px; height: 3px; background: var(--surface); border-radius: 2px; overflow: hidden; border: 1px solid var(--border);">
<div style="height: 100%; width: {conf*100}%; background: {bar_color}; box-shadow: 0 0 8px {shadow_col}; transition: background 0.3s ease;"></div>
</div>
</div>
</div>
'''
        timeline_html += '</div>'
        
        if not classes:
            timeline_html = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 1rem 0;">No significant entities localized.</div>'

        st.markdown(f'''
<div class="bento-card" style="height: 500px; overflow-y: auto;">
<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem;">Classification Log</div>
{timeline_html}
</div>
''', unsafe_allow_html=True)
        
        if len(boxes) > 0:
            import pandas as pd
            df = pd.DataFrame({
                "ID": range(1, len(boxes)+1), 
                "Species": classes, 
                "Confidence": [round(c, 4) for c in confidences]
            })
            csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
            st.markdown(f'''
<div style="margin-top: 16px; text-align: right;">
<a href="data:file/csv;base64,{csv_b64}" download="aquasight_export.csv" style="display:inline-flex; align-items:center; gap:8px; padding:10px 18px; background:var(--text-main); color:var(--bg); text-decoration:none; font-family:'Instrument Sans', sans-serif; font-size:0.75rem; font-weight:500; border-radius:30px; box-shadow:0 4px 12px rgba(0,0,0,0.15); transition:transform 0.2s ease;">
  ↓ Export Classification Matrix
</a>
</div>
''', unsafe_allow_html=True)
        
render_footer(latency=infer_ms, load=num_boxes)
