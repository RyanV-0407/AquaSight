import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io
import base64
import streamlit.components.v1 as components

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
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Top Metrics Row (Bento Grid)
if uploaded_file:
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
<div class="bento-card" style="height: auto;">
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
