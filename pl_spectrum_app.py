import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import io

# ─────────────────────────────────────────
# Page config — must be FIRST st call
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PL Spectrum Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# Global CSS — dark lab aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #0d0f14;
  --surface:   #13161e;
  --border:    #1f2433;
  --accent:    #7eeaff;
  --accent2:   #ff7eb3;
  --text:      #e2e8f4;
  --muted:     #6b7694;
  --r:         #ff5b5b;
  --g:         #5bff9a;
  --b:         #5bb4ff;
  --sum:       #ffd35b;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Space Mono', monospace !important;
  letter-spacing: -0.5px;
}

/* ── Hero header ── */
.hero {
  padding: 2rem 0 1.5rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.hero-title {
  font-family: 'Space Mono', monospace;
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -1px;
  text-shadow: 0 0 32px rgba(126,234,255,0.35);
  margin: 0;
}
.hero-sub {
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 0.25rem;
  font-family: 'Space Mono', monospace;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}
.card-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--muted);
  margin-bottom: 1rem;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
.metric-tile {
  flex: 1;
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
  border: 1px solid;
}
.metric-tile.r { background: rgba(255,91,91,0.08); border-color: rgba(255,91,91,0.3); }
.metric-tile.g { background: rgba(91,255,154,0.08); border-color: rgba(91,255,154,0.3); }
.metric-tile.b { background: rgba(91,180,255,0.08); border-color: rgba(91,180,255,0.3); }
.metric-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 0.4rem;
}
.metric-tile.r .metric-label { color: var(--r); }
.metric-tile.g .metric-label { color: var(--g); }
.metric-tile.b .metric-label { color: var(--b); }
.metric-value {
  font-family: 'Space Mono', monospace;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
}

/* ── Color swatch ── */
.swatch-wrap { display:flex; align-items:center; gap:1rem; margin-top:0.75rem; }
.swatch {
  width: 56px; height: 56px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.12);
  flex-shrink: 0;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.swatch-label { font-size: 0.8rem; color: var(--muted); font-family: 'Space Mono', monospace; }

/* ── Source toggle ── */
.stRadio > div { gap: 0.5rem !important; }
.stRadio label {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.5rem 1.25rem !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  cursor: pointer !important;
  transition: border-color 0.2s !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.82rem !important;
}
.stNumberInput label, .stCheckbox label, .stRadio label span {
  color: var(--text) !important;
  font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
  letter-spacing: 1px !important;
  padding: 0.6rem 1.5rem !important;
  width: 100% !important;
  transition: opacity 0.2s, box-shadow 0.2s !important;
  box-shadow: 0 0 20px rgba(126,234,255,0.2) !important;
}
.stButton > button:hover {
  opacity: 0.85 !important;
  box-shadow: 0 0 32px rgba(126,234,255,0.4) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  width: 100% !important;
}
.stDownloadButton > button:hover {
  background: rgba(126,234,255,0.08) !important;
}

/* ── Sidebar section labels ── */
.sidebar-section {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--muted);
  padding: 0.5rem 0 0.25rem 0;
  border-top: 1px solid var(--border);
  margin-top: 1rem;
}

/* ── Parameter row labels ── */
.param-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  display: inline-block;
  margin-bottom: 0.5rem;
  font-weight: 700;
}
.param-r { background: rgba(255,91,91,0.15); color: var(--r); }
.param-g { background: rgba(91,255,154,0.15); color: var(--g); }
.param-b { background: rgba(91,180,255,0.15); color: var(--b); }

/* ── Info / warning boxes ── */
.stInfo, .stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Image ── */
[data-testid="stImage"] img {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
}
[data-testid="stFileUploaderDropzone"] { border-radius: 10px !important; }

/* ── Camera ── */
[data-testid="stCameraInput"] { border-radius: 10px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Core model
# ─────────────────────────────────────────
def gaussian(wl, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-((wl - center) ** 2) / (2 * sigma ** 2))


def rgb_to_pl(R, G, B,
              wl=None,
              normalize=True,
              center_r=600, fwhm_r=150,
              center_g=500, fwhm_g=80,
              center_b=430, fwhm_b=60):
    if wl is None:
        wl = np.arange(360, 800, 1)
    r_n, g_n, b_n = R / 255.0, G / 255.0, B / 255.0
    r_eq = r_n * gaussian(wl, center_r, fwhm_r)
    g_eq = g_n * gaussian(wl, center_g, fwhm_g)
    b_eq = b_n * gaussian(wl, center_b, fwhm_b)
    pl   = r_eq + g_eq + b_eq
    if normalize:
        m = pl.max()
        if m > 0:
            r_eq, g_eq, b_eq, pl = r_eq/m, g_eq/m, b_eq/m, pl/m
    return wl, r_eq, g_eq, b_eq, pl


# ─────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────
CROP_SIZE = 175

def decode_to_rgb(file_obj) -> np.ndarray:
    data = np.frombuffer(file_obj.getvalue(), dtype=np.uint8)
    bgr  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image — please use PNG or JPG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def center_crop(img: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = max(size/h, size/w)
        img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    x0, y0 = (w-size)//2, (h-size)//2
    return img[y0:y0+size, x0:x0+size].copy()

def mean_rgb(img: np.ndarray):
    m = img.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])


# ─────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────
def make_fig(wl, r_eq, g_eq, b_eq, pl, show_sum, show_r, show_g, show_b, normalized):
    mpl.rcParams.update({
        "figure.facecolor":  "#13161e",
        "axes.facecolor":    "#0d0f14",
        "axes.edgecolor":    "#1f2433",
        "axes.labelcolor":   "#6b7694",
        "xtick.color":       "#6b7694",
        "ytick.color":       "#6b7694",
        "grid.color":        "#1f2433",
        "grid.linewidth":    0.7,
        "legend.facecolor":  "#13161e",
        "legend.edgecolor":  "#1f2433",
        "legend.labelcolor": "#e2e8f4",
        "font.family":       "monospace",
        "font.size":         9,
    })

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.grid(True, linestyle="--", alpha=0.5)

    if show_sum:
        ax.plot(wl, pl,    color="#ffd35b", lw=2,   label="Simulated PL", zorder=5)
        ax.fill_between(wl, pl, alpha=0.06, color="#ffd35b")
    if show_r:
        ax.plot(wl, r_eq,  color="#ff5b5b", lw=1.4, label="R component",  zorder=4)
        ax.fill_between(wl, r_eq, alpha=0.05, color="#ff5b5b")
    if show_g:
        ax.plot(wl, g_eq,  color="#5bff9a", lw=1.4, label="G component",  zorder=4)
        ax.fill_between(wl, g_eq, alpha=0.05, color="#5bff9a")
    if show_b:
        ax.plot(wl, b_eq,  color="#5bb4ff", lw=1.4, label="B component",  zorder=4)
        ax.fill_between(wl, b_eq, alpha=0.05, color="#5bb4ff")

    ax.set_xlabel("Wavelength (nm)", labelpad=8)
    ax.set_ylabel("Normalized Intensity (a.u.)" if normalized else "Intensity (a.u.)", labelpad=8)
    ax.set_xlim(360, 800)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.1rem;">⚗ PL Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem; color:#6b7694; font-family:Space Mono,monospace;">v2.0 · RGB → Spectrum</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="sidebar-section">Display</p>', unsafe_allow_html=True)
    normalize = st.checkbox("Normalize spectrum", value=True)
    show_sum  = st.checkbox("PL sum (simulated)", value=True)
    show_r    = st.checkbox("R component", value=True)
    show_g    = st.checkbox("G component", value=True)
    show_b    = st.checkbox("B component", value=True)

    st.markdown('<p class="sidebar-section">About</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem; color:#6b7694; font-family:DM Sans,sans-serif; line-height:1.6;">Each RGB channel is mapped to a Gaussian emission band. The sum approximates a photoluminescence spectrum.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem; color:#6b7694; font-family:Space Mono,monospace; margin-top:0.5rem;">Crop: 175 × 175 px (center)</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🔬 PL Spectrum Analyzer</p>
  <p class="hero-sub">Upload or capture an image → extract RGB → simulate photoluminescence</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Step 1 — Image source
# ─────────────────────────────────────────
st.markdown("### 01 · Select Image Source")
source = st.radio("", ["📁  Upload image", "📷  Use camera"], horizontal=True, label_visibility="collapsed")

img_rgb = None
if "Upload" in source:
    f = st.file_uploader("Drop a PNG or JPG here", type=["png","jpg","jpeg"], label_visibility="collapsed")
    if f:
        try: img_rgb = decode_to_rgb(f)
        except Exception as e: st.error(str(e)); st.stop()
    else:
        st.info("⬆ Upload an image to get started.")
        st.stop()
else:
    cam = st.camera_input("", label_visibility="collapsed")
    if cam:
        try: img_rgb = decode_to_rgb(cam)
        except Exception as e: st.error(str(e)); st.stop()
    else:
        st.info("📷 Take a photo to get started.")
        st.stop()


# ─────────────────────────────────────────
# Process image
# ─────────────────────────────────────────
crop  = center_crop(img_rgb, CROP_SIZE)
R, G, B = mean_rgb(crop)
hex_color = "#{:02x}{:02x}{:02x}".format(int(R), int(G), int(B))

st.markdown("---")

# ─────────────────────────────────────────
# Step 2 — Inspect
# ─────────────────────────────────────────
st.markdown("### 02 · Inspect Crop & Color")

col_img, col_info = st.columns([1, 1.6], gap="large")

with col_img:
    st.markdown('<p class="card-title">Center crop · 175 × 175 px</p>', unsafe_allow_html=True)
    st.image(crop, use_container_width=True)

with col_info:
    st.markdown('<p class="card-title">Mean RGB values</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile r">
        <div class="metric-label">Red</div>
        <div class="metric-value">{R:.1f}</div>
      </div>
      <div class="metric-tile g">
        <div class="metric-label">Green</div>
        <div class="metric-value">{G:.1f}</div>
      </div>
      <div class="metric-tile b">
        <div class="metric-label">Blue</div>
        <div class="metric-value">{B:.1f}</div>
      </div>
    </div>
    <div class="swatch-wrap">
      <div class="swatch" style="background:{hex_color};"></div>
      <div>
        <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e2e8f4;">{hex_color.upper()}</div>
        <div class="swatch-label">Mean color of cropped region</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# Step 3 — Gaussian model parameters
# ─────────────────────────────────────────
st.markdown("### 03 · Tune Gaussian Model")
st.markdown('<p style="font-size:0.82rem;color:#6b7694;margin-bottom:1rem;">Each channel maps to a Gaussian emission band. Adjust center wavelength and FWHM to match your material.</p>', unsafe_allow_html=True)

p_col_r, p_col_g, p_col_b = st.columns(3, gap="medium")

with p_col_r:
    st.markdown('<span class="param-label param-r">● Red channel</span>', unsafe_allow_html=True)
    center_r = st.number_input("Peak wavelength (nm)", min_value=550, max_value=650, value=600, step=10, key="cr", help="Center wavelength for the red emission band")
    fwhm_r   = st.number_input("FWHM (nm)", min_value=50, max_value=250, value=150, step=10, key="fr", help="Full-width at half-maximum for red")

with p_col_g:
    st.markdown('<span class="param-label param-g">● Green channel</span>', unsafe_allow_html=True)
    center_g = st.number_input("Peak wavelength (nm)", min_value=450, max_value=550, value=500, step=10, key="cg")
    fwhm_g   = st.number_input("FWHM (nm)", min_value=30, max_value=150, value=80, step=5, key="fg")

with p_col_b:
    st.markdown('<span class="param-label param-b">● Blue channel</span>', unsafe_allow_html=True)
    center_b = st.number_input("Peak wavelength (nm)", min_value=380, max_value=480, value=430, step=10, key="cb")
    fwhm_b   = st.number_input("FWHM (nm)", min_value=30, max_value=120, value=60, step=5, key="fb")


# ─────────────────────────────────────────
# Compute spectrum
# ─────────────────────────────────────────
wl, r_eq, g_eq, b_eq, pl = rgb_to_pl(
    R, G, B, normalize=normalize,
    center_r=center_r, fwhm_r=fwhm_r,
    center_g=center_g, fwhm_g=fwhm_g,
    center_b=center_b, fwhm_b=fwhm_b,
)

st.markdown("---")

# ─────────────────────────────────────────
# Step 4 — Spectrum
# ─────────────────────────────────────────
st.markdown("### 04 · Simulated PL Spectrum")

fig = make_fig(wl, r_eq, g_eq, b_eq, pl, show_sum, show_r, show_g, show_b, normalize)
st.pyplot(fig, clear_figure=True)

# Peak info
if show_sum and pl.max() > 0:
    peak_wl = float(wl[np.argmax(pl)])
    st.markdown(f'<p style="font-family:Space Mono,monospace;font-size:0.78rem;color:#6b7694;text-align:right;">Peak emission · <span style="color:#ffd35b;">{peak_wl:.0f} nm</span></p>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# Step 5 — Export
# ─────────────────────────────────────────
st.markdown("### 05 · Export Results")

dl_csv, dl_png = st.columns(2, gap="medium")

with dl_csv:
    rows = ["wavelength_nm,sum_pl,r_component,g_component,b_component"]
    for i in range(len(wl)):
        rows.append(f"{wl[i]:.1f},{pl[i]:.6f},{r_eq[i]:.6f},{g_eq[i]:.6f},{b_eq[i]:.6f}")
    st.download_button(
        "⬇ Download spectrum.csv",
        data="\n".join(rows).encode("utf-8"),
        file_name="pl_spectrum.csv",
        mime="text/csv",
    )

with dl_png:
    buf = io.BytesIO()
    fig2 = make_fig(wl, r_eq, g_eq, b_eq, pl, show_sum, show_r, show_g, show_b, normalize)
    fig2.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="#13161e")
    plt.close(fig2)
    st.download_button(
        "⬇ Download graph.png",
        data=buf.getvalue(),
        file_name="pl_spectrum.png",
        mime="image/png",
    )

st.markdown('<p style="text-align:center;font-family:Space Mono,monospace;font-size:0.65rem;color:#2a2f3e;margin-top:3rem;">PL Spectrum Analyzer · RGB → Gaussian emission model</p>', unsafe_allow_html=True)
