import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Fixed RGB -> PL model (yours)
# -----------------------------
def rgb_to_pl_components(R, G, B, wavelenght=np.arange(360, 800, 1), normalize=True):
    def ndopeeqs(wavelenght, center, fwhm):
        sigma = fwhm / 2.355
        return np.exp(-((wavelenght - center) ** 2) / (2 * (sigma ** 2)))

    # FIXED parameters
    center_r = 600
    center_g = 500
    center_b = 430
    fwhm_r = 150
    fwhm_g = 80
    fwhm_b = 60

    r_norm = R / 255.0
    g_norm = G / 255.0
    b_norm = B / 255.0

    r_eq = r_norm * ndopeeqs(wavelenght, center_r, fwhm_r)
    g_eq = g_norm * ndopeeqs(wavelenght, center_g, fwhm_g)
    b_eq = b_norm * ndopeeqs(wavelenght, center_b, fwhm_b)

    pl_spectrum = r_eq + g_eq + b_eq

    if normalize:
        m = float(pl_spectrum.max()) if pl_spectrum.size else 1.0
        if m > 0:
            r_eq = r_eq / m
            g_eq = g_eq / m
            b_eq = b_eq / m
            pl_spectrum = pl_spectrum / m

    return wavelenght, r_eq, g_eq, b_eq, pl_spectrum


# -----------------------------
# Image utilities (fixed crop)
# -----------------------------
CROP_SIZE = 500

def decode_upload_to_rgb(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image. Try PNG/JPG.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def resize_if_too_small(img_rgb: np.ndarray, min_size: int = CROP_SIZE) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h >= min_size and w >= min_size:
        return img_rgb
    scale = max(min_size / h, min_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized

def center_crop(img_rgb: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    img_rgb = resize_if_too_small(img_rgb, min_size=size)
    h, w = img_rgb.shape[:2]
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return img_rgb[y0:y0+size, x0:x0+size].copy()

def mean_rgb(img_rgb: np.ndarray):
    m = img_rgb.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])  # R,G,B


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RGB → PL Graph (Fixed Model)", layout="wide")
st.title("RGB → PL Graph (Fixed Model)")

with st.sidebar:
    st.header("Settings (Fixed)")
    st.write("Crop size: **500×500 px (center)**")
    normalize = st.checkbox("Normalize spectrum (0–1)", value=True)
    st.divider()
    st.subheader("Plot toggles")
    show_sum = st.checkbox("Show SUM (PL)", value=True)
    show_r = st.checkbox("Show R component", value=True)
    show_g = st.checkbox("Show G component", value=True)
    show_b = st.checkbox("Show B component", value=True)
    st.divider()
    st.subheader("Model constants (locked)")
    st.code(
        "center_r=600, fwhm_r=150\n"
        "center_g=500, fwhm_g=80\n"
        "center_b=430, fwhm_b=60\n"
        "wl=360..799 step 1",
        language="text"
    )

uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("plase upload an image to proceed.")
    st.stop()

try:
    img_rgb = decode_upload_to_rgb(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

crop_rgb_img = center_crop(img_rgb, CROP_SIZE)
R, G, B = mean_rgb(crop_rgb_img)

wl, r_eq, g_eq, b_eq, pl = rgb_to_pl_components(R, G, B, normalize=normalize)

left, right = st.columns([1, 1])

with left:
    st.subheader("Center crop (500×500)")
    st.image(crop_rgb_img, use_container_width=True)

with right:
    st.subheader("Mean RGB (in crop)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R", f"{R:.2f}")
    c2.metric("G", f"{G:.2f}")
    c3.metric("B", f"{B:.2f}")

st.subheader("PL Spectrum")
fig = plt.figure()
ax = fig.add_subplot(111)

if show_sum:
    ax.plot(wl, pl, label="PL", color="orange")
if show_r:
    ax.plot(wl, r_eq, label="R", color="red")
if show_g:
    ax.plot(wl, g_eq, label="G", color="green")
if show_b:
    ax.plot(wl, b_eq, label="B", color="blue")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Intensity (a.u.)" if normalize else "Intensity (a.u.)")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig, clear_figure=True)

st.subheader("Export")
col1, col2 = st.columns([1, 1])

with col1:
    csv = "wavelength_nm,sum_pl,r_component,g_component,b_component\n"
    for i in range(len(wl)):
        csv += f"{wl[i]},{pl[i]},{r_eq[i]},{g_eq[i]},{b_eq[i]}\n"
    st.download_button(
        "Download spectrum.csv",
        data=csv.encode("utf-8"),
        file_name="spectrum.csv",
        mime="text/csv"
    )

with col2:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button(
        "Download graph.png",
        data=buf.getvalue(),
        file_name="pl_graph.png",
        mime="image/png"
    )
