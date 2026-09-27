import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Fixed RGB -> PL model (yours)
# -----------------------------
normalize=True
def rgb_to_pl_components(R, G, B, wavelenght=np.arange(360, 800, 1), normalize=True, center_r=600, center_g=500, center_b=430, fwhm_r=150, fwhm_g=80, fwhm_b=60):
    def peak(wavelenght, center, fwhm):
        sigma = fwhm / 2.355
        return np.exp(-((wavelenght - center) ** 2) / (2 * (sigma ** 2)))

    r_norm = R / 255.0
    g_norm = G / 255.0
    b_norm = B / 255.0

    r_eq = r_norm * peak(wavelenght, center_r, fwhm_r)
    g_eq = g_norm * peak(wavelenght, center_g, fwhm_g)
    b_eq = b_norm * peak(wavelenght, center_b, fwhm_b)

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
CROP_SIZE = 175

def decode_upload_to_rgb(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image. Try PNG/JPG.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def decode_camera_to_rgb(camera_file) -> np.ndarray:
    # camera_file is an UploadedFile-like object
    data = np.frombuffer(camera_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode camera image.")
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
st.title("RGB → PL Graph")

with st.sidebar:
    st.header("Settings")
    st.write("Crop size: **175×175 px (center)**")
    st.divider()
    st.subheader("Plot toggles")
    show_sum = st.checkbox("Show SUM (PL)", value=True)
    show_r = st.checkbox("Show R component", value=True)
    show_g = st.checkbox("Show G component", value=True)
    show_b = st.checkbox("Show B component", value=True)


source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)

uploaded = None
camera = None

if source == "Upload":
    uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
else:
    camera = st.camera_input("Take a photo")

if source == "Upload":
    if uploaded is None:
        st.info("Please upload an image to proceed.")
        st.stop()
    try:
        img_rgb = decode_upload_to_rgb(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

else:  # Camera
    if camera is None:
        st.info("Please take a photo to proceed.")
        st.stop()
    try:
        img_rgb = decode_camera_to_rgb(camera)
    except Exception as e:
        st.error(str(e))
        st.stop()


crop_rgb_img = center_crop(img_rgb, CROP_SIZE)
R, G, B = mean_rgb(crop_rgb_img)

wl, r_eq, g_eq, b_eq, pl = rgb_to_pl_components(R, G, B, normalize=normalize)

left, right = st.columns([1, 1])

with left:
    st.subheader("Center crop (175*175)")
    st.image(crop_rgb_img, use_container_width=True)
    
    if st.button("Show PL Graph"):
        st.divider()
        st.subheader("PL Spectrum")
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if show_sum:
            ax.plot(wl, pl, label="Simulated PL spectra", color="orange")
        if show_r:
            ax.plot(wl, r_eq, label="Red ", color="red")
        if show_g:
            ax.plot(wl, g_eq, label="Green", color="green")
        if show_b:
            ax.plot(wl, b_eq, label="Blue", color="blue")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Intensity (a.u.)" if normalize else "Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Mean RGB (Average R,G and B value)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R", f"{R:.2f}")
    c2.metric("G", f"{G:.2f}")
    c3.metric("B", f"{B:.2f}")
    
    st.divider()
    st.subheader("Model constants")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        center_r = st.number_input("Corresponding wavelenght of R (nm)", min_value=550, max_value=650, value=600, step=10, key="cr")
    with col_r2:
        fwhm_r = st.number_input("FWHM R", min_value=50, max_value=250, value=150, step=10, key="fr")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        center_g = st.number_input("Corresponding wavelenght of G (nm)", min_value=450, max_value=550, value=500, step=10, key="cg")
    with col_g2:
        fwhm_g = st.number_input("FWHM G", min_value=30, max_value=150, value=80, step=5, key="fg")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        center_b = st.number_input("Corresponding wavelenght of B (nm)", min_value=380, max_value=480, value=430, step=10, key="cb")
    with col_b2:
        fwhm_b = st.number_input("FWHM B", min_value=30, max_value=120, value=60, step=5, key="fb")
    
    # Recalculate with new parameters
    wl, r_eq, g_eq, b_eq, pl = rgb_to_pl_components(R, G, B, normalize=normalize, center_r=center_r, center_g=center_g, center_b=center_b, fwhm_r=fwhm_r, fwhm_g=fwhm_g, fwhm_b=fwhm_b)

st.divider()
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if show_sum:
        ax.plot(wl, pl, label="Simulated PL spectra", color="orange")
    if show_r:
        ax.plot(wl, r_eq, label="Red ", color="red")
    if show_g:
        ax.plot(wl, g_eq, label="Green", color="green")
    if show_b:
        ax.plot(wl, b_eq, label="Blue", color="blue")
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity (a.u.)" if normalize else "Intensity (a.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button(
        "Download graph.png",
        data=buf.getvalue(),
        file_name="pl_graph.png",
        mime="image/png"
    )
