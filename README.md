# PL Graph from Image (Fixed RGB->PL Model)

Streamlit app that:
- Uploads an image
- Center-crops **500x500 px**
- Computes mean RGB in the crop
- Uses a fixed 3-Gaussian model (your `rgb_to_pl`) to create a PL spectrum
- Allows toggling R/G/B component curves on the plot

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Notes
- If the uploaded image is smaller than 500x500, the app will **scale it up** (keeping aspect ratio) before center-cropping.
- All processing constants are fixed in code (centers, FWHM, wavelength range).
