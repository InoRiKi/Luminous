# PL Graph from Image (Fixed RGB->PL Model)
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
