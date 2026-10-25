# VaniScorer 🏏

A voice-first cricket scorer for local matches. Say what happened on the ball, and the app updates the score, commentary, batter and bowler stats, and wagon wheel.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Add your Sarvam key as `SARVAM_API_KEY` in `.env` locally or in Streamlit Community Cloud under **App settings → Secrets**:

```toml
SARVAM_API_KEY = "your-key"
```

The app uses Sarvam Saaras speech recognition and Sarvam Chat Completions to interpret ball descriptions. Do not commit your key. Without a key, manual scoring remains available.

## Deploy

Create a Streamlit Community Cloud app from this GitHub repository, select `app.py` as the entry point, then add `SARVAM_API_KEY` in the app's Secrets settings.
