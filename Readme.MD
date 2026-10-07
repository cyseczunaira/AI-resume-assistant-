# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An overall **ATS score** (0-100) plus a breakdown by category
- Strengths and weaknesses
- Keywords found and **missing keywords**
- Prioritised, actionable improvements
- Before/after rewrites of weak bullet points
- A downloadable Markdown report

Built with **Streamlit** and **Google Gemini Flash** (via the `google-genai` SDK).

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then either paste it in the
sidebar or set it up as a secret/environment variable:

```bash
# Option A: environment variable
export GEMINI_API_KEY="your-key"          # Windows (PowerShell): $env:GEMINI_API_KEY="your-key"

# Option B: .streamlit/secrets.toml  (never commit this file!)
GEMINI_API_KEY = "your-key"
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration

| Setting | How | Default |
|---|---|---|
| API key | Sidebar, `GEMINI_API_KEY` secret, or env var | none |
| Model | Sidebar field or `GEMINI_MODEL` env var | `gemini-2.5-flash` |

## Notes and limitations

- The score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Scanned or image-only PDFs can't be read (ATS software can't read them either).
- Resumes are sent to Google's Gemini API for analysis and are not stored by this app.
- Files are limited to 5 MB and ~20,000 characters of resume text.
