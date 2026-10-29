"""Central configuration, read from environment variables (and `.env` if present).

Import this module before anything else reads `os.environ` so values from
`.env` are visible. See `.env.example` for documentation of every setting.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; plain environment variables still work.
    pass


def _bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Flask / web server ---
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', '').strip()
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = _int('PORT', 5000)
DEBUG = _bool('FLASK_DEBUG', False)
OPEN_BROWSER = _bool('OPEN_BROWSER', True)
MAX_UPLOAD_MB = _int('MAX_UPLOAD_MB', 20)

# --- Authentication ---
AUTH_ENABLED = _bool('AUTH_ENABLED', False)
APP_USERNAME = os.environ.get('APP_USERNAME', 'admin')
# Generate with: python -c "from werkzeug.security import generate_password_hash as g; print(g('your-password'))"
APP_PASSWORD_HASH = os.environ.get('APP_PASSWORD_HASH', '').strip()

# --- Summarization / translation ---
# LIGHTWEIGHT_MODE skips all Hugging Face models (BART, mBART, Marian, MiniLM)
# and uses TF-IDF + frequency-based extractive summarization instead.
LIGHTWEIGHT_MODE = _bool('LIGHTWEIGHT_MODE', False)
# Allow falling back to Google Translate (via deep-translator; needs internet).
ENABLE_ONLINE_TRANSLATION = _bool('ENABLE_ONLINE_TRANSLATION', True)
# Allow /api/tts to synthesize speech with gTTS (needs internet).
ENABLE_ONLINE_TTS = _bool('ENABLE_ONLINE_TTS', True)
# int8-quantize the translation model on CPU (about 2x faster, same quality in tests).
QUANTIZE_TRANSLATION = _bool('QUANTIZE_TRANSLATION', True)
# Beam size for the translation model. In tests 2 cost about the same as greedy (1)
# and fixed its word errors; 4 was slightly better again but about twice as slow.
TRANSLATION_BEAMS = max(1, _int('TRANSLATION_BEAMS', 2))
# Load the models in the background when the server starts.
PRELOAD_MODELS = _bool('PRELOAD_MODELS', True)
DEFAULT_LANGUAGE = os.environ.get('DEFAULT_LANGUAGE', 'kn')
DEFAULT_SUMMARY_MODE = os.environ.get('DEFAULT_SUMMARY_MODE', 'balanced')
# Optional: where Hugging Face caches downloaded models (standard HF variable).
# HF_HOME is read by transformers directly; listed here for documentation only.

# --- PDF export ---
# Extra directory searched for Noto / Nirmala TTF fonts (Indic script support).
PDF_FONT_DIR = os.environ.get('PDF_FONT_DIR', '').strip()
