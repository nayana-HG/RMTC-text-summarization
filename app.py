import functools
import io
import logging
import os
import secrets
import sys
import threading

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash

import config  # loads .env before the other modules read the environment
import backend
import exporter

# Ensure UTF-8 console output for Windows terminal safety
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
if config.SECRET_KEY:
    app.secret_key = config.SECRET_KEY
else:
    app.secret_key = secrets.token_hex(32)
    log.warning("FLASK_SECRET_KEY is not set; using a random key, so sessions reset on every restart.")
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024

if config.AUTH_ENABLED and not config.APP_PASSWORD_HASH:
    log.warning("AUTH_ENABLED is set but APP_PASSWORD_HASH is empty; nobody will be able to log in.")

LANGUAGE_NAMES = {
    'en': 'English',
    'kn': 'Kannada (ಕನ್ನಡ)',
    'hi': 'Hindi (हिंदी)',
    'bn': 'Bengali (বাংলা)',
    'ta': 'Tamil (தமிழ்)',
    'te': 'Telugu (తెలుగు)',
    'ml': 'Malayalam (മലയാളം)',
    'mr': 'Marathi (मराठी)',
    'gu': 'Gujarati (ગુજરાતી)',
    'pa': 'Punjabi (ਪੰਜਾਬੀ)',
    'ur': 'Urdu (اردو)',
    'es': 'Spanish (Español)',
    'fr': 'French (Français)',
    'de': 'German (Deutsch)',
}

SUMMARY_MODE_NAMES = {
    'simple': 'Simple (short and easy to read)',
    'balanced': 'Balanced',
    'detailed': 'Detailed (longer, in-depth)',
}

# Languages gTTS can speak (used by /api/tts).
_TTS_LANGUAGES = {'en', 'kn', 'hi', 'bn', 'ta', 'te', 'ml', 'mr', 'gu', 'pa', 'ur', 'es', 'fr', 'de'}


def login_required(view):
    """Require a logged-in session when AUTH_ENABLED is on; no-op otherwise."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if config.AUTH_ENABLED and not session.get('logged_in'):
            if request.path.startswith('/api/'):
                return jsonify({'status': 'error', 'message': 'Login required.'}), 401
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped


def _language_choice(value):
    value = (value or config.DEFAULT_LANGUAGE).strip().lower()
    return value if value in LANGUAGE_NAMES else 'en'


def _summarize_request(raw_text, uploaded_files, language, summary_mode, explain=True):
    """Extract text from the inputs and summarize each document separately.

    Returns (doc_results, errors): one dict per document, and user-facing
    messages for files that could not be read.
    """
    docs, errors = [], []
    for f in uploaded_files:
        if not f or not f.filename.strip():
            continue
        try:
            docs.append((f.filename, backend.extract_text(f)))
        except backend.ExtractionError as e:
            errors.append(str(e))
    if raw_text:
        docs.append(("Raw Text Input", raw_text))

    results = []
    for doc_name, doc_text in docs:
        try:
            r = backend.summarize(doc_text, language=language, explain=explain, summary_mode=summary_mode)
            results.append(dict(r.to_dict(), doc_name=doc_name))
        except Exception as e:
            log.exception("Error processing document %s", doc_name)
            errors.append(f"Could not summarize {doc_name}: {e}")
    return results, errors


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    message = f"Upload is too large (limit {config.MAX_UPLOAD_MB} MB)."
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': message}), 413
    return render_template('summary.html', error=message), 413


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if not config.AUTH_ENABLED:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if (config.APP_PASSWORD_HASH and secrets.compare_digest(username, config.APP_USERNAME)
                and check_password_hash(config.APP_PASSWORD_HASH, password)):
            session.clear()
            session['logged_in'] = True
            next_url = request.args.get('next', '')
            # Only follow local redirects to avoid an open redirect.
            if not next_url.startswith('/') or next_url.startswith('//'):
                next_url = url_for('index')
            return redirect(next_url)
        error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))


@app.route('/index')
@login_required
def index():
    return render_template('index.html', languages=LANGUAGE_NAMES, modes=SUMMARY_MODE_NAMES,
                           default_language=config.DEFAULT_LANGUAGE,
                           default_mode=backend.normalize_mode(config.DEFAULT_SUMMARY_MODE))


@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method != 'POST':
        return redirect(url_for('index'))

    language = _language_choice(request.form.get('language'))
    summary_mode = backend.normalize_mode(request.form.get('summary_mode'))
    doc_results, errors = _summarize_request(
        request.form.get('rawtext', '').strip(), request.files.getlist('pdf_file'), language, summary_mode)

    if not doc_results:
        error = ' '.join(errors) or "Please enter some text or upload at least one document."
        return render_template('summary.html', error=error)

    return render_template(
        'summary.html',
        doc_results=doc_results,
        errors=errors,
        language=language,
        language_name=LANGUAGE_NAMES[language],
        summary_mode=summary_mode,
    )


@app.route('/api/summarize', methods=['POST'])
@login_required
def handle_summarize():
    language = _language_choice(request.form.get('language'))
    summary_mode = backend.normalize_mode(request.form.get('summary_mode'))
    explain = request.form.get('explain', 'true').lower() in ('true', '1', 'yes')
    files = request.files.getlist('files') or request.files.getlist('pdf_file')

    try:
        doc_results, errors = _summarize_request(
            request.form.get('text', '').strip(), files, language, summary_mode, explain=explain)
    except Exception as e:
        log.exception("Summarization failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

    if not doc_results:
        return jsonify({
            'status': 'error',
            'message': ' '.join(errors) or 'No text input or files provided.',
            'errors': errors,
        }), 400

    first = doc_results[0]
    return jsonify({
        'status': 'success',
        'doc_results': doc_results,
        'errors': errors,
        # Top-level fields mirror the first document for older clients.
        'summary': first['summary'],
        'original_text': first['original_text'],
        'explanation': first['explanation'],
        'keywords': first['keywords'],
        'language': language,
        'language_name': LANGUAGE_NAMES[language],
    })


def _export_args():
    data = request.get_json(force=True, silent=True) or {}
    keywords = data.get('keywords') or []
    if not isinstance(keywords, list):
        keywords = []
    return dict(
        summary=str(data.get('summary', '')),
        original_text=str(data.get('original_text', '')),
        language_name=str(data.get('language_name', 'Output')),
        keywords=[str(k) for k in keywords],
        explanation=str(data.get('explanation', '') or ''),
    )


@app.route('/api/download/pdf', methods=['POST'])
@login_required
def download_pdf():
    try:
        pdf_bytes = exporter.generate_pdf_bytes(**_export_args())
    except exporter.ExportUnavailable as e:
        return jsonify({'status': 'error', 'message': str(e)}), 501
    except Exception as e:
        log.exception("PDF export failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf',
                     as_attachment=True, download_name='summary.pdf')


@app.route('/api/download/docx', methods=['POST'])
@login_required
def download_docx():
    try:
        docx_bytes = exporter.generate_docx_bytes(**_export_args())
    except exporter.ExportUnavailable as e:
        return jsonify({'status': 'error', 'message': str(e)}), 501
    except Exception as e:
        log.exception("DOCX export failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    return send_file(
        io.BytesIO(docx_bytes),
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name='summary.docx',
    )


@app.route('/api/tts', methods=['POST'])
@login_required
def text_to_speech():
    """Synthesize speech with gTTS. The frontend falls back to browser speech on any error."""
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get('text', '')).strip()
    language = str(data.get('language', 'en')).strip().lower()
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided.'}), 400
    if not config.ENABLE_ONLINE_TTS or language not in _TTS_LANGUAGES:
        return jsonify({'status': 'error', 'message': 'Speech synthesis is not available.'}), 501
    try:
        from gtts import gTTS
    except ImportError:
        return jsonify({'status': 'error', 'message': 'Speech synthesis needs the gTTS package.'}), 501
    try:
        audio = io.BytesIO()
        gTTS(text=text[:5000], lang=language).write_to_fp(audio)
        audio.seek(0)
    except Exception as e:
        log.warning("gTTS failed for %s: %s", language, e)
        return jsonify({'status': 'error', 'message': 'Speech synthesis failed.'}), 502
    return send_file(audio, mimetype='audio/mpeg')


if __name__ == '__main__':
    url = f"http://{config.HOST}:{config.PORT}/index"
    print(f"Starting Text Summarization App on {url}")
    # The debug reloader runs this block twice; load models only in the serving child.
    if config.PRELOAD_MODELS and (not config.DEBUG or os.environ.get('WERKZEUG_RUN_MAIN') == 'true'):
        threading.Thread(target=backend.preload_models, name='preload-models', daemon=True).start()
    if config.OPEN_BROWSER:
        import webbrowser
        from threading import Timer
        Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
