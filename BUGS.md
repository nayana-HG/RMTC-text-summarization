# Code Review — RMTC Text Summarization

**Reviewed:** commit `fa29a8d` (master), 2026-10-28
**Scope:** `app.py`, `backend.py`, `exporter.py`, `templates/*.html`, `static/`, tests, config
**Method:** Manual read of the code. Each issue was traced through the code paths. The app was **not** run, because the ML dependencies and Flask are not installed in this environment.

## Fix status (2026-10-28)

**27 of 29 findings are fixed. #26 is open, waiting on a decision, and #14 is superseded by #29.** Regression tests are in
`test_app.py` (31 tests, hermetic, all passing).

| # | Status | How it was fixed |
|---|---|---|
| 1 | ✅ | Removed the back-translation branch; `test_non_english_input_with_english_output_stays_english` |
| 2 | ✅ | Rewrite failure falls back to the extractive summary; `test_abstractive_failure_falls_back_to_extractive` |
| 3 | ✅ | Per-word Unicode font selection (Noto / Nirmala / Lohit) + HarfBuzz shaping (`uharfbuzz`); jsPDF fallback removed |
| 4 | ✅ | All text XML-escaped before reaching ReportLab |
| 5 | ✅ | Sentence-aware chunked translation for every backend (`_chunk_text`, `_translate_chunks`) |
| 6 | ✅ | `_generation_lengths` caps BART lengths by input size; mode selector added to the form |
| 7 | ✅ | `extract_text` raises `ExtractionError`; errors shown separately, never summarized |
| 8 | ✅ | `ExportUnavailable` → HTTP 501; frontend shows the reason |
| 9 | ✅ | Per-document data passed via a `tojson` JSON block; buttons only pass an index |
| 10 | ✅ | SRI hashes replaced with values computed from the live CDN files |
| 11 | ✅ | `/api/tts` implemented with gTTS (optional, `ENABLE_ONLINE_TTS`) |
| 12 | ✅ | Exports include keywords and explanation |
| 13 | ✅ | `es`/`fr`/`de` use Marian `opus-mt-en-{es,fr,de}` and mBART codes |
| 14 | ✅ | Superseded by #29 |
| 15 | ✅ | Unicode-aware tokens, NLTK + extended stopwords, keywords translated (or labelled "English") |
| 16 | ✅ | Fixed seed, short-Latin-text and confidence guards |
| 17 | ✅ | Random noise removed; vectorised deterministic MMR |
| 18 | ✅ | Model loading lock; `src_lang` + encode under a lock |
| 19 | ✅ | Pages joined with `\n` |
| 20 | ✅ | Optional real auth (`AUTH_ENABLED`, hashed password, `login_required`, safe `next` redirect) |
| 21 | ✅ | `FLASK_SECRET_KEY` from env; `MAX_UPLOAD_MB` limit with a 413 handler |
| 22 | ✅ | NumPy cosine, no sklearn dependency for selection; frequency fallback instead of random vectors |
| 23 | ✅ | Summary-style dropdown; one default (`DEFAULT_SUMMARY_MODE`) for both entry points |
| 24 | ✅ | Error page renders no document cards |
| 25 | ✅ | `requirements.txt` (core) + `requirements-ml.txt`; googletrans replaced by deep-translator; README rewritten |
| 26 | ⏸️ **Open** | GloVe file left in place: removing it from Git LFS is a repo-history decision for the maintainers. Documented as unused. |
| 27 | ✅ | Tests are hermetic (lightweight mode, mocked translation) with specific regression tests |
| 28 | ✅ | Unused JS/CSS removed; `OFFLINE_ONLY` replaced by `LIGHTWEIGHT_MODE`; explanation shows real relevance scores |
| 29 | ✅ | Translation model tables rebuilt from models verified to exist on the Hub (see below) |
| 30 | ✅ | Kannada/Telugu/Tamil routed to NLLB-200 first; `opus-mt-mul-en` no longer used for them (see below) |

### 29. 🔴 Most offline translation models didn't exist (found while fixing #14)
**`backend.py` `_get_marian_model` / `_MBART_LANG_MAP` (original code)**

The Hugging Face Hub was checked for each model ID:
- **8 of 12** `Helsinki-NLP/opus-mt-en-*` models don't exist: `bn, gu, kn, or, pa, ta, te, ne`.
- mBART-50 has **no** `kn_IN`, `pa_IN`, `or_IN`, `sd_IN` or `sa_IN`, and Urdu is `ur_PK`, not `ur_IN`.

So **Kannada, the default output language, could never be translated offline**, and neither could Punjabi.
Every request for those languages silently depended on Google Translate.

**Fix:** Kannada, Tamil and Telugu now use `opus-mt-en-dra` (`>>kan<<`, `>>tam<<`, `>>tel<<`). Bengali, Gujarati,
Punjabi, Odia and Nepali use `opus-mt-en-inc`, and any language → English uses `opus-mt-mul-en`. The
multilingual fallback is configurable (mBART-50 by default; NLLB-200 also supported, covering every language).

### 30. 🔴 Kannada, Telugu and Tamil input was summarized from made-up English (found 2026-10-28)
**`backend.py` `_translate_text`**

Tested with the real models. Marian was always tried first, and `opus-mt-mul-en` turned
Kannada, Telugu and Tamil into unrelated English. Because the output wasn't empty, it was
accepted, and the summary described content that wasn't in the document. Examples:
- Kannada, about drought and crop yields → *"In the United States, the number of people who live in the United States…"*
- Tamil, same content → *"According to a report published by the U.S. The king has announced…"*

mBART-50 (the configured fallback) has no Kannada and mixed Malayalam script into Telugu.

**Fix:** NLLB-200 is now the default `MULTILINGUAL_MODEL`. For kn/te/ta it is tried first, then
Google, then Marian (English → language only). Script-based detection was added for these three
languages, and a Google rate-limit error now gets one retry. `run_translation_test.py` runs all 16
en/kn/te/ta input→output pairs through `summarize()` with the real models. All 16 pass.

---

Severity levels:
- 🔴 **High**: wrong output, a crash, or a security problem in the normal flow
- 🟠 **Medium**: breaks a feature or degrades results for common inputs
- 🟡 **Low**: edge cases, cleanup, or config problems

| # | Severity | Area | Summary |
|---|---|---|---|
| 1 | 🔴 | backend | Choosing English output for non-English input returns the source language |
| 2 | 🔴 | backend | `UnboundLocalError` in the BART fallback path |
| 3 | 🔴 | exporter | PDF export can't render Indic scripts (shows blank boxes) |
| 4 | 🔴 | exporter | Unescaped text in reportlab markup crashes PDF export |
| 5 | 🔴 | backend | Long documents are silently cut to about 512–1024 tokens during translation |
| 6 | 🔴 | app / UI | Default "detailed" mode makes BART pad short inputs with made-up text |
| 7 | 🟠 | backend | Error and placeholder strings get summarized as if they were document text |
| 8 | 🟠 | exporter | Fallback "PDF" is plain text and fallback DOCX is empty |
| 9 | 🟠 | templates | Filename injected into `onclick` JS: XSS, and buttons break on `'` |
| 10 | 🟠 | templates | Corrupted SRI hashes on Bootstrap/jQuery/Popper CDN tags |
| 11 | 🟠 | templates | `/api/tts` doesn't exist, so every non-English read-aloud starts with a 404 |
| 12 | 🟠 | templates | Word/PDF downloads always leave out keywords and explanation |
| 13 | 🟠 | backend | Spanish/French/German only work through googletrans |
| 14 | 🟠 | backend | Odia/Sindhi/Sanskrit codes aren't in mBART-50 |
| 15 | 🟠 | backend | Keywords are English-only and never translated |
| 16 | 🟠 | backend | Language detection is unreliable and gives different answers on reruns |
| 17 | 🟠 | backend | Summaries vary between runs because of random noise in sentence selection |
| 18 | 🟠 | backend | Shared tokenizer state can mix up concurrent requests |
| 19 | 🟠 | backend | PDF pages are joined with no separator |
| 20 | 🟠 | security | Login is fake: hardcoded credentials, never checked |
| 21 | 🟠 | security | Hardcoded `secret_key`, no upload size limit |
| 22 | 🟡 | backend | Crash if scikit-learn is missing; random embeddings as a last resort |
| 23 | 🟡 | app | No summary-mode selector in the UI |
| 24 | 🟡 | app | Error page also shows an empty "Document Output" card |
| 25 | 🟡 | config | `requirements.txt` is missing Flask, python-docx, reportlab |
| 26 | 🟡 | config | GloVe file (347 MB, LFS) is unused, and `.gitignore` excludes it |
| 27 | 🟡 | tests | Tests need model downloads and the internet, so they aren't reproducible |
| 28 | 🟡 | cleanup | Dead code and unused static files |

---

## 🔴 High

### 1. Choosing English output for non-English input returns the source language
**`backend.py:621-625`**

```python
else:
    if language == 'en' and source_lang != 'en' and translated_to_en:
        back_trans = _translate_text(summary, 'en', source_lang)
        if back_trans:
            summary = back_trans
```

The user picked English, and the summary is already in English at this point. The code then translates it **back** into the detected source language.

**Example:** paste Hindi text and select English. The summary comes back in Hindi.
**Fix:** Delete this `else` branch. The summary is already in the requested language.

---

### 2. `UnboundLocalError` in the BART fallback path
**`backend.py:580-602`**

```python
try:
    _load_abstractive_model()
    ...
    summary = abstractive
except Exception as e:
    print(f"Abstractive rewriter failed: {e}")
    if not summary:          # 'summary' was never assigned
        summary = extractive_summary
```

`summary` is first assigned *inside* the `try`. Suppose `_load_abstractive_model()` or `generate()` raises: model download fails, the machine runs out of memory, or there's a CUDA error. Then `if not summary` raises `UnboundLocalError` inside the `except` block. The intended fallback never runs. `/api/summarize` returns a 500, and `/analyze` shows "Error summarizing …".

**Fix:** Set `summary = ""` before the `try`. Or in the `except`, simply set `summary = extractive_summary`.

---

### 3. PDF export can't render Indic scripts
**`exporter.py:109-195`**, and the jsPDF fallback in **`templates/summary.html:322-330`**

reportlab's built-in fonts (Helvetica) and jsPDF's default font only cover Latin-1. Kannada, Hindi, Tamil, Telugu, Bengali, Malayalam, Gujarati, Punjabi and Urdu text all come out as black boxes or blank space. Those are the app's main output languages, and Kannada is the default.

**Fix:** Register a Unicode TTF with reportlab (e.g. Noto Sans Kannada/Devanagari/…, chosen per language) using `pdfmetrics.registerFont(TTFont(...))` and use it in every `ParagraphStyle`. Note that reportlab doesn't do complex text shaping, so some conjunct characters may still render wrong. For fully correct Indic rendering, generate the PDF from HTML with a shaping-aware engine such as WeasyPrint.

---

### 4. Unescaped text in reportlab markup crashes PDF export
**`exporter.py:171, 176-177, 183, 189, 197-198`**

`Paragraph` reads its input as XML-like markup. Summary, explanation, original text, keywords and `language_name` are all inserted raw. Any `<`, `>` or `&` in the input causes a parse error, e.g. `"profit < 5% & rising"` or text from a scientific PDF. `/api/download/pdf` then returns a 500. Because `language_name` and `keywords` come from the request body, a client can also inject markup (`<font>`, `<a href>`).

**Fix:** Run every value through `xml.sax.saxutils.escape()` **before** replacing `\n` with `<br/>`.

---

### 5. Long documents are silently cut during translation
**`backend.py:216, 220, 239, 243, 256, 260`**

Every translation call uses `truncation=True` (about 1024 input tokens for mBART, 512 for Marian) and `max_length=512` for the output. In the non-English flow, the **whole document** is translated to English first (`backend.py:~503`). So for a multi-page Kannada or Hindi PDF, only roughly the first page gets translated, and the summary covers only that. Nothing tells the user this happened.

The same limit applies when translating the summary and explanation to the target language. A "detailed" summary can be up to 600 tokens (see #6), so its end gets cut off. googletrans also has a limit of about 5000 characters per request.

**Fix:** Split the text into sentences, group them into chunks under the token limit, translate each chunk, and join the results.

---

### 6. Default "detailed" mode makes BART pad short inputs with made-up text
**`app.py:72`, `templates/index.html` (no `summary_mode` field), `backend.py:541-560`**

The HTML form has no summary-mode field, so `/analyze` always uses `'detailed'`. In that mode, `min_length = 120`. For non-English output it rises to `int(120*1.2) = 144` tokens, with `max_length = 600`. BART is **forced** to produce at least 144 tokens even when the input is two sentences, and it fills the gap with repeated or invented content. This is the default path for every user of the web UI.

**Fix:** Cap the length settings relative to the input, e.g. `gen_min_length = min(gen_min_length, int(0.5 * input_token_count))` and `gen_max_length = min(gen_max_length, input_token_count)`. Also add a mode selector (#23).

---

## 🟠 Medium

### 7. Error and placeholder strings get summarized as if they were document text
**`backend.py:323-353`, `app.py:76-82, 145-151`**

When extraction fails, `_extract_texts_from_files` returns text such as:

- `[Error extracting report.pdf: ...]`
- `[Image uploaded — OCR not available on server]`
- `[DOCX uploaded — python-docx library not installed]`

`app.py` only checks `t.strip()`, so these strings are passed through as real content. The user then sees a "summary" of an error message, translated into Kannada. Scanned PDFs (which have no text layer) give an empty string and are silently dropped, with no message.

**Fix:** Return errors separately, e.g. `(text, error)`, and show them on the page.

### 8. Fallback "PDF" is plain text and fallback DOCX is empty
**`exporter.py:24-33, 101-106`**

- Without python-docx, `docx` is `None`, so the function returns `b""`. The user downloads a 0-byte `summary.docx` with HTTP 200.
- Without reportlab, the function returns UTF-8 plain text, but `app.py` sends it as `application/pdf` named `summary.pdf`, which is a corrupt file.

**Fix:** Raise an error (or return a 501) when the library is missing so the frontend's own fallback runs. Or add both libraries to the requirements (#25).

### 9. Filename injected into `onclick` JavaScript
**`templates/summary.html:197-200`**

```html
onclick="downloadTXT({{ loop.index0 }}, '{{ item.doc_name }}')"
```

Jinja's escaping turns `'` into `&#39;`, but the browser decodes that back to `'` *before* running the `onclick` code. So the escaping doesn't protect anything here:

- A file named `John's notes.pdf` breaks all four export buttons with a JS syntax error.
- A file named `x');alert(document.cookie);//.txt` runs arbitrary script. Today this only affects the person who uploaded it, but it becomes a real XSS hole if results are ever shared.

**Fix:** `onclick="downloadTXT({{ loop.index0 }}, {{ item.doc_name | tojson | forceescape }})"`. Or better, put the name in `data-doc-name="{{ item.doc_name }}"` and read it from JS.

### 10. Corrupted SRI hashes on CDN tags
**`templates/index.html:11, 84, 89, 94`, `templates/home.html:11, 112, 117, 122`, `templates/summary.html:11`**

The `integrity` hashes don't match the real Bootstrap 4.0.0, jQuery 3.2.1-slim and Popper 1.12.9 hashes. Some even contain `@`, which isn't a valid base64 character. They look like they were OCR'd or retyped by hand. For example, the real Bootstrap CSS hash is `sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm`. Browsers block any resource whose hash doesn't match, so **Bootstrap CSS likely isn't loading** and the grid layout falls back to unstyled HTML. Check the DevTools console for "Failed to find a valid digest".

**Fix:** Copy the correct tags from the Bootstrap 4.0.0 docs, or remove the `integrity`/`crossorigin` attributes.

### 11. `/api/tts` doesn't exist
**`templates/summary.html:70`, `static/js/app.js:310`**

For every non-English language, "Play" first POSTs to `/api/tts`, which isn't a route in `app.py`. It gets a 404 and then falls back to the browser's speech engine. So read-aloud does work, but with one wasted round trip. It also depends entirely on the OS having a voice for the language; most desktop browsers have no Kannada voice, so it reads silently or in the wrong accent.

**Fix:** Either add a `/api/tts` route (e.g. gTTS, which supports kn/hi/ta/te/…) or remove the fetch call.

### 12. Word/PDF downloads always leave out keywords and explanation
**`templates/summary.html:264-265, 307-308`**

The download functions hardcode `keywords: []` and `explanation: ""`, even though `exporter.py` supports both and the page displays them. The exported reports never include those sections.

**Fix:** Pass `{{ item.keywords | tojson }}` and the explanation through `data-*` attributes and include them in the request.

### 13. Spanish, French and German only work through googletrans
**`backend.py:101-117, 151-164`**

`es`, `fr` and `de` are offered in the UI but missing from both `_MBART_LANG_MAP` and the Marian map. mBART-50 supports them as `es_XX`, `fr_XX` and `de_DE`. So these languages always go to googletrans, which needs internet access and often breaks (#25). If it fails, the user gets `[Translation to es not available]`.

**Fix:** Add `'es': 'es_XX', 'fr': 'fr_XX', 'de': 'de_DE'` to `_MBART_LANG_MAP`.

### 14. Odia, Sindhi and Sanskrit codes aren't in mBART-50
**`backend.py:109, 113-114`**

`or_IN`, `sd_IN` and `sa_IN` are not mBART-50 language codes, so `lang_code_to_id[tag]` raises `KeyError`. The error is caught, but the code quietly falls through to googletrans. Several `Helsinki-NLP/opus-mt-en-*` models in the Marian map (e.g. `en-kn`, `en-te`, `en-gu`, `en-pa`) may also not exist on the Hugging Face Hub. Each one costs a failed network request the first time it's tried.

**Fix:** Remove the unsupported mBART codes. Check that each Marian model ID exists, and keep only those.

### 15. Keywords are English-only and never translated
**`backend.py:443, 604`**

- The regex `[a-zA-Z]{3,}` matches only Latin letters. If translation to English failed, Indic input produces **no keywords**.
- When it works, keywords come from the English working text and are never translated. A Kannada summary is shown with English keywords.
- The stopword list has only 25 entries, so common words like "will", "their" and "also" become keywords.

**Fix:** Use `\w{3,}` with a real stopword list (e.g. NLTK's) or TF-IDF/KeyBERT on the embeddings you already compute. Translate the keywords, or label them as English.

### 16. Language detection is unreliable
**`backend.py:490-495`**

- `langdetect` gives different answers on reruns unless `DetectorFactory.seed = 0` is set.
- It's unreliable on short text, often labelling short English snippets as `af`, `nl` or `so`. That sends English text through an unnecessary round-trip translation and loses quality.
- It also returns codes that no translation map knows, such as `zh-cn`.

**Fix:** Set the seed. Skip detection (assume English) for text under about 50 characters, or use `detect_langs` and require probability > 0.9.

### 17. Summaries vary between runs
**`backend.py:401-405`**

```python
import random
if random.random() < 0.08:
    score += random.uniform(-0.05, 0.05)
```

Random noise is added to sentence scores, so the same input can produce a different summary each time. This makes testing and debugging harder for no clear benefit. (`import random` inside the inner loop is also wasteful.)

**Fix:** Remove it, or seed it from a hash of the input.

### 18. Shared tokenizer state can mix up concurrent requests
**`backend.py:215, 238`**

`_MBART_TOKENIZER.src_lang` is a shared global that every request modifies. Flask's server is multi-threaded by default. Two requests with different source languages can overwrite each other's `src_lang` between setting it and encoding, so one gets translated with the wrong source language. Model loading is also unguarded, so two first requests can both load a multi-GB model at once.

**Fix:** Put a `threading.Lock` around model loading and around the set-`src_lang`-then-encode step. Or run with `threaded=False` or one worker per process.

### 19. PDF pages are joined with no separator
**`backend.py:297, 332`**

`''.join(page texts)` glues the last word of one page to the first word of the next, e.g. `...the resultsMethodology...`. The sentence splitter then treats them as one sentence.

**Fix:** Use `'\n'.join(...)`.

### 20. Login is fake
**`app.py:46-61`**

- The credentials are hardcoded as `admin` / `password`.
- `session['logged_in']` is set but never checked. `/index`, `/analyze` and every `/api/*` route are fully public. The comment *"Direct access allowed for convenience"* confirms this.

**Fix:** Either remove the login page, or add a `login_required` decorator and real user storage with hashed passwords.

### 21. Hardcoded `secret_key`, no upload size limit
**`app.py:18`**

- `app.secret_key = 'your_secret_key_here'`: anyone can forge session cookies.
- `MAX_CONTENT_LENGTH` isn't set, so a single huge upload is read fully into memory (`f.read()` at `backend.py:323`), and a large one can exhaust memory.

**Fix:** Read the key with `os.environ['FLASK_SECRET_KEY']` and set `app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024`.

---

## 🟡 Low

### 22. Crash if scikit-learn is missing; random embeddings as a last resort
**`backend.py:44-46, 375-379, 389`**

If scikit-learn is missing, `cosine_similarity` is `None`, and `_hybrid_selector` crashes with `TypeError: 'NoneType' object is not callable`. In the same situation, `_compute_embeddings` returns **random vectors**, so sentence selection would be meaningless anyway.

**Fix:** In this case, use `_simple_extractive_summary` instead of the hybrid selector.

### 23. No summary-mode selector in the UI
**`templates/index.html`**

The backend supports `simple`/`balanced`/`detailed`, but the form has no control for it, so the UI always runs in `detailed` mode (see #6). The two entry points also have different defaults: `/analyze` uses `detailed` and `/api/summarize` uses `balanced`. And `_build_explanation` accepts mode names (`technical`, `easy`) that the summary code doesn't.

### 24. Error page also shows an empty "Document Output" card
**`app.py:87`, `templates/summary.html:151`**

When no input is given, `summary.html` is rendered with only `error`. The template's fallback `docs_list` then builds an empty "Document Output" card under the error alert.

**Fix:** Wrap the loop in `{% if doc_results %}`, or redirect back to `/index` and flash the error.

### 25. Dependency and config problems
**`requirements.txt`, `README.md`**

- Missing packages: `flask`, `python-docx`, `reportlab`. The app won't start from a clean install.
- `googletrans==4.0.0-rc1` is unmaintained and pins an old `httpx`. It often conflicts with modern packages and breaks whenever Google changes its endpoint.
- NLTK's `punkt` (or `punkt_tab` on NLTK ≥ 3.8.2) data and the system `tesseract` binary must be installed separately. The README doesn't mention this.
- No versions are pinned for `torch`/`transformers`. `lang_code_to_id` has been deprecated in recent transformers versions; use `tokenizer.convert_tokens_to_ids(tag)` instead.
- The README is titled "Project: BB" and only gives Windows instructions.

### 26. GloVe file is unused, and `.gitignore` excludes it
**`glove.6B.100d.txt`, `.gitignore:8`, `.gitattributes`**

No code reads GloVe. It adds 347 MB of LFS storage and bandwidth to every clone. Also, `.gitignore` lists `glove.6B.*.txt`, but commit `fa29a8d` force-added it anyway.

**Fix:** Run `git rm --cached glove.6B.100d.txt` and remove the LFS rule. Or, if a GloVe-based summarizer is planned, document that.

### 27. Tests need model downloads and the internet
**`test_app.py`, `run_translation_test.py`**

The backend and API tests run the full pipeline. That downloads several GB of models or calls googletrans, so results depend on network, cache state and randomness (#16, #17). The assertions are only `len(...) > 0`, so they wouldn't catch #1 or #6. The `werkzeug.__version__` monkey-patch is a workaround for an old Flask/Werkzeug version mismatch.

**Fix:** Mock the model loaders and translators in unit tests. Add specific tests for: English output from non-English input (#1), BART load failure (#2), PDF export with `<&` and Kannada text (#3, #4), and filenames containing `'` (#9).

### 28. Dead code and unused files
- `static/js/app.js`, `static/css/style.css` and `static/ts.js` aren't referenced by any template.
- `OFFLINE_ONLY = False` is hardcoded, so its whole branch (`backend.py:500-528`) never runs.
- `funforfrontend_translated`, `_translate_with_mbart` and `_GT_AVAILABLE` are unused.
- `TF_ENABLE_ONEDNN_OPTS` is set twice. `_build_explanation` ignores its `original_text` parameter.
- The "explanation" is template text (*"Reason: This sentence was selected because it contains key concepts from the source."*) and doesn't reflect real selection scores.

---

## Suggested fix order
1. **#2, #1**: one-line fixes for a crash and wrong-language output.
2. **#4, #3**: PDF export is broken for the app's main languages.
3. **#6, #23**: default-mode quality.
4. **#5, #7**: long and failed documents.
5. **#9, #10, #20, #21**: frontend and security cleanup.
6. **#25**: make it installable from a clean environment.
