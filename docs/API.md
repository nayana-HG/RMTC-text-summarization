# HTTP API

Base URL: `http://127.0.0.1:5000` (see `HOST`/`PORT` in `.env`).

When `AUTH_ENABLED=true`, every `/api/*` route needs a logged-in session cookie
(log in through `POST /login`). Requests without one get `401`.

Error responses always have the shape:

```json
{ "status": "error", "message": "Human-readable reason" }
```

---

## `POST /api/summarize`

Summarize pasted text and/or uploaded files. Each input is summarized separately.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | — | Raw text to summarize. |
| `files` (or `pdf_file`) | file, repeatable | — | `.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`. |
| `language` | string | `DEFAULT_LANGUAGE` | Output language: `en kn hi bn ta te ml mr gu pa ur es fr de`. Unknown codes fall back to `en`. |
| `summary_mode` | string | `balanced` | `simple`, `balanced` or `detailed`. |
| `explain` | bool | `true` | Include the explanation. |

At least one of `text` or `files` is required.

**Response:** `200`

```json
{
  "status": "success",
  "language": "hi",
  "language_name": "Hindi (हिंदी)",
  "doc_results": [
    {
      "doc_name": "report.pdf",
      "summary": "…",
      "original_text": "…",
      "explanation": "…",
      "keywords": ["…"],
      "keywords_language": "hi",
      "source_language": "en",
      "language": "hi",
      "method": "hybrid",
      "warnings": []
    }
  ],
  "errors": ["scan.pdf has no extractable text (it may be a scanned PDF). …"],
  "summary": "…", "original_text": "…", "explanation": "…", "keywords": ["…"]
}
```

- `doc_results`: one entry per readable input. Uploaded files come first, then `text` as `"Raw Text Input"`.
- `method`: `hybrid` (extractive + BART rewrite) or `extractive` (no rewrite: models unavailable, or input too short).
- `keywords_language`: `en` if the keywords couldn't be translated.
- `warnings`: problems that didn't stop the summary, e.g. a translation that wasn't available.
- `errors`: files that couldn't be read.
- The top-level `summary`, `original_text`, `explanation` and `keywords` fields repeat the first document's values, for older clients.

**Errors:** `400` no usable input (with `errors`), `401` not logged in, `413` upload
too large, `500` unexpected failure.

```bash
curl -F text="$(cat article.txt)" -F language=kn -F summary_mode=detailed \
     http://127.0.0.1:5000/api/summarize
curl -F files=@report.pdf -F files=@notes.docx -F language=en \
     http://127.0.0.1:5000/api/summarize
```

---

## `POST /api/download/pdf` and `POST /api/download/docx`

Render a result as a PDF or Word document.

**Request:** `application/json`

```json
{
  "summary": "…",
  "original_text": "…",
  "language_name": "Kannada (ಕನ್ನಡ)",
  "keywords": ["…"],
  "explanation": "…"
}
```

All fields are optional strings, except `keywords`, which is a list of strings.

**Response:** `200`, the file as an attachment (`summary.pdf` / `summary.docx`).

**Errors:** `501` the export library isn't installed (reportlab / python-docx),
`500` rendering failed.

---

## `POST /api/tts`

Synthesize speech for a summary with Google TTS.

**Request:** `application/json`: `{ "text": "…", "language": "kn" }`

**Response:** `200`, `audio/mpeg`.

**Errors:** `400` empty text, `501` TTS disabled (`ENABLE_ONLINE_TTS=false`),
gTTS not installed or language unsupported, `502` Google TTS failed. On any error the
web UI falls back to the browser's built-in speech synthesis.

---

## Pages

| Route | Description |
|---|---|
| `GET /`, `GET /home` | Landing page |
| `GET /index` | Input form |
| `POST /analyze` | Form submission, renders the results page (same fields as `/api/summarize`, with `rawtext` and `pdf_file`) |
| `GET/POST /login`, `GET /logout` | Only active when `AUTH_ENABLED=true` |
