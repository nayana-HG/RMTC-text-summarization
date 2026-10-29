"""Unit tests. Hermetic: no model downloads, no network.

Run with:  python -m unittest -v test_app
"""
import io
import os
import re
import unittest
from unittest import mock

# Configure before importing the app: no ML models, no online services.
os.environ.update({
    'LIGHTWEIGHT_MODE': 'true',
    'ENABLE_ONLINE_TRANSLATION': 'false',
    'ENABLE_ONLINE_TTS': 'false',
    'FLASK_SECRET_KEY': 'test-secret',
    'OPEN_BROWSER': 'false',
})

from werkzeug.security import generate_password_hash  # noqa: E402

import backend  # noqa: E402
import config  # noqa: E402
import exporter  # noqa: E402
from app import app  # noqa: E402

ENGLISH_TEXT = (
    "Artificial intelligence is transforming modern technology. "
    "It empowers machine learning algorithms to process complex data. "
    "Hospitals use these systems to detect diseases earlier. "
    "Banks rely on them to flag fraudulent transactions in real time. "
    "Teachers use adaptive software to personalise lessons for students. "
    "Critics warn about bias in training data and the loss of jobs. "
    "Governments are drafting rules to make these systems transparent. "
    "Researchers continue to improve the accuracy of machine learning models."
)
HINDI_TEXT = (
    "कृत्रिम बुद्धिमत्ता आधुनिक तकनीक को बदल रही है। यह मशीन लर्निंग एल्गोरिदम को जटिल डेटा संसाधित करने में "
    "सक्षम बनाती है। अस्पताल इन प्रणालियों का उपयोग बीमारियों का जल्दी पता लगाने के लिए करते हैं। बैंक धोखाधड़ी "
    "वाले लेनदेन को पकड़ने के लिए इन पर निर्भर हैं।"
)


def fake_translate(text, src, tgt):
    """Stand-in translator: tags text with the target language."""
    if not text or src == tgt:
        return text
    if tgt == 'en':
        return ENGLISH_TEXT
    return '\n\n'.join(f"[{tgt}] {part}" for part in text.split('\n\n'))


class BackendTests(unittest.TestCase):

    def test_english_summary(self):
        r = backend.summarize(ENGLISH_TEXT, language='en', explain=True)
        self.assertTrue(r.summary)
        self.assertEqual(r.source_language, 'en')
        self.assertEqual(r.warnings, [])
        self.assertIn('intelligence', ' '.join(r.keywords) + ' ' + r.summary.lower())
        self.assertTrue(r.explanation)

    def test_summary_is_deterministic(self):
        first = backend.summarize(ENGLISH_TEXT, language='en').summary
        for _ in range(5):
            self.assertEqual(backend.summarize(ENGLISH_TEXT, language='en').summary, first)

    def test_non_english_input_with_english_output_stays_english(self):
        # Bug #1: the English summary used to be translated back to the source language.
        with mock.patch.object(backend, '_translate_text', side_effect=fake_translate) as tr:
            r = backend.summarize(HINDI_TEXT, language='en')
        self.assertEqual(r.source_language, 'hi')
        self.assertNotIn('[hi]', r.summary)
        targets = [c.args[2] for c in tr.call_args_list]
        self.assertEqual(targets, ['en'])

    def test_translation_to_target_language(self):
        with mock.patch.object(backend, '_translate_text', side_effect=fake_translate):
            r = backend.summarize(ENGLISH_TEXT, language='kn', explain=True)
        self.assertTrue(r.summary.startswith('[kn]'))
        self.assertTrue(r.explanation.startswith('[kn]'))
        # Isolated words translate badly, so keywords stay English (and are labelled so).
        self.assertEqual(r.keywords_language, 'en')
        self.assertFalse(any(k.startswith('[kn]') for k in r.keywords))

    def test_missing_translation_gives_warning_not_garbage(self):
        with mock.patch.object(backend, '_translate_text', return_value=None):
            r = backend.summarize(ENGLISH_TEXT, language='kn')
        self.assertTrue(r.summary)
        self.assertNotIn('[Translation', r.summary)
        self.assertEqual(len(r.warnings), 1)

    def test_abstractive_failure_falls_back_to_extractive(self):
        # Bug #2: an exception here used to raise UnboundLocalError.
        with mock.patch.object(backend, '_abstractive_rewrite', side_effect=RuntimeError('CUDA OOM')):
            r = backend.summarize(ENGLISH_TEXT, language='en')
        self.assertTrue(r.summary)
        self.assertEqual(r.method, 'extractive')

    def test_legacy_tuple_api(self):
        summary, original, explanation, keywords = backend.funforfrontend(ENGLISH_TEXT, language='en', explain=True)
        self.assertTrue(summary)
        self.assertEqual(original, ENGLISH_TEXT)
        self.assertTrue(explanation)
        self.assertTrue(keywords)

    def test_generation_lengths_never_exceed_input(self):
        # Bug #6: detailed mode forced 120+ output tokens even for 2-sentence inputs.
        detailed = backend._mode_settings('detailed', 10)
        self.assertIsNone(backend._generation_lengths(25, detailed))
        min_len, max_len = backend._generation_lengths(60, detailed)
        self.assertLessEqual(max_len, 60)
        self.assertLess(min_len, max_len)
        self.assertEqual(backend._generation_lengths(2000, detailed), (120, 400))

    def test_incomplete_generated_sentence_is_dropped(self):
        drop = backend._drop_incomplete_sentence
        self.assertEqual(drop("One. Two is complete."), "One. Two is complete.")
        self.assertEqual(drop("One is complete. Scientists say cutting"), "One is complete.")
        self.assertEqual(drop('He said "yes." Then he'), 'He said "yes."')
        self.assertIsNone(drop("no sentence end at all"))

    def test_chunking_respects_limit_and_paragraphs(self):
        # Bug #5: long documents used to be truncated to one model input.
        text = ' '.join(f"Sentence number {i} is here." for i in range(200)) + '\n\nSecond paragraph.'
        chunks = backend._chunk_text(text, len, 300)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertIn('\n\n', chunks)
        self.assertEqual(backend._join_chunks(chunks).replace('\n\n', ' '), text.replace('\n\n', ' '))

    def test_chunked_translation_covers_whole_text(self):
        text = ' '.join(f"Sentence {i}." for i in range(500))
        upper = lambda chunks: [c.upper() for c in chunks]  # noqa: E731
        self.assertEqual(backend._translate_chunks(text, upper, len, 200), text.upper())
        self.assertEqual(backend._translate_chunks(text, upper, len, 200, batch_size=8), text.upper())
        self.assertIsNone(backend._translate_chunks(text, lambda chunks: [None] * len(chunks), len, 200))

    def test_model_translation_is_one_sentence_per_input(self):
        # Marian drops sentences when given several at once, so each must be its own input.
        seen = []

        def batch(chunks):
            seen.extend(chunks)
            return chunks

        text = "First sentence here. Second one follows. Third ends it.\n\nNew paragraph."
        out = backend._translate_chunks(text, batch, len, 400, per_sentence=True, batch_size=8)
        # Batches are ordered by length to reduce padding, so compare as sets.
        self.assertEqual(sorted(seen), sorted(["First sentence here.", "Second one follows.", "Third ends it.",
                                               "New paragraph."]))
        self.assertEqual(out, text)

    def test_language_detection(self):
        self.assertEqual(backend._detect_language("Short English text."), 'en')
        self.assertEqual(backend._detect_language(HINDI_TEXT), 'hi')
        self.assertEqual(backend._detect_language("12345 !!!"), 'en')

    def test_dravidian_script_detection(self):
        # Mixed with English words, which can push langdetect below its threshold.
        self.assertEqual(backend._detect_language("ರೈತರು AI tools use ಮಾಡುತ್ತಿದ್ದಾರೆ, Government scheme ಘೋಷಿಸಿದೆ."), 'kn')
        self.assertEqual(backend._detect_language("రైతులు AI tools వాడుతున్నారు. Government scheme ప్రకటించింది."), 'te')
        self.assertEqual(backend._detect_language("விவசாயிகள் AI tools பயன்படுத்துகின்றனர். Government scheme அறிவித்தது."), 'ta')

    def test_dravidian_languages_prefer_nllb_and_skip_mul_en(self):
        # opus-mt-mul-en invents unrelated English for Kannada/Telugu/Tamil.
        with mock.patch.object(backend, 'MULTILINGUAL_MODEL', 'facebook/nllb-200-distilled-600M'):
            for lang in ('kn', 'te', 'ta'):
                self.assertIs(backend._translation_backends(lang, 'en')[0], backend._translate_multilingual)
                self.assertIs(backend._translation_backends('en', lang)[0], backend._translate_multilingual)
                self.assertIsNone(backend._translate_marian("ಪಠ್ಯ.", lang, 'en'))
        with mock.patch.object(backend, 'MULTILINGUAL_MODEL', 'facebook/mbart-large-50-many-to-many-mmt'):
            self.assertNotIn(backend._translate_multilingual, backend._translation_backends('te', 'en'))
        self.assertIs(backend._translation_backends('en', 'es')[0], backend._translate_marian)

    def test_translation_cache_translates_each_sentence_once(self):
        seen = []

        def batch(chunks):
            seen.extend(chunks)
            return [f"<{c}>" for c in chunks]

        with mock.patch.object(backend, '_TRANSLATION_CACHE', backend.collections.OrderedDict()):
            key = ('model', 'en', 'kn')
            first = backend._translate_chunks("One. Two. One.", batch, len, 400, per_sentence=True,
                                              batch_size=8, cache_key=key)
            # The summary repeated inside the explanation is not translated again.
            second = backend._translate_chunks("Key points:\nTwo.\n\nSummary:\nOne. Two.", batch, len, 400,
                                               per_sentence=True, batch_size=8, cache_key=key)
        self.assertEqual(first, "<One.> <Two.> <One.>")
        self.assertEqual(second, "<Key points:>\n<Two.>\n\n<Summary:>\n<One.> <Two.>")
        self.assertEqual(sorted(seen), ["Key points:", "One.", "Summary:", "Two."])

    def test_contact_details_are_not_translated(self):
        text = ("Mail a.b@example.com or call +91 98765 43210. Code at https://github.com/x/y. "
                "Follow @someone.")
        seen = []

        def fake_backend(t, src, tgt):
            seen.append(t)
            return t.upper()  # a translator that would mangle anything it touches

        with mock.patch.object(backend, '_translation_backends', return_value=(fake_backend,)):
            out = backend._translate_text(text, 'en', 'kn')
        self.assertNotIn('@', seen[0])
        self.assertEqual(out, "MAIL a.b@example.com OR CALL +91 98765 43210. CODE AT https://github.com/x/y. "
                              "FOLLOW @someone.")

    def test_lost_placeholder_falls_back_to_unmasked_translation(self):
        calls = []

        def drops_placeholders(t, src, tgt):
            calls.append(t)
            return re.sub(r'\[\[\d+\]\]', '', t)

        with mock.patch.object(backend, '_translation_backends', return_value=(drops_placeholders,)):
            out = backend._translate_text("Write to a@b.co today.", 'en', 'kn')
        self.assertEqual(calls, ["Write to [[0]] today.", "Write to a@b.co today."])
        self.assertEqual(out, "Write to a@b.co today.")

    def test_long_non_english_document_translates_only_central_sentences(self):
        sentences = [f"ವಾಕ್ಯ ಸಂಖ್ಯೆ {i} ವಿಷಯ{i % 7} ಬಗ್ಗೆ ಹೇಳುತ್ತದೆ." for i in range(200)]
        translated = []

        def fake(text, src, tgt):
            if tgt == 'en':
                translated.append(text)
                return ENGLISH_TEXT
            return text

        with mock.patch.object(backend, '_translate_text', side_effect=fake):
            backend.summarize(' '.join(sentences), language='en')
        self.assertEqual(len(backend._split_sentences(translated[0])), backend._PRESELECT_SENTENCES)

    def test_online_translation_retries_once_when_rate_limited(self):
        if backend.TooManyRequests is None:
            self.skipTest('deep-translator not installed')
        translator = mock.Mock()
        translator.translate.side_effect = [backend.TooManyRequests(), 'translated.']
        with mock.patch.object(config, 'ENABLE_ONLINE_TRANSLATION', True), \
                mock.patch.object(backend, 'GoogleTranslator', return_value=translator), \
                mock.patch.object(backend.time, 'sleep'):
            self.assertEqual(backend._translate_online("Text.", 'en', 'kn'), 'translated.')
        self.assertEqual(translator.translate.call_count, 2)

    def test_keywords_unicode_and_stopwords(self):
        self.assertEqual(backend._extract_keywords("will will will also also their"), [])
        self.assertIn('café', backend._extract_keywords("café café résumé"))

    def test_mode_normalisation(self):
        self.assertEqual(backend.normalize_mode('Detailed'), 'detailed')
        self.assertEqual(backend.normalize_mode('easy'), 'simple')
        self.assertEqual(backend.normalize_mode('nonsense'), 'balanced')

    def test_extract_text_errors_are_not_content(self):
        # Bug #7: error strings used to be summarized as document text.
        from werkzeug.datastructures import FileStorage
        empty = FileStorage(io.BytesIO(b''), filename='empty.txt')
        with self.assertRaises(backend.ExtractionError):
            backend.extract_text(empty)
        broken_pdf = FileStorage(io.BytesIO(b'not a pdf'), filename='broken.pdf')
        with self.assertRaises(backend.ExtractionError):
            backend.extract_text(broken_pdf)
        txt = FileStorage(io.BytesIO('hello world'.encode()), filename='a.txt')
        self.assertEqual(backend.extract_text(txt), 'hello world')


class ExporterTests(unittest.TestCase):

    def test_pdf_with_markup_characters_and_indic_text(self):
        # Bugs #3 and #4: '<' and '&' crashed reportlab; Indic text needs Unicode fonts.
        pdf = exporter.generate_pdf_bytes(
            summary="ಹವಾಮಾನ ಬದಲಾವಣೆ (CO2 < 5% & rising) <b>not markup</b>",
            original_text="Profit < 5% & rising",
            language_name="Kannada (ಕನ್ನಡ)",
            keywords=["a<b", "ಹವಾಮಾನ"],
            explanation="हिंदी व्याख्या & more",
        )
        self.assertTrue(pdf.startswith(b'%PDF'))

    def test_docx(self):
        data = exporter.generate_docx_bytes(summary="Summary", original_text="Original",
                                            language_name="Hindi", keywords=["k"], explanation="E")
        self.assertTrue(data.startswith(b'PK'))  # .docx is a zip archive

    def test_markup_wraps_scripts_in_fonts(self):
        missing = set()
        markup = exporter._markup("Kannada (ಕನ್ನಡ) & x<y", missing)
        self.assertIn('&amp;', markup)
        self.assertIn('x&lt;y', markup)
        if not missing:
            # The whole word, including its brackets, uses one font (needed for shaping).
            self.assertIn('<font name="RMTC-kannada">(ಕನ್ನಡ)', markup)

    def test_missing_library_raises(self):
        # Bug #8: used to return an empty .docx / a text file labelled as PDF.
        with mock.patch.object(exporter, '_DOCX_AVAILABLE', False):
            with self.assertRaises(exporter.ExportUnavailable):
                exporter.generate_docx_bytes("x")
        with mock.patch.object(exporter, '_REPORTLAB_AVAILABLE', False):
            with self.assertRaises(exporter.ExportUnavailable):
                exporter.generate_pdf_bytes("x")


class AppTests(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_api_summarize(self):
        res = self.client.post('/api/summarize', data={
            'text': ENGLISH_TEXT, 'language': 'en', 'summary_mode': 'balanced', 'explain': 'true'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertTrue(data['summary'])
        self.assertEqual(data['doc_results'][0]['doc_name'], 'Raw Text Input')

    def test_api_summarize_requires_input(self):
        res = self.client.post('/api/summarize', data={'language': 'en'})
        self.assertEqual(res.status_code, 400)

    def test_api_unknown_language_falls_back_to_english(self):
        res = self.client.post('/api/summarize', data={'text': ENGLISH_TEXT, 'language': 'xx'})
        self.assertEqual(res.get_json()['language'], 'en')

    def test_unreadable_file_is_reported(self):
        res = self.client.post('/api/summarize', data={
            'text': ENGLISH_TEXT, 'language': 'en',
            'files': (io.BytesIO(b'not a pdf'), 'broken.pdf')}, content_type='multipart/form-data')
        data = res.get_json()
        self.assertEqual(len(data['doc_results']), 1)
        self.assertEqual(len(data['errors']), 1)
        self.assertIn('broken.pdf', data['errors'][0])

    def test_analyze_page_escapes_filenames(self):
        # Bug #9: filenames were injected into onclick JavaScript.
        name = "x');alert(1);//.txt"
        res = self.client.post('/analyze', data={
            'language': 'en', 'summary_mode': 'simple',
            'pdf_file': (io.BytesIO(ENGLISH_TEXT.encode()), name)}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertNotIn("alert(1);//", html.split('<script id="doc-data"')[0].split('<body>')[-1])
        self.assertIn('onclick="downloadPDF(0)"', html)

    def test_analyze_without_input_shows_error_only(self):
        # Bug #24: an empty "Document Output" card used to appear under the error.
        html = self.client.post('/analyze', data={'language': 'en'}).get_data(as_text=True)
        self.assertIn('alert-danger', html)
        self.assertNotIn('doc-card-block', html)

    def test_index_has_mode_selector(self):
        html = self.client.get('/index').get_data(as_text=True)
        self.assertIn('name="summary_mode"', html)

    def test_download_endpoints(self):
        payload = {'summary': 'Test < & summary', 'original_text': 'Original', 'language_name': 'Hindi',
                   'keywords': ['test'], 'explanation': 'Explanation'}
        res_pdf = self.client.post('/api/download/pdf', json=payload)
        self.assertEqual(res_pdf.status_code, 200)
        self.assertTrue(res_pdf.data.startswith(b'%PDF'))
        res_docx = self.client.post('/api/download/docx', json=payload)
        self.assertEqual(res_docx.status_code, 200)
        self.assertTrue(res_docx.data.startswith(b'PK'))

    def test_download_returns_501_when_library_missing(self):
        with mock.patch.object(exporter, '_REPORTLAB_AVAILABLE', False):
            res = self.client.post('/api/download/pdf', json={'summary': 'x'})
        self.assertEqual(res.status_code, 501)

    def test_tts_disabled_returns_501(self):
        # Bug #11: /api/tts used to be missing (404).
        res = self.client.post('/api/tts', json={'text': 'hello', 'language': 'en'})
        self.assertEqual(res.status_code, 501)

    def test_upload_size_limit(self):
        with mock.patch.dict(app.config, {'MAX_CONTENT_LENGTH': 1024}):
            res = self.client.post('/api/summarize', data={
                'files': (io.BytesIO(b'x' * 4096), 'big.txt')}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 413)

    def test_auth(self):
        # Bug #20: login used to be hardcoded and never enforced.
        with mock.patch.multiple(config, AUTH_ENABLED=True, APP_USERNAME='alice',
                                 APP_PASSWORD_HASH=generate_password_hash('s3cret')):
            self.assertEqual(self.client.get('/index').status_code, 302)
            self.assertEqual(self.client.post('/api/summarize', data={'text': 'x'}).status_code, 401)
            bad = self.client.post('/login', data={'username': 'alice', 'password': 'wrong'})
            self.assertIn('Invalid credentials', bad.get_data(as_text=True))
            evil = self.client.post('/login?next=//evil.example', data={'username': 'alice', 'password': 's3cret'})
            self.assertTrue(evil.headers['Location'].endswith('/index'))
            self.assertEqual(self.client.get('/index').status_code, 200)

    def test_login_redirects_when_auth_disabled(self):
        self.assertEqual(self.client.get('/login').status_code, 302)


if __name__ == '__main__':
    unittest.main()
