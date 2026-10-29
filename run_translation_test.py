"""Manual smoke test: summarize sample texts across English, Kannada, Telugu and Tamil.

Uses whatever models/services your .env enables, so the first run may download
models or call Google Translate. For the hermetic unit tests run:
    python -m unittest -v test_app

Usage:
    python run_translation_test.py            # every input language -> every output language
    python run_translation_test.py kn         # every input language -> Kannada
    python run_translation_test.py kn te      # Kannada input -> Telugu
"""
import sys

import backend

SAMPLES = {
    'en': (
        "Climate change is causing more frequent extreme weather events around the world. "
        "Heatwaves, floods and droughts are becoming longer and more intense. "
        "This affects agriculture, health, and infrastructure. "
        "Farmers face lower crop yields, while cities struggle with flooding and heat stress. "
        "Scientists say cutting greenhouse gas emissions quickly can limit the damage."
    ),
    'kn': (
        "ಹವಾಮಾನ ಬದಲಾವಣೆಯಿಂದಾಗಿ ಪ್ರಪಂಚದಾದ್ಯಂತ ತೀವ್ರ ಹವಾಮಾನ ಘಟನೆಗಳು ಹೆಚ್ಚಾಗುತ್ತಿವೆ. "
        "ಬಿಸಿಗಾಳಿ, ಪ್ರವಾಹ ಮತ್ತು ಬರಗಾಲಗಳು ಹೆಚ್ಚು ದೀರ್ಘ ಮತ್ತು ತೀವ್ರವಾಗುತ್ತಿವೆ. "
        "ಇದು ಕೃಷಿ, ಆರೋಗ್ಯ ಮತ್ತು ಮೂಲಸೌಕರ್ಯದ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ. "
        "ರೈತರು ಕಡಿಮೆ ಬೆಳೆ ಇಳುವರಿಯನ್ನು ಎದುರಿಸುತ್ತಿದ್ದಾರೆ, ನಗರಗಳು ಪ್ರವಾಹ ಮತ್ತು ಶಾಖದಿಂದ ತೊಂದರೆ ಅನುಭವಿಸುತ್ತಿವೆ. "
        "ಹಸಿರುಮನೆ ಅನಿಲಗಳ ಹೊರಸೂಸುವಿಕೆಯನ್ನು ಬೇಗನೆ ಕಡಿಮೆ ಮಾಡಿದರೆ ಹಾನಿಯನ್ನು ಮಿತಿಗೊಳಿಸಬಹುದು ಎಂದು ವಿಜ್ಞಾನಿಗಳು ಹೇಳುತ್ತಾರೆ."
    ),
    'te': (
        "వాతావరణ మార్పుల వల్ల ప్రపంచవ్యాప్తంగా తీవ్రమైన వాతావరణ సంఘటనలు తరచుగా జరుగుతున్నాయి. "
        "వడగాల్పులు, వరదలు మరియు కరువులు మరింత ఎక్కువ కాలం మరియు తీవ్రంగా మారుతున్నాయి. "
        "ఇది వ్యవసాయం, ఆరోగ్యం మరియు మౌలిక సదుపాయాలపై ప్రభావం చూపుతుంది. "
        "రైతులు తక్కువ పంట దిగుబడిని ఎదుర్కొంటున్నారు, నగరాలు వరదలు మరియు వేడితో ఇబ్బంది పడుతున్నాయి. "
        "గ్రీన్‌హౌస్ వాయువుల ఉద్గారాలను త్వరగా తగ్గిస్తే నష్టాన్ని పరిమితం చేయవచ్చని శాస్త్రవేత్తలు చెబుతున్నారు."
    ),
    'ta': (
        "காலநிலை மாற்றத்தால் உலகம் முழுவதும் தீவிர வானிலை நிகழ்வுகள் அடிக்கடி ஏற்படுகின்றன. "
        "வெப்ப அலைகள், வெள்ளம் மற்றும் வறட்சி ஆகியவை நீண்டதாகவும் தீவிரமாகவும் மாறி வருகின்றன. "
        "இது விவசாயம், சுகாதாரம் மற்றும் உள்கட்டமைப்பைப் பாதிக்கிறது. "
        "விவசாயிகள் குறைந்த பயிர் விளைச்சலை எதிர்கொள்கின்றனர், நகரங்கள் வெள்ளம் மற்றும் வெப்பத்தால் சிரமப்படுகின்றன. "
        "பசுமை இல்ல வாயு உமிழ்வை விரைவாகக் குறைத்தால் பாதிப்பைக் கட்டுப்படுத்தலாம் என்று விஞ்ஞானிகள் கூறுகின்றனர்."
    ),
}


def run(source, target):
    result = backend.summarize(SAMPLES[source], language=target)
    detected = '' if result.source_language == source else f"  (DETECTED AS {result.source_language})"
    print(f"\n=== {source} -> {target}   method: {result.method}{detected}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print(result.summary)
    return not result.warnings and result.source_language == source


def main(target=None, source=None):
    sources = [source] if source else list(SAMPLES)
    targets = [target] if target else list(SAMPLES)
    failures = [(s, t) for s in sources for t in targets if not run(s, t)]
    print(f"\n{len(sources) * len(targets) - len(failures)} passed, {len(failures)} with warnings: {failures}")


if __name__ == '__main__':
    args = sys.argv[1:3]
    if len(args) == 2:
        main(target=args[1], source=args[0])
    else:
        main(*args)
