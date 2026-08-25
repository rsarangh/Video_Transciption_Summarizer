import os
from http.cookiejar import MozillaCookieJar
import requests
from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import google.generativeai as genai
from pytube import extract

app = Flask(__name__)

# --- AI CONFIGURATION ---
# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemini Backup Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "45"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "15000"))
MAX_GEMINI_MODELS_TO_TRY = int(os.getenv("MAX_GEMINI_MODELS_TO_TRY", "6"))
YOUTUBE_COOKIES_PATH = os.getenv("YOUTUBE_COOKIES_PATH", "")

@app.route('/', methods=['GET'])
def root_get():
    return jsonify({
        'status': 'ok',
        'message': 'Use POST / with form field video_id to summarize a YouTube video transcript.'
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

SYSTEM_PROMPT = """
You are an expert multilingual text summarization assistant.
Generate a clear, accurate summary of the provided text.
- Capture main ideas and key facts.
- Keep the summary within 150 words.
- If the text is not in English, provide the summary in the original language AND an English translation.
- Return ONLY the summary text.
"""

def summarize_with_openai(text):
    """Primary Summarization Engine"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=AI_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this video transcript:\n\n{text}"}
        ],
        temperature=0.3,
        max_tokens=600
    )
    return response.choices[0].message.content

def summarize_with_gemini(text):
    """Backup Summarization Engine"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=GEMINI_API_KEY)
    candidate_models = []

    if GEMINI_MODEL:
        candidate_models.append(GEMINI_MODEL)

    candidate_models.extend([
        "gemini-3.6-flash",
        "gemini-3.1-pro-preview",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    ])

    # Add currently available models from API discovery when possible.
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model_name = m.name.replace("models/", "")
                lowered = model_name.lower()
                # Skip non-text or specialized models that are noisy for this endpoint.
                if "tts" in lowered or "audio" in lowered:
                    continue
                if model_name not in candidate_models:
                    candidate_models.append(model_name)
    except Exception as e:
        print(f"Gemini model discovery failed: {e}")

    # Bound retries to avoid very long request times.
    candidate_models = candidate_models[:MAX_GEMINI_MODELS_TO_TRY]

    def extract_response_text(response):
        """Get plain text from Gemini responses, including multi-part candidates."""
        text = getattr(response, "text", None)
        if text:
            return text

        pieces = []
        candidates = getattr(response, "candidates", []) or []
        for c in candidates:
            content = getattr(c, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", []) or []
            for p in parts:
                part_text = getattr(p, "text", None)
                if part_text:
                    pieces.append(part_text)

        if pieces:
            return "\n".join(pieces)

        raise RuntimeError("Gemini returned no text content")

    last_error = None
    for model_name in candidate_models:
        try:
            print(f"Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(model_name)
            # Gemini uses a single prompt, so we combine system instructions
            combined_prompt = f"{SYSTEM_PROMPT}\n\nTranscript to summarize:\n{text}"
            response = model.generate_content(
                combined_prompt,
                request_options={"timeout": AI_TIMEOUT_SECONDS}
            )
            return extract_response_text(response)
        except Exception as e:
            last_error = e
            print(f"Gemini model failed ({model_name}): {e}")

    raise RuntimeError(f"No Gemini model could generate content. Last error: {last_error}")

def build_cookie_session():
    """Load a requests session with YouTube cookies to bypass cloud IP blocking."""
    if not YOUTUBE_COOKIES_PATH or not os.path.exists(YOUTUBE_COOKIES_PATH):
        return None

    jar = MozillaCookieJar(YOUTUBE_COOKIES_PATH)
    jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = jar
    return session

def get_transcript_text(video_id):
    """Fetch transcript text across youtube_transcript_api versions."""
    preferred_languages = ['en', 'hi', 'ml', 'es']
    cookie_session = build_cookie_session()

    if hasattr(YouTubeTranscriptApi, 'get_transcript'):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
        except Exception:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([snippet['text'] for snippet in transcript_list])

    api = YouTubeTranscriptApi(http_client=cookie_session) if cookie_session else YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id, languages=preferred_languages)
    except Exception:
        transcript = api.fetch(video_id)
    return " ".join([snippet.text for snippet in transcript])

@app.route('/', methods=['POST'])
def handle_summarization():
    try:
        video_url = request.form.get('video_id')
        if not video_url:
            return jsonify({'summary': 'Error: No URL provided', 'transcription': ''}), 400

        # 1. Extract Video ID
        try:
            video_id = extract.video_id(video_url)
        except:
            video_id = video_url

        # 2. Extract Transcript
        try:
            full_text = get_transcript_text(video_id)
        except Exception as e:
            print(f"Transcript fetch failed for video_id={video_id}: {e}")
            return jsonify({'summary': "Captions are disabled for this video.", 'transcription': "", 'transcript_error': str(e)})

        # 3. Summarization with Fallback Logic
        text_for_summary = full_text[:MAX_TRANSCRIPT_CHARS]
        summary_result = ""
        engine_used = ""
        openai_error = ""
        gemini_error = ""
        try:
            # Try OpenAI First
            print("Attempting OpenAI summarization...")
            summary_result = summarize_with_openai(text_for_summary)
            engine_used = "OpenAI"
            print("Summary generated by: OpenAI")
        except Exception as e:
            openai_error = str(e)
            print(f"OpenAI failed: {e}. Falling back to Gemini...")
            try:
                # If OpenAI fails, try Gemini
                summary_result = summarize_with_gemini(text_for_summary)
                engine_used = "Gemini"
                print("Summary generated by: Gemini")
            except Exception as e2:
                gemini_error = str(e2)
                print(f"Gemini also failed: {e2}")
                return jsonify({
                    'summary': "Both AI engines are currently busy. Please try again later.",
                    'transcription': full_text,
                    'engine_used': '',
                    'engine_errors': {
                        'openai': openai_error,
                        'gemini': gemini_error
                    }
                }), 500

        print(f"Request complete. Engine used: {engine_used}")

        return jsonify({
            'summary': summary_result,
            'transcription': full_text,
            'engine_used': engine_used
        })

    except Exception as e:
        return jsonify({'summary': "Server Error: Unable to process request.", 'transcription': ""}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)