import os
import json
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
import soundfile as sf
from vosk import Model, KaldiRecognizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import requests
import whisper

# Load Whisper model (small for speed, medium/large for accuracy)
whisper_model = whisper.load_model("small")

# gTTS supported language codes mapping
GTTS_LANG_MAP = {
    "en": "en",   # English
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "mr": "mr",   # Marathi
    "bn": "bn",   # Bengali
    "gu": "gu",   # Gujarati
    "pa": "pa",   # Punjabi
    "or": "or",   # Odia
    "ur": "ur",   # Urdu
    "fr": "fr",   # French
    "es": "es",   # Spanish
    "de": "de",   # German
    "ar": "ar",   # Arabic
    # ❌ Not directly supported: zh (Chinese), ja (Japanese), etc.
}


# ------------------------
# App Initialization
# ------------------------
app = FastAPI(title="Railway NLP System API", version="2.0")

logging.basicConfig(level=logging.INFO)

# Setup static folder for saving audio outputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------
# Vosk Model Setup (Hindi/English)
# ------------------------
vosk_model_path = os.path.join(BASE_DIR, "models", "vosk-model-small-hi-0.22")
if not os.path.exists(vosk_model_path):
    raise Exception(f"❌ Vosk model not found at {vosk_model_path}")

vosk_model = Model(vosk_model_path)

# ------------------------
# Hugging Face M2M100 Translator
# ------------------------
translator_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translator_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using M2M100 model."""
    tokenizer = translator_tokenizer
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = translator_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# ------------------------
# Railway API Setup
# ------------------------
RAILWAY_API_KEY = os.getenv("RAILWAY_API_KEY", "your_api_key_here")

def get_pnr_status(pnr: str):
    url = f"https://api.railwayapi.com/v2/pnr-status/pnr/{pnr}/apikey/{RAILWAY_API_KEY}/"
    res = requests.get(url)
    return res.json()

def get_train_status(train_no: str, date: str):
    url = f"https://api.railwayapi.com/v2/live/train/{train_no}/date/{date}/apikey/{RAILWAY_API_KEY}/"
    res = requests.get(url)
    return res.json()

# ------------------------
# Pydantic Models
# ------------------------
class TTSRequest(BaseModel):
    text: str
    lang: str = "en"

# ------------------------
# Speech-to-Text Endpoint
# ------------------------
@app.post("/speech-to-text/")
async def speech_to_text(
    file: UploadFile = File(...),
    src_lang: str = "en",
    target_lang: str = "en",
    engine: str = "whisper"  # 👈 default to whisper
):
    try:
        temp_file = os.path.join(BASE_DIR, "temp.webm")
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Convert webm -> wav
        wav_file = os.path.join(BASE_DIR, "temp.wav")
        from pydub import AudioSegment
        sound = AudioSegment.from_file(temp_file, format="webm")
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_file, format="wav")

        # --- Whisper Multilingual ---
        if engine == "whisper":
            result = whisper_model.transcribe(wav_file, language=src_lang)
            text = result.get("text", "").strip()

        # --- Vosk (fallback for Hindi/English) ---
        else:
            rec = KaldiRecognizer(vosk_model, 16000)
            with sf.SoundFile(wav_file) as f:
                while True:
                    data = f.read(4000, dtype="int16")
                    if len(data) == 0:
                        break
                    rec.AcceptWaveform(data.tobytes())
            result = json.loads(rec.Result())
            text = result.get("text", "")

        if not text:
            return {"error": "No speech detected"}

        # Translate output
        translated = translate_text(text, src_lang, target_lang)

        return {
            "original_text": text,
            "translated_text": translated,
            "src_lang": src_lang,
            "target_lang": target_lang,
            "engine": engine
        }

    except Exception as e:
        logging.error(f"❌ Speech-to-Text error: {e}")
        return {"error": str(e)}

    try:
        temp_file = os.path.join(BASE_DIR, "temp.webm")
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Convert webm -> wav
        wav_file = os.path.join(BASE_DIR, "temp.wav")
        from pydub import AudioSegment
        sound = AudioSegment.from_file(temp_file, format="webm")
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_file, format="wav")

        rec = KaldiRecognizer(vosk_model, 16000)
        with sf.SoundFile(wav_file) as f:
            while True:
                data = f.read(4000, dtype="int16")
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data.tobytes())

        result = json.loads(rec.Result())
        text = result.get("text", "")

        if not text:
            return {"error": "No speech detected"}

        translated = translate_text(text, src_lang, target_lang)

        return {
            "original_text": text,
            "translated_text": translated,
            "src_lang": src_lang,
            "target_lang": target_lang
        }

    except Exception as e:
        logging.error(f"❌ Speech-to-Text error: {e}")
        return {"error": str(e)}


# ------------------------
# Text-to-Speech Endpoint
# ------------------------
@app.post("/text-to-speech/")
async def text_to_speech(req: TTSRequest):
    if not req.text or req.text.strip() == "":
        return {"error": "No text provided for TTS"}

    try:
        lang_code = GTTS_LANG_MAP.get(req.lang, "en")  # fallback to English if not supported

        if req.lang not in GTTS_LANG_MAP:
            logging.warning(f"⚠️ gTTS does not support '{req.lang}', falling back to English")

        tts = gTTS(text=req.text, lang=lang_code)
        filename = f"output_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(STATIC_DIR, filename)
        tts.save(output_path)

        return {"audio_url": f"/static/{filename}", "used_lang": lang_code}

    except Exception as e:
        logging.error(f"❌ TTS Error: {e}")
        return {"error": str(e)}


# ------------------------
# Railway Query Endpoints
# ------------------------
@app.get("/pnr-status/")
async def pnr_status(pnr: str = Query(...)):
    return get_pnr_status(pnr)

@app.get("/train-status/")
async def train_status(train_no: str = Query(...), date: str = Query(...)):
    return get_train_status(train_no, date)

# ------------------------
# Root + UI
# ------------------------
@app.get("/")
async def home():
    return {"message": "🚉 Railway NLP System API is running"}

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    file_path = os.path.join(BASE_DIR, "frontend", "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading UI: {e}</h1>", status_code=500)
