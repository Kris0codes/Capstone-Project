import os
import json
import uuid
import logging
import base64
import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Form
from fastapi import Request

import re
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# --- Google Cloud Imports ---
import google.generativeai as genai
from google.cloud import speech, texttospeech
from pydub import AudioSegment
from geopy.distance import geodesic
import math

# Store user locations temporarily (keyed by session or IP)
USER_LOCATION_CACHE: Dict[str, Dict[str, float]] = {}


load_dotenv()

# Set seed for consistent language detection
DetectorFactory.seed = 0

SYSTEM_INSTRUCTION = (
    "You are RailInfo Assistant, a smart, friendly, and conversational AI specialized in Indian Railways. "
    "Your tone should be warm, helpful, and conversational - like a knowledgeable friend. "
    "When a user asks a question that requires real-time data (like PNR status, live train status, seat availability, train schedule, or searching trains), "
    "you MUST use the provided tools. If a tool call is successful, use the data to generate a concise, human-readable response. "
    
    "For casual conversation, be engaging and friendly. You can discuss: "
    "- Current date, time, day, year "
    "- Indian railway history, facts, and heritage "
    "- General information about India, culture, and travel tips "
    "- Weather, greetings, and general knowledge "
    "- Railway safety tips and travel advice "
    
    "Remember conversation context and be naturally conversational. "
    "Use emojis occasionally to make the conversation more friendly. "
    "If someone greets you, respond warmly and ask how you can help with their railway journey. "
    
    "Do not invent information. If a tool fails, inform the user that the real-time API failed. "
    "The user's language is provided by the `lang` parameter, respond directly in that language."
)

# Casual chat responses for common queries (fallback when tools aren't needed)
CASUAL_RESPONSES = {
    "date": lambda lang: f"📅 Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}",
    "time": lambda lang: f"⏰ Current time is {datetime.datetime.now().strftime('%I:%M %p')}",
    "day": lambda lang: f"🗓️ Today is {datetime.datetime.now().strftime('%A')}",
    "year": lambda lang: f"🎉 We're in the year {datetime.datetime.now().strftime('%Y')}",
}

# Indian Railway Facts for casual conversation
INDIAN_RAILWAY_FACTS = [
    "🚂 Indian Railways is the 4th largest railway network in the world by size!",
    "🇮🇳 The first passenger train in India ran between Bombay (Mumbai) and Thane on April 16, 1853.",
    "🎯 Indian Railways operates more than 13,000 passenger trains daily.",
    "👥 It carries over 23 million passengers daily – that's almost the population of Australia!",
    "🌉 The Chenab Bridge in Jammu & Kashmir is the world's highest railway bridge.",
    "🍛 Indian Railways has the largest railway kitchen in the world at the New Delhi railway station.",
    "⚡ The Vande Bharat Express is India's first indigenously built semi-high speed train.",
    "🏔️ The Darjeeling Himalayan Railway is a UNESCO World Heritage Site.",
]


async def detect_language(text: str) -> str:
    """
    Detect the language of the given text and map to supported language codes.
    Returns the detected language code or 'en' as default.
    """
    try:
        # Common words that might cause false detection
        railway_terms = {'pnr', 'train', 'status', 'schedule', 'seat', 'availability', 
                        'station', 'railway', 'number', 'time', 'date'}
        
        # Clean the text for better detection
        clean_text = ' '.join([word for word in text.split() if word.lower() not in railway_terms])
        
        if len(clean_text.strip()) < 3:  # Too short for reliable detection
            return 'en'
            
        detected_lang = detect(clean_text)
        
        # Map detected language to our supported codes
        lang_mapping = {
            'hi': 'hi',  # Hindi
            'ta': 'ta',  # Tamil
            'te': 'te',  # Telugu
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'mr': 'mr',  # Marathi
            'bn': 'bn',  # Bengali
            'gu': 'gu',  # Gujarati
            'pa': 'pa',  # Punjabi
            'ur': 'ur',  # Urdu
            'fr': 'fr',  # French
            'es': 'es',  # Spanish
            'de': 'de',  # German
            'ar': 'ar',  # Arabic
        }
        
        return lang_mapping.get(detected_lang, 'en')
        
    except Exception as e:
        logging.warning(f"Language detection failed: {e}, defaulting to English")
        return 'en'

# ------------------------
# Configuration & Setup
# ------------------------

# Environment variables setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") # Assumed necessary for railway APIs
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
RAILWAY_RAPIDAPI_HOST = "irctc1.p.rapidapi.com"

# System Instruction for Gemini
SYSTEM_INSTRUCTION = (
    "You are RailInfo Assistant, a smart, conversational AI specialized in Indian Railways. "
    "Your tone must be helpful, friendly, and brief. "
    "When a user asks a question that requires real-time data (like PNR status, live train status, seat availability, train schedule, or searching trains), "
    "you MUST use the provided tools. If a tool call is successful, use the data to generate a concise, human-readable response. "
    "If the query is a general conversation or a greeting, respond without using a tool. "
    "Do not invent information. If a tool fails, inform the user that the real-time API failed. "
    "The user's language is provided by the `lang` parameter, respond directly in that language."
)

# Initialize Google Generative AI (Gemini)
# Model is configured later after RAILWAY_TOOLS is defined.
model = None 

# Google STT & TTS Language Codes Mapping
GCLOUD_LANG_MAP_STT = {
    "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN",
    "gu": "gu-IN", "pa": "pa-IN", "ur": "ur-IN",
    "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "ar": "ar-SA"
}
GCLOUD_LANG_MAP_TTS = {
    "en": "en-US", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN",
}

# ------------------------
# App Initialization & Utility
# ------------------------

app = FastAPI(title="Refactored Railway NLP System API", version="4.0")

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add this after your other environment variable setup
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
else:
    key_file_path = os.path.join(BASE_DIR, "workshop-bmsit-c0e200f01f34.json")
    if os.path.exists(key_file_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path
        logging.info(f"✅ Using Google Cloud credentials from: {key_file_path}")
    else:
        logging.warning("❌ Google Cloud credentials file not found. Speech services may not work.")

# Helper function to call the railway RapidAPI
async def call_rapidapi(endpoint: str, params: Dict[str, Union[str, int]]):
    """Generic function to call a railway information API."""
    if not RAPIDAPI_KEY:
        logging.error("❌ RAPIDAPI_KEY not set.")
        return {"error": "Server configuration error: Railway API key missing."}

    url = f"https://{RAILWAY_RAPIDAPI_HOST}{endpoint}"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAILWAY_RAPIDAPI_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # **FIX: The API uses "status": True to indicate success, not "response_code".**
        if data.get("status") is True and "data" in data:
            # This is a successful response
            return {"status": "success", "data": data}
        
        # If 'status' is not True or 'data' is missing, it's an error.
        error_msg = f"External Railway API Error: {data.get('message', 'API returned an error or unexpected format')}"
        logging.warning(f"❌ RapidAPI non-success: {error_msg} | FullData: {data}")
        return {"error": error_msg, "data": data}
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ RapidAPI request error for {endpoint}: {e}")
        return {"error": f"Failed to connect to the railway information service: {e}"}

# ------------------------
# Railway API Tool Functions
# ------------------------

async def get_pnr_status(pnr_number: str) -> Dict[str, Any]:
    return await call_rapidapi("/api/v3/getPNRStatus", {"pnrNumber": pnr_number})


async def get_live_train_status(train_number: str, start_day: int = 0) -> Dict[str, Any]:
    """
    Retrieves the live running status of a train.
    start_day: 0 = Today, 1 = Yesterday, 2 = 2 Days Ago, etc.
    """
    return await call_rapidapi("/api/v1/liveTrainStatus", {
        "trainNo": train_number,
        "startDay": str(start_day) # Pass '0' for today, '1' for yesterday, etc.
    })

async def get_seat_availability(train_no: str, from_station: str, to_station: str, date: str, class_code: str, quota: str = "GN") -> Dict[str, Any]:
    """
    Checks seat availability for a specific train, route, date, class, and quota.
    Common class codes: 1A, 2A, 3A, SL. Common quotas: GN (General), CK (Tatkal), PT (Premium Tatkal).
    Date format should be YYYY-MM-DD.
    """
    return await call_rapidapi("/api/v1/checkSeatAvailability", {
        "trainNo": train_no,
        "fromStationCode": from_station.upper(),
        "toStationCode": to_station.upper(),
        "date": date,
        "classType": class_code,  # Corrected key
        "quota": quota
    })

async def search_trains_between_stations(from_station_code: str, to_station_code: str, date_of_journey: str) -> Dict[str, Any]:
    """Searches for trains running between two station codes on a specific date. Station codes must be short (e.g., HYB, MAS). Date format should be YYYY-MM-DD."""
    return await call_rapidapi("/api/v3/trainBetweenStations", {
        "fromStationCode": from_station_code.upper(),
        "toStationCode": to_station_code.upper(),
        "dateOfJourney": date_of_journey
    })
    
async def get_train_schedule(train_number: str) -> Dict[str, Any]:
    """Retrieves the complete route and time-table for a given train number."""
    return await call_rapidapi("/api/v1/getTrainSchedule", {"trainNo": train_number})

RAILWAY_TOOLS = [
    get_pnr_status,
    get_live_train_status,
    get_seat_availability,
    search_trains_between_stations,
    get_train_schedule,
]

# Re-configure model with tools after RAILWAY_TOOLS is defined
if GEMINI_API_KEY and RAILWAY_TOOLS:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash", 
        system_instruction=SYSTEM_INSTRUCTION,
        tools=RAILWAY_TOOLS
    )
else:
    logging.error("❌ GEMINI_API_KEY not set or RAILWAY_TOOLS is empty. Chatbot disabled.")


# ------------------------
# Google Cloud Service Endpoints
# ------------------------

class TextChatRequest(BaseModel):
    query: str
    lang: str = "en"

@app.post("/text-to-speech/")
async def text_to_speech_only(
    text: str = Form(...),
    lang: str = Form("en")
):
    """Endpoint to convert text to speech without chatbot processing"""
    try:
        tts_result = await text_to_speech_cloud(text, lang)
        if tts_result["status"] == "success":
            return JSONResponse({"audio_base64": tts_result["bot_audio"]})
        else:
            raise HTTPException(status_code=500, detail="TTS generation failed")
    except Exception as e:
        logging.error(f"❌ TTS-only endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech from text")

async def text_to_speech_cloud(text: str, lang: str) -> Dict[str, Optional[str]]:
    """
    Convert text to speech using Google Cloud TTS and return a base64 encoded MP3 string.
    Enhanced with better language and voice selection.
    """
    try:
        tts_client = texttospeech.TextToSpeechClient()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Enhanced language mapping with more Indian languages
        language_mapping = {
            "en": "en-US", 
            "hi": "hi-IN", 
            "ta": "ta-IN", 
            "te": "te-IN",
            "kn": "kn-IN", 
            "ml": "ml-IN", 
            "mr": "mr-IN", 
            "bn": "bn-IN",
            "gu": "gu-IN",
            "pa": "pa-IN",
            "ur": "ur-IN",
            "fr": "fr-FR", 
            "es": "es-ES", 
            "de": "de-DE"
        }
        
        language_code = language_mapping.get(lang, "en-US")
        
        # Enhanced voice selection for better quality in different languages
        voice_selection = {
            "en-US": "en-US-Standard-C",      # Male voice for English
            "en-IN": "en-IN-Standard-A",      # Indian English
            "hi-IN": "hi-IN-Standard-A",      # Hindi female voice
            "ta-IN": "ta-IN-Standard-A",      # Tamil female voice  
            "te-IN": "te-IN-Standard-A",      # Telugu female voice
            "kn-IN": "kn-IN-Standard-A",      # Kannada female voice
            "ml-IN": "ml-IN-Standard-A",      # Malayalam female voice
            "mr-IN": "mr-IN-Standard-A",      # Marathi female voice
            "bn-IN": "bn-IN-Standard-A",      # Bengali female voice
            "gu-IN": "gu-IN-Standard-A",      # Gujarati female voice
            "pa-IN": "pa-IN-Standard-A",      # Punjabi female voice
            "ur-IN": "ur-IN-Standard-A",      # Urdu female voice
        }
        
        voice_name = voice_selection.get(language_code, "en-US-Standard-C")
            
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0,          # Normal pitch
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        bot_audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        
        logging.info(f"✅ TTS generated for language: {lang} (voice: {voice_name})")
        return {"status": "success", "bot_audio": bot_audio_base64}
        
    except Exception as e:
        logging.error(f"❌ Google Cloud TTS error for language {lang}: {e}")
        return {"status": "error", "bot_audio": None}

@app.post("/speech-to-text-only/")
async def speech_to_text(
    file: UploadFile = File(...),
    src_lang: str = Form("en")
):
    print(f"Received file type: {file.content_type}")
    
    # Support more audio formats for better compatibility
    allowed_content_types = [
        "audio/webm", "video/webm", "audio/webm; codecs=opus", "audio/ogg", "audio/mpeg", 
        "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac",
        "audio/x-m4a", "audio/flac"
    ]
    
    # Check if content type starts with any allowed prefix
    content_type_allowed = any(file.content_type.startswith(allowed.split(';')[0]) 
                              for allowed in allowed_content_types)
    
    if not content_type_allowed:
        raise HTTPException(status_code=400, detail=f"Invalid audio file format. Supported formats: {', '.join(allowed_content_types)}")

    temp_file = os.path.join(BASE_DIR, f"temp_{uuid.uuid4().hex}")
    
    # Determine file extension based on content type
    if "webm" in file.content_type:
        temp_file += ".webm"
        input_format = "webm"
    elif "ogg" in file.content_type:
        temp_file += ".ogg"
        input_format = "ogg"
    elif "wav" in file.content_type:
        temp_file += ".wav"
        input_format = "wav"
    elif "mp4" in file.content_type or "m4a" in file.content_type:
        temp_file += ".m4a"
        input_format = "mp4"
    elif "mpeg" in file.content_type:
        temp_file += ".mp3"
        input_format = "mp3"
    elif "flac" in file.content_type:
        temp_file += ".flac"
        input_format = "flac"
    else:
        temp_file += ".webm"  # default
        input_format = "webm"
    
    wav_file = os.path.join(BASE_DIR, f"temp_{uuid.uuid4().hex}.wav")
    
    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Convert to 16kHz mono 16-bit WAV for Google STT
        try:
            sound = AudioSegment.from_file(temp_file, format=input_format)
            sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 2 bytes = 16 bits
            sound.export(wav_file, format="wav")
        except Exception as audio_error:
            logging.error(f"❌ Audio conversion error: {audio_error}")
            # Try with different approach for problematic files
            try:
                sound = AudioSegment.from_file(temp_file)
                sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                sound.export(wav_file, format="wav")
            except Exception as fallback_error:
                logging.error(f"❌ Fallback audio conversion also failed: {fallback_error}")
                raise HTTPException(status_code=400, detail="Unable to process the audio file. Please try a different format.")

        client = speech.SpeechClient()
        language_code = GCLOUD_LANG_MAP_STT.get(src_lang, "en-IN")

        with open(wav_file, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True
        )

        response = client.recognize(config=config, audio=audio)
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
            
        return JSONResponse({"status": "success", "transcript": transcript})
        
    except Exception as e:
        logging.error(f"❌ Google Cloud STT/Audio processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process speech-to-text request.")
    finally:
        # Clean up temporary files
        for temp_path in [temp_file, wav_file]:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    logging.warning(f"Could not delete temp file {temp_path}: {cleanup_error}")

# ------------------------
# Main Chatbot Logic (Fixed Function Calling)
# ------------------------

async def call_gemini_with_tools(user_query: str, lang: str) -> str:
    """
    Uses Gemini Function Calling (Tools) to get a response, executing required tools in the process.
    Responds in the specified language.
    """
    if not model:
        return "Chatbot service is unavailable due to missing API key."

    # Enhanced prompt that emphasizes responding in the detected language
    language_prompt = f"User query in language '{lang}': {user_query}. You MUST respond in {lang} language only."
    
    # Tools are now configured in the GenerativeModel constructor.
    chat = model.start_chat() 
    
    response = chat.send_message(language_prompt)
    
    tool_calls = []
    if response.candidates and response.candidates[0].content.parts:
        tool_calls.extend([
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ])
    
    if tool_calls:
        function_responses = []
        # Use the correctly extracted tool_calls list
        for function_call in tool_calls:
            try:
                func = globals()[function_call.name]
                # Await the async tool function
                tool_output = await func(**dict(function_call.args))
                
                function_responses.append(genai.types.Part.from_function_response(
                    name=function_call.name,
                    response=tool_output
                ))
                
            except Exception as e:
                logging.error(f"❌ Tool execution error for {function_call.name}: {e}")
                function_responses.append(genai.types.Part.from_function_response(
                    name=function_call.name,
                    response={"error": "The external railway API failed to process the request."}
                ))
        
        # Send the tool responses back to the model with language reminder
        final_response_prompt = f"Based on the tool responses, provide the answer in {lang} language:"
        final_response = chat.send_message([final_response_prompt] + function_responses)
        return final_response.text
    
    return response.text

@app.post("/chatbot/")
async def unified_chatbot_voice_or_text(
    request: Request,
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
    src_lang: str = Form("en")
):
    """
    Unified chatbot endpoint for both voice (file) and text queries.
    Handles PNR, train info, and emergency location-aware services with memory.
    """

    user_text = ""
    detected_language = src_lang
    user_id = request.client.host  # approximate unique key for caching

    if file:
        try:
            stt_response = await speech_to_text(file=file, src_lang=src_lang)
            user_text = json.loads(stt_response.body.decode()).get("transcript", "")
            if not user_text:
                return JSONResponse({
                    "user_text": "[Unrecognized speech]",
                    "bot_response": "Sorry, I couldn't understand your voice command.",
                    "bot_audio": "",
                    "detected_language": "en"
                })
            detected_language = await detect_language(user_text)
        except Exception as e:
            logging.error(f"STT error: {e}")
            return JSONResponse({
                "user_text": "[STT Error]",
                "bot_response": "Speech-to-text failed. Please try again.",
                "bot_audio": "",
                "detected_language": "en"
            }, status_code=500)
    elif query:
        user_text = query.strip()
        if user_text and len(user_text) > 10:
            detected_language = await detect_language(user_text)

    if not user_text:
        raise HTTPException(status_code=400, detail="No query or audio file provided.")

    logging.info(f"🧠 Query ({detected_language}): {user_text}")
    query_lower = user_text.lower()
    bot_response_text = None

    # --- Detect and cache user location if provided ---
    lat_match = re.search(r"latitude\s*([\d.]+)", query_lower)
    lng_match = re.search(r"longitude\s*([\d.]+)", query_lower)
    if lat_match and lng_match:
        USER_LOCATION_CACHE[user_id] = {
            "lat": float(lat_match.group(1)),
            "lng": float(lng_match.group(1))
        }
        bot_response_text = "✅ Got your location! You can now ask for nearby police stations, ATMs, or hotels."

    # --- Handle railway automations ---
    pnr_match = re.search(r"\b\d{10}\b", query_lower)
    if not bot_response_text and pnr_match:
        pnr_number = pnr_match.group(0)
        api_result = await get_pnr_status(pnr_number)
        if "error" in api_result:
            bot_response_text = f"⚠️ Unable to fetch details for PNR {pnr_number}. Error: {api_result['error']}"
        else:
            data = api_result["data"].get("data", {})
            passengers = data.get("passengers", [])
            info = "\n".join(
                [f"👤 Passenger {p['no']}: {p['current_status']}" for p in passengers]
            )
            bot_response_text = (
                f"🚆 *PNR {pnr_number}*\n"
                f"Train: {data.get('train_name', 'N/A')}\n"
                f"Date: {data.get('doj', 'N/A')}\n"
                f"{info or 'No passenger details found.'}"
            )

    elif not bot_response_text and ("live" in query_lower and "train" in query_lower):
        train_num_match = re.search(r"\b\d{4,5}\b", query_lower)
        if train_num_match:
            train_number = train_num_match.group(0)
            api_result = await get_live_train_status(train_number, start_day=0)
            if "error" in api_result:
                bot_response_text = f"⚠️ Could not fetch live status for train {train_number}. Error: {api_result['error']}"
            else:
                pos = api_result["data"].get("data", {}).get("position", "No data.")
                bot_response_text = f"🚉 Train {train_number} Live Status:\n{pos}"
        else:
            bot_response_text = "Please provide a valid train number."

    elif not bot_response_text and ("schedule" in query_lower or "route" in query_lower):
        train_num_match = re.search(r"\b\d{4,5}\b", query_lower)
        if train_num_match:
            train_number = train_num_match.group(0)
            api_result = await get_train_schedule(train_number)
            if "error" in api_result:
                bot_response_text = f"⚠️ Could not fetch schedule for train {train_number}. Error: {api_result['error']}"
            else:
                route = api_result["data"].get("data", {}).get("route", [])
                stops = [f"{i+1}. {r['station_name']} ({r['station_code']})" for i, r in enumerate(route[:6])]
                bot_response_text = "🗓️ " + "\n".join(stops)
        else:
            bot_response_text = "Please enter a valid train number."

    # --- Handle emergency / nearby queries ---
    elif not bot_response_text and any(k in query_lower for k in ["nearest", "nearby", "close", "around"]):
        if any(w in query_lower for w in ["police", "hospital", "atm", "hotel", "restaurant", "pharmacy"]):
            service = next((w for w in ["police", "hospital", "atm", "hotel", "restaurant", "pharmacy"] if w in query_lower), None)
            coords = USER_LOCATION_CACHE.get(user_id)
            if coords:
                url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={coords['lat']},{coords['lng']}&radius=3000&type={service}&key={GOOGLE_MAPS_API_KEY}"
                res = requests.get(url).json()
                if res.get("results"):
                    places = [
                        f"{i+1}. {p['name']} - {p.get('vicinity', 'N/A')}\n🔗 [View on Map](http://googleusercontent.com/maps.google.com/6{p['place_id']})"
                        for i, p in enumerate(res["results"][:5])
                    ]
                    bot_response_text = f"📍 Nearest {service.capitalize()}s near you:\n" + "\n".join(places)
                else:
                    bot_response_text = f"⚠️ No nearby {service}s found within 3 km."
            else:
                bot_response_text = (
                    f"📍 To find the nearest {service}, please share your location like:\n"
                    "'My location is latitude 12.97 and longitude 77.59.'"
                )

    # --- Fallback to Gemini ---
    if not bot_response_text:
        bot_response_text = await call_gemini_with_tools(user_text, detected_language)

    # --- Text to Speech if needed ---
    bot_audio_base64 = ""
    if file:
        tts = await text_to_speech_cloud(bot_response_text, detected_language)
        bot_audio_base64 = tts.get("bot_audio", "")

    return JSONResponse({
        "user_text": user_text,
        "bot_response": bot_response_text,
        "bot_audio": bot_audio_base64,
        "detected_language": detected_language
    })

# ------------------------
# Other Railway API Endpoints 
# ------------------------

@app.get("/pnr-status/")
async def pnr_status_endpoint(pnr: str = Query(...)):
    result = await get_pnr_status(pnr)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    # If successful, return the whole 'data' object from the API response
    return result["data"]

@app.get("/train-status/")
async def train_status_endpoint(train_no: str = Query(...), start_day: int = Query(0)):
    result = await get_live_train_status(train_no, start_day)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]

@app.get("/search-trains/")
async def search_trains_endpoint(from_station: str = Query(...), to_station: str = Query(...), date: str = Query(...)):
    result = await search_trains_between_stations(from_station, to_station, date)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]

@app.get("/seat-availability/")
async def seat_availability_endpoint(train_no: str = Query(...), from_station: str = Query(...), to_station: str = Query(...), date: str = Query(...), class_code: str = Query(...), quota: str = Query("GN")):
    result = await get_seat_availability(train_no, from_station, to_station, date, class_code, quota)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]

@app.get("/train-schedule/")
async def train_schedule_endpoint(train_no: str = Query(...)):
    result = await get_train_schedule(train_no)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]

@app.get("/station-autocomplete/")
async def station_autocomplete(query_str: str = Query(...)):
    # Placeholder for a real station autocomplete API call
    return {"status": "success", "data": [{"station_name": f"{query_str} Station", "station_code": query_str.upper()}]}

# Nearest Police Station 
async def enhanced_nearest_police_search(lat: float, lon: float, radius: int = 10000):
    """
    Enhanced police station search using Google Places API with better error handling
    """
    if not GOOGLE_MAPS_API_KEY:
        logging.error("❌ GOOGLE_MAPS_API_KEY not set.")
        return {"error": "Server configuration error: Maps API key missing."}

    # First, find police stations using Places API
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": "police",
        "key": GOOGLE_MAPS_API_KEY,
    }

    try:
        response = requests.get(places_url, params=places_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        logging.info(f"🔍 Places API response status: {data.get('status')}")
        logging.info(f"🔍 Found {len(data.get('results', []))} results")
        
        if data.get("status") != "OK" or not data.get("results"):
            # Try with a broader search if police type doesn't work
            places_params["type"] = "point_of_interest"
            places_params["keyword"] = "police"
            response = requests.get(places_url, params=places_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK" or not data.get("results"):
                return {"error": "No police stations found nearby. Please try in a more populated area.", "details": data.get("status")}

        # Get the nearest police station
        nearest_station = data["results"][0]
        place_id = nearest_station.get("place_id")
        
        # Get detailed information including phone number
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "name,formatted_address,formatted_phone_number,geometry,website,international_phone_number",
            "key": GOOGLE_MAPS_API_KEY,
        }
        
        details_response = requests.get(details_url, params=details_params, timeout=10)
        details_response.raise_for_status()
        details_data = details_response.json()
        
        if details_data.get("status") != "OK":
            # If details fail, use basic info from nearby search
            station_lat = nearest_station["geometry"]["location"]["lat"]
            station_lng = nearest_station["geometry"]["location"]["lng"]
            station_name = nearest_station.get("name", "Police Station")
            address = nearest_station.get("vicinity", "Address not available")
        else:
            station_details = details_data.get("result", {})
            station_lat = station_details.get("geometry", {}).get("location", {}).get("lat", 
                         nearest_station["geometry"]["location"]["lat"])
            station_lng = station_details.get("geometry", {}).get("location", {}).get("lng",
                         nearest_station["geometry"]["location"]["lng"])
            station_name = station_details.get("name", nearest_station.get("name", "Police Station"))
            address = station_details.get("formatted_address", nearest_station.get("vicinity", "Address not available"))
        
        # Calculate distance
        user_location = (lat, lon)
        station_location = (station_lat, station_lng)
        distance_km = round(geodesic(user_location, station_location).kilometers, 2)
        
        # Get phone number (try multiple fields)
        phone_number = "100"  # Default emergency number
        if details_data.get("status") == "OK":
            station_details = details_data.get("result", {})
            phone_number = station_details.get("formatted_phone_number") or \
                          station_details.get("international_phone_number") or "100"
        
        return {
            "status": "success",
            "stationName": station_name,
            "address": address,
            "phone": phone_number,
            "website": details_data.get("result", {}).get("website", "") if details_data.get("status") == "OK" else "",
            "latitude": station_lat,
            "longitude": station_lng,
            "distance_km": distance_km,
            "place_id": place_id,
            "maps_url": f"http://googleusercontent.com/maps.google.com/7{station_lat},{station_lng}&travelmode=driving",
            "user_location": {"lat": lat, "lng": lon}
        }

    except requests.exceptions.Timeout:
        logging.error("❌ Google Maps API request timeout")
        return {"error": "Location service timeout. Please try again."}
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Google Maps API request error: {e}")
        return {"error": "Failed to connect to the mapping service. Please check your internet connection."}
    except Exception as e:
        logging.error(f"❌ Unexpected error in police search: {e}")
        return {"error": "An unexpected error occurred while searching for police stations."}

async def find_nearby_services(lat: float, lon: float, service_type: str = "hospital", radius: int = 5000):
    """
    Find nearby services like hospitals, pharmacies, etc.
    """
    if not GOOGLE_MAPS_API_KEY:
        return {"error": "Google Maps API key not configured"}

    service_mapping = {
        "hospital": "hospital",
        "pharmacy": "pharmacy",
        "atm": "atm",
        "restaurant": "restaurant",
        "hotel": "lodging"
    }
    
    place_type = service_mapping.get(service_type, "point_of_interest")
    
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": place_type,
        "key": GOOGLE_MAPS_API_KEY,
    }

    try:
        response = requests.get(places_url, params=places_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK" or not data.get("results"):
            return {"error": f"No {service_type} found nearby. Try increasing search radius.", "details": data.get("status")}

        services = []
        for place in data["results"][:5]:  # Limit to 5 results
            place_lat = place["geometry"]["location"]["lat"]
            place_lng = place["geometry"]["location"]["lng"]
            user_location = (lat, lon)
            place_location = (place_lat, place_lng)
            distance_km = round(geodesic(user_location, place_location).kilometers, 2)
            
            services.append({
                "name": place.get("name", "Unknown"),
                "address": place.get("vicinity", "Address not available"),
                "latitude": place_lat,
                "longitude": place_lng,
                "distance_km": distance_km,
                "rating": place.get("rating", "Not rated"),
                "place_id": place.get("place_id"),
                "maps_url": f"http://googleusercontent.com/maps.google.com/8{place_lat},{place_lng}&travelmode=driving"
            })
        
        return {
            "status": "success",
            "service_type": service_type,
            "services": services
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Google Places API request error: {e}")
        return {"error": "Failed to connect to the mapping service."}

@app.get("/emergency-contacts/")
async def get_emergency_contacts():
    """Get standard emergency contact numbers for India"""
    emergency_contacts = {
        "status": "success",
        "contacts": [
            {"name": "Police", "number": "100", "description": "Emergency Police"},
            {"name": "Ambulance", "number": "102", "description": "Medical Emergency"},
            {"name": "Fire", "number": "101", "description": "Fire Department"},
            {"name": "Disaster Management", "number": "108", "description": "Disaster Management Services"},
            {"name": "Women Helpline", "number": "1091", "description": "Women in Distress"},
            {"name": "Railway Enquiry", "number": "139", "description": "Railway Information"},
            {"name": "Railway Security", "number": "182", "description": "Railway Security Helpline"}
        ]
    }
    return JSONResponse(emergency_contacts)

# Enhanced police endpoint with better error handling
@app.get("/enhanced-nearest-police/")
async def get_enhanced_nearest_police(lat: float = Query(...), lon: float = Query(...)):
    """Enhanced police station search with detailed information"""
    result = await enhanced_nearest_police_search(lat, lon)
    return JSONResponse(result)

@app.get("/nearby-services/")
async def get_nearby_services(
    lat: float = Query(...), 
    lon: float = Query(...), 
    service_type: str = Query("hospital")
):
    """Find nearby services like hospitals, pharmacies, etc."""
    result = await find_nearby_services(lat, lon, service_type)
    return JSONResponse(result)

# Serve the index.html file
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    file_path = os.path.join(BASE_DIR, "frontend", "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        # Fallback to the original path if frontend/index.html doesn't exist
        file_path = os.path.join(BASE_DIR, "index.html")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return HTMLResponse(content=f"<h1>Error loading UI: {e}</h1>", status_code=500)
