# main.py
import os
import re
import random
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from twilio.twiml.voice_response import VoiceResponse, Gather

# Optional OpenAI import (only used if you install openai and set OPENAI_API_KEY)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- Config ----------
API_TITLE = "Air India Conversational IVR - Merged"
API_VERSION = "1.0.0-merged"
# Use your Twilio/agent phone number here or set AGENT_NUMBER env var
AGENT_NUMBER = os.getenv("AGENT_NUMBER", "+18314805664")

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "static")
STATIC_INDEX = os.path.join(FRONTEND_DIR, "index.html")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

# ---------- App ----------
app = FastAPI(title=API_TITLE, version=API_VERSION)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airindia_ivr_merged")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend if present
if os.path.isfile(STATIC_INDEX):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.info("Static frontend index not found at %s", STATIC_INDEX)

# ---------- Simple models & in-memory state ----------
class Booking(BaseModel):
    flight_id: int
    passenger_name: str = Field(..., min_length=2)
    seats: int = Field(1, gt=0)

booking_db: List[Dict[str, Any]] = []
call_sessions: Dict[str, Dict[str, Any]] = {}

IVR_MENU = {
    "main": {
        "prompt": "Welcome to Air India. Press 1 for Booking, 2 for Flight Status, 3 for Baggage & Refunds, or 9 to speak to an agent.",
        "options": {"1": "booking", "2": "status", "3": "baggage", "9": "agent"}
    },
    "booking": {"prompt": "Booking: Press 1 for Domestic, 2 for International, or 0 to return to main."},
    "status": {"prompt": "Please say or enter your 6 digit PNR after the beep."},
    "baggage": {"prompt": "Please explain your baggage or refund issue after the beep."},
}

# ---------- Helpers ----------
def format_phone_for_speech(phone_number: str) -> str:
    digits = re.sub(r'[^\d]', '', phone_number or "")
    digit_words = {str(i): w for i, w in enumerate(["zero","one","two","three","four","five","six","seven","eight","nine"])}
    if len(digits) == 10:
        part1 = ' '.join([digit_words.get(d,d) for d in digits[:3]])
        part2 = ' '.join([digit_words.get(d,d) for d in digits[3:6]])
        part3 = ' '.join([digit_words.get(d,d) for d in digits[6:]])
        return f"{part1}, {part2}, {part3}"
    return ' '.join([digit_words.get(d,d) for d in digits])

def format_flight_status(pnr_or_id: str) -> str:
    pnr_or_id = (pnr_or_id or "").upper().replace(" ", "")
    if not pnr_or_id:
        return "No PNR provided."
    if pnr_or_id.startswith("AI"):
        return f"Flight {pnr_or_id} is confirmed and on time. You are booked from Mumbai to Delhi."
    if len(pnr_or_id) == 6 and re.match(r'^[A-Z0-9]{6}$', pnr_or_id):
        return f"Your booking under PNR {pnr_or_id} for flight AI101 is confirmed. Is there anything else?"
    return f"I couldn't find a record for {pnr_or_id}."

def get_or_create_session(call_sid: str) -> Dict[str, Any]:
    if call_sid not in call_sessions:
        call_sessions[call_sid] = {
            "menu": "main",
            "history": [],
            "state": "main_menu",
            "pending_contact": None,
            "retry_count": 0,
            "meta": {}
        }
    return call_sessions[call_sid]

# Lightweight NLU (no external deps)
def simple_intent(user: str) -> str:
    s = (user or "").lower()
    if any(w in s for w in ("book","booking","ticket","reservation","domestic","international")):
        return "booking"
    if any(w in s for w in ("pnr","status","flight","ai")):
        return "status"
    if any(w in s for w in ("baggage","refund","lost","damage")):
        return "baggage"
    if any(w in s for w in ("agent","human","representative")):
        return "agent"
    if any(w in s for w in ("bye","goodbye","thanks","thank you")):
        return "goodbye"
    return "unknown"

# Optional small LLM fallback
def generate_ai_reply(prompt: str) -> str:
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            messages = [{"role":"system","content":"You are a concise IVR assistant for Air India."},
                        {"role":"user","content": prompt}]
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, max_tokens=120, temperature=0.2)
            return resp.choices[0].message.content.strip()
        except Exception:
            logger.exception("OpenAI call failed; falling back to rule-based reply.")
    # simple fallback rules
    p = (prompt or "").lower()
    if "pnr" in p or re.search(r'\b[a-z0-9]{6}\b', p):
        return format_flight_status(prompt)
    if "booking" in p:
        return "For booking, our team will contact you shortly. For international bookings please visit airindia.com."
    if "baggage" in p or "refund" in p:
        return "Please provide your booking reference and details—our baggage team will contact you within 24 hours."
    if "agent" in p:
        return "Please hold while I connect you to an agent."
    return "I didn't understand — would you like to go to the main menu or speak to an agent?"

# ---------- Twilio-like IVR endpoints (use /ivr per your choice) ----------
@app.post("/ivr")
async def ivr_entry(request: Request):
    """
    Entry endpoint expected by the frontend simulator.
    Accepts multipart form data with optional CallSid and returns TwiML XML.
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    logger.info("Incoming /ivr call_sid=%s", call_sid)
    session = get_or_create_session(call_sid)

    # Build TwiML main menu
    resp = VoiceResponse()
    gather = Gather(num_digits=1, input="speech dtmf", timeout=6, action="/handle-key", method="POST")
    gather.say(IVR_MENU["main"]["prompt"], voice="alice")
    resp.append(gather)
    resp.say("We did not receive any input. Goodbye.", voice="alice")
    resp.hangup()
    return Response(content=str(resp), media_type="application/xml")

@app.post("/handle-key")
async def handle_key(request: Request):
    """
    Handle DTMF or speech input after main menu.
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    digits = (form.get("Digits") or "").strip()
    speech = (form.get("SpeechResult") or "").strip()
    logger.info("handle-key call_sid=%s digits=%r speech=%r", call_sid, digits, speech)
    session = get_or_create_session(call_sid)

    choice = digits or None
    if not choice and speech:
        it = simple_intent(speech)
        if it == "booking": choice = "1"
        elif it == "status": choice = "2"
        elif it == "baggage": choice = "3"
        elif it == "agent": choice = "9"

    resp = VoiceResponse()
    # Booking route
    if choice == "1":
        gather = Gather(input="speech dtmf", timeout=6, action="/twilio/booking", method="POST")
        gather.say(IVR_MENU["booking"]["prompt"], voice="alice")
        resp.append(gather)
        resp.say("Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    # Flight status
    if choice == "2":
        gather = Gather(input="speech dtmf", num_digits=6, timeout=8, action="/twilio/conversation", method="POST")
        resp.say("Please say your 6 digit P N R after the beep, or enter it on your keypad.", voice="alice")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    # Baggage/refund
    if choice == "3":
        gather = Gather(input="speech", timeout=8, action="/twilio/conversation", method="POST")
        resp.say("Please describe your baggage or refund issue after the beep. I will note it.", voice="alice")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    # Agent
    if choice == "9":
        resp.say("Connecting you to an agent. Please hold.", voice="alice")
        resp.dial(AGENT_NUMBER)
        return Response(content=str(resp), media_type="application/xml")

    # fallback
    resp.say("Sorry, I didn't understand that. Returning to main menu.", voice="alice")
    resp.redirect("/ivr")
    return Response(content=str(resp), media_type="application/xml")


@app.post("/twilio/booking")
async def twilio_booking(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    digits = (form.get("Digits") or "").strip()
    speech = (form.get("SpeechResult") or "").strip()
    logger.info("twilio/booking call_sid=%s digits=%r speech=%r", call_sid, digits, speech)

    choice = digits or None
    if not choice and speech:
        s = speech.lower()
        if "domestic" in s: choice = "1"
        elif "international" in s: choice = "2"
        elif "main" in s or "back" in s: choice = "0"

    resp = VoiceResponse()
    if choice == "1":
        resp.say("Domestic booking confirmed. Our team will call you shortly to finish the booking. Goodbye.", voice="alice")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")
    if choice == "2":
        resp.say("International booking requires additional details. Please visit airindia.com or hold to speak to an agent.", voice="alice")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")
    if choice == "0":
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    resp.say("Sorry I didn't get that. Returning to main menu.", voice="alice")
    resp.redirect("/ivr")
    return Response(content=str(resp), media_type="application/xml")


@app.post("/twilio/conversation")
async def twilio_conversation(request: Request):
    """
    Receives speech or digits and responds (PNR detection + AI fallback).
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    speech = (form.get("SpeechResult") or "").strip()
    digits = (form.get("Digits") or "").strip()
    user_input = (speech or digits or "").strip()
    logger.info("twilio/conversation call_sid=%s input=%r", call_sid, user_input)

    # quick detect PNR-like input
    if user_input:
        up = user_input.replace(" ", "")
        if re.match(r'^[A-Za-z0-9]{6,}$', up):
            reply_text = format_flight_status(up)
            resp = VoiceResponse()
            resp.say(reply_text, voice="alice")
            resp.hangup()
            return Response(content=str(resp), media_type="application/xml")

    # fallback: AI or templated
    reply_text = generate_ai_reply(user_input or "Hello. How can I help?")

    resp = VoiceResponse()
    resp.say(reply_text, voice="alice")
    gather = Gather(input="speech dtmf", timeout=7, action="/twilio/conversation", method="POST")
    resp.append(gather)
    return Response(content=str(resp), media_type="application/xml")

# ---------- simple web / api endpoints ----------
@app.get("/_status")
def status():
    return {"ok": True, "calls_in_memory": len(call_sessions), "bookings": len(booking_db)}

@app.get("/")
async def root():
    if os.path.isfile(STATIC_INDEX):
        return FileResponse(STATIC_INDEX)
    return JSONResponse({"message":"IVR backend running", "version": API_VERSION})

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})
