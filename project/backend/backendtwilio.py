# main.py
import os
import re
import random
import logging
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from twilio.twiml.voice_response import VoiceResponse, Gather

# Optional OpenAI import (if you put API key in env and installed openai)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- Config ----------
API_TITLE = "Air India Conversational IVR - Merged"
API_VERSION = "1.0.0-merged"
AGENT_NUMBER = os.getenv("AGENT_NUMBER", "+911234567890")

# Serve frontend from ./static (put index.html and assets in static/)
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
    allow_origins=["*"],  # in prod lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend if present
if os.path.isfile(STATIC_INDEX):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.info("Static frontend index not found at %s", STATIC_INDEX)

# ---------- Simple models ----------
class Booking(BaseModel):
    flight_id: int
    passenger_name: str = Field(..., min_length=2)
    seats: int = Field(1, gt=0)

# ---------- In-memory state ----------
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
    digit_words = {'0': 'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
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
        call_sessions[call_sid] = {"menu":"main","history":[],"state":"main_menu","pending_contact":None}
    return call_sessions[call_sid]

# ---------- Optional lightweight NLU (no external deps) ----------
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

# ---------- Twilio-like IVR endpoints ----------
@app.post("/ivr")
async def ivr_entry(request: Request):
    """
    Frontend calls POST /ivr (multipart/form-data) with optional CallSid.
    Returns TwiML XML (VoiceResponse).
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    session = get_or_create_session(call_sid)

    # Set a simple main menu TwiML
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
    Receives Digits or SpeechResult. Returns TwiML response depending on choice.
    Frontend will call this when user presses keys during an active call.
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    digits = (form.get("Digits") or "").strip()
    speech = (form.get("SpeechResult") or "").strip()
    session = get_or_create_session(call_sid)

    # prefer digits, else infer from speech
    choice = digits or None
    if not choice and speech:
        it = simple_intent(speech)
        if it == "booking": choice = "1"
        elif it == "status": choice = "2"
        elif it == "baggage": choice = "3"
        elif it == "agent": choice = "9"

    resp = VoiceResponse()
    # Booking flow
    if choice == "1":
        gather = Gather(input="speech dtmf", timeout=6, action="/twilio/booking", method="POST")
        gather.say(IVR_MENU["booking"]["prompt"], voice="alice")
        resp.append(gather)
        resp.say("Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    # Flight status
    if choice == "2":
        # ask for PNR
        gather = Gather(input="speech dtmf", num_digits=6, timeout=8, action="/twilio/conversation", method="POST")
        resp.say("Please say your 6 digit P N R after the beep, or enter it on your keypad.", voice="alice")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

    # Baggage
    if choice == "3":
        gather = Gather(input="speech", timeout=8, action="/twilio/conversation", method="POST")
        resp.say("Please describe your baggage or refund issue after the beep. I will note it.", voice="alice")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/ivr")
        return Response(content=str(resp), media_type="application/xml")

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
    This endpoint receives a digits (PNR) or speech result and replies with TwiML.
    The function includes simple NLU and templated replies. If OPENAI_API_KEY present
    and openai installed, it will attempt a short AI reply.
    """
    form = await request.form()
    call_sid = form.get("CallSid") or f"FE_{random.randint(1000,9999)}"
    speech = (form.get("SpeechResult") or "").strip()
    digits = (form.get("Digits") or "").strip()
    user_input = speech or digits or ""

    # If the input looks like a PNR or flight id -> reply with status
    if user_input:
        # quick PNR detection
        up = user_input.strip()
        if re.match(r'^[A-Za-z0-9]{6,}$', up.replace(" ", "")):
            reply_text = format_flight_status(up)
            resp = VoiceResponse()
            resp.say(reply_text, voice="alice")
            resp.hangup()
            return Response(content=str(resp), media_type="application/xml")

    # fallback: simple AI-ish or templated reply
    reply_text = "I'm sorry, I couldn't detect a P N R. Please try again or say 'agent' to speak to a human."

    # If openai available, attempt short completion (safe guard)
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            msg = f"User input: {user_input}\nReply shortly as an IVR: provide helpful instruction."
            # minimal safe usage: use completion/chat based on your openai package
            rsp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if "gpt-4o-mini" else "gpt-4o-mini",
                messages=[{"role":"user","content":msg}],
                max_tokens=120,
                temperature=0.2
            )
            ai_text = rsp.choices[0].message.content.strip()
            if ai_text:
                reply_text = ai_text
        except Exception:
            logger.exception("OpenAI attempt failed, using fallback reply.")

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
    # If static index exists, FileResponse will be served by mounted StaticFiles already.
    if os.path.isfile(STATIC_INDEX):
        return FileResponse(STATIC_INDEX)
    return JSONResponse({"message":"IVR backend running", "version": API_VERSION})

# Global exception handler (keeps logs clean)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})
