# main.py
import os
import logging
import random
import re
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twilio.twiml.voice_response import VoiceResponse, Gather

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- Configuration ----------
API_TITLE = "Air India Conversational IVR - Merged"
API_VERSION = "1.0.0-merged"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

AGENT_NUMBER = os.getenv("AGENT_NUMBER", "+911234567890")
FRONTEND_DIR = os.getenv("FRONTEND_DIR", os.path.join(os.path.dirname(__file__), "frontend"))
STATIC_CHECK_PATH = os.path.join(FRONTEND_DIR, "index.html")

# ---------- App setup ----------
app = FastAPI(title=API_TITLE, version=API_VERSION)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airindia_ivr_merged")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static if available
if os.path.isfile(STATIC_CHECK_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    # serve a basic root if no frontend
    pass

# ---------- Data models & state ----------
class Booking(BaseModel):
    flight_id: int
    passenger_name: str = Field(..., min_length=2)
    seats: int = Field(1, gt=0)

class BookingMenu(BaseModel):
    booking_id: str
    trsna_id: str
    passenger_fullname: str
    passenger_contact: str
# In-memory DBs (replace with DB for production)
booking_db: List[Dict[str, Any]] = []
flights = [
    {"flights_id": "AI1", "origin": "Mumbai", "destination": "Chennai", "status": "Confirmed"},
    {"flights_id": "AI2", "origin": "Chennai", "destination": "Kochi", "status": "Delayed"},
    {"flights_id": "AI3", "origin": "Delhi", "destination": "Bangalore", "status": "Cancelled"},
]

# In-memory call sessions (shared across features)
call_sessions: Dict[str, Dict[str, Any]] = {}

IVR_MENU = {
    "main": {
        "prompt": "Welcome to Air India. Press 1 for Booking, 2 for Flight Status, 3 for Baggage & Refunds, or 9 to speak to an agent.",
        "options": {"1": "booking", "2": "status", "3": "baggage", "9": "agent"}
    },
    "booking": {"prompt": "Booking: Press 1 for Domestic, 2 for International, or 0 to return to main.", "options": {"1": "domestic", "2": "international", "0": "main"}},
    "status": {"prompt": "Please say or enter your 6 digit PNR after the beep.", "options": {}},
    "baggage": {"prompt": "Please explain your baggage or refund issue after the beep.", "options": {}},
}

# ---------- Enums & utilities ----------
class IvrIntent(Enum):
    BOOK_FLIGHT = "book_flight"
    CHECK_STATUS = "check_status"
    BAG_REFUND = "bag_refund"
    SPEAK_AGENT = "speak_agent"
    GREETING = "greeting"
    HELP = "help"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

class IvrConversationState(Enum):
    INITIAL = "initial"
    MAIN_MENU = "main_menu"
    PROACTIVE_OFFER = "proactive_offer"
    COLLECTING_BOOKING_TYPE = "collecting_booking_type"
    COLLECTING_PNR_OR_ID = "collecting_pnr_or_id"
    COLLECTING_ISSUE_DETAILS = "collecting_issue_details"
    COLLECTING_CALLBACK_CONTACT = "collecting_callback_contact"
    CONFIRMING_CONTACT = "confirming_contact"
    COMPLETED = "completed"

def format_phone_for_speech(phone_number: str) -> str:
    digits = re.sub(r'[^\d]', '', phone_number or "")
    digit_words = {'0': 'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
    if len(digits) == 10:
        part1 = ' '.join([digit_words.get(d,d) for d in digits[:3]])
        part2 = ' '.join([digit_words.get(d,d) for d in digits[3:6]])
        part3 = ' '.join([digit_words.get(d,d) for d in digits[6:]])
        return f"{part1}, {part2}, {part3}"
    return ', '.join([digit_words.get(d, d) for d in digits])

def format_flight_status(pnr_or_id: str) -> str:
    pnr_or_id = (pnr_or_id or "").upper().replace(" ", "")
    if not pnr_or_id:
        return "No PNR provided."
    if pnr_or_id.startswith("AI"):
        return f"Flight {pnr_or_id} is confirmed and on time. You are booked from Mumbai to Delhi."
    if len(pnr_or_id) == 6 and re.match(r'^[A-Z0-9]{6}$', pnr_or_id):
        return f"Your booking under PNR {pnr_or_id} for flight AI101 is confirmed. Is there anything else?"
    return f"I couldn't find a record for {pnr_or_id}."

def log_final_action(call_sid: str, action: str, data: Dict) -> None:
    logger.info(f"FINAL ACTION LOGGED: Call={call_sid}, Action={action}, Data={data}")

def simulate_llm_issue_elicitation(user_issue_desc: str) -> str:
    u = (user_issue_desc or "").lower()
    if "lost" in u and "mumbai" in u:
        summary = "You mentioned a lost bag related to your flight into Mumbai."
        target = "Do you have the baggage tag number ready, or should I start a new tracing report?"
    elif "refund" in u and "delayed" in u:
        summary = "You are requesting a refund related to a flight delay."
        target = "Is this for flight AI2 which was delayed yesterday, or a different flight?"
    else:
        summary = "I understand you have an important baggage or refund query."
        target = "Please confirm your PNR or Flight ID so I can access your records."
    return f"{summary} Based on this, {target}"

def get_proactive_offer(call_sid: str) -> Optional[Dict[str, str]]:
    # Simulated proactive match — replace with real DB lookup
    if '421' in (call_sid or ""):
        return {"name": "Mr. Sharma", "flight": "AI421", "issue": "Your luggage was mistakenly sent to Goa."}
    return None

# ---------- Intent recognizer ----------
class IvrIntentRecognizer:
    def __init__(self):
        self.intent_patterns = {
            IvrIntent.BOOK_FLIGHT: [r'\b(book|reservation|ticket|buy|new flight|domestic|international)\b'],
            IvrIntent.CHECK_STATUS: [r'\b(check|status|view|details|look up|flight|pnr|ai\s*\d+)\b'],
            IvrIntent.BAG_REFUND: [r'\b(baggage|refund|lost bag|damaged bag|compensation|money back)\b'],
            IvrIntent.SPEAK_AGENT: [r'\b(agent|representative|speak to|human|person)\b'],
            IvrIntent.GREETING: [r'\b(hi|hello|hey|greetings|good morning)\b'],
            IvrIntent.HELP: [r'\b(help|what can you do|options|menu|assistance|support)\b'],
            IvrIntent.GOODBYE: [r'\b(bye|goodbye|thanks|thank you|done|finish|exit|end)\b'],
        }
        self.compiled_patterns = {intent: [re.compile(p, re.IGNORECASE) for p in patterns] for intent, patterns in self.intent_patterns.items()}

    def recognize_intent(self, user_input: str) -> Tuple[IvrIntent, float]:
        if not user_input:
            return IvrIntent.UNKNOWN, 0.0
        s = user_input.lower().strip()
        scores = {}
        for intent, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0
            for patt in patterns:
                m = patt.search(s)
                if m:
                    matches += 1
                    position_weight = 1.0 if m.start() < len(s)*0.3 else 0.7
                    score += position_weight
            if matches > 0:
                scores[intent] = min(score / len(patterns) * matches, 1.0)
        if scores:
            best = max(scores, key=scores.get)
            conf = scores[best]
            if conf >= 0.3:
                return best, conf
        return IvrIntent.UNKNOWN, 0.0

    def extract_entities(self, user_input: str) -> Dict[str, Optional[str]]:
        entities: Dict[str, Optional[str]] = {}
        u_up = (user_input or "").upper()
        pnr_match = re.search(r'\b([A-Z0-9]{6,})\b', u_up)
        if pnr_match:
            entities['pnr_or_id'] = pnr_match.group(1)

        # normalize number words for phone detection
        normalized = (user_input or "").lower()
        number_word_map = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','oh':'0','o':'0'}
        for w,d in number_word_map.items():
            normalized = normalized.replace(f' {w} ', f' {d} ')
            normalized = normalized.replace(f' {w}', f' {d}')
            normalized = normalized.replace(f'{w} ', f'{d} ')
        # phone patterns
        phone_pat = [r'\b(\d{10,15})\b']
        for p in phone_pat:
            m = re.search(p, normalized)
            if m:
                phone = re.sub(r'[^\d]', '', m.group(1))
                if len(phone) >= 10:
                    entities['contact'] = phone
                    break
        return entities

intent_recognizer = IvrIntentRecognizer()

# ---------- Session helpers ----------
def get_or_create_session(call_sid: str) -> Dict[str, Any]:
    if call_sid not in call_sessions:
        call_sessions[call_sid] = {
            "menu": "main",
            "dialogue_state": IvrConversationState.INITIAL.value,
            "history": [],
            "pending_contact": None,
            "action_type": None,
            "retry_count": 0,
            "proactive_data": None,
            "last_prompt": None,
            "meta": {}
        }
    return call_sessions[call_sid]

def update_session_state(call_sid: str, new_state: IvrConversationState) -> None:
    session = get_or_create_session(call_sid)
    session["dialogue_state"] = new_state.value
    session["retry_count"] = 0
    logger.info(f"Call {call_sid} state -> {new_state.value}")

def get_current_state(call_sid: str) -> IvrConversationState:
    session = get_or_create_session(call_sid)
    return IvrConversationState(session.get("dialogue_state", IvrConversationState.INITIAL.value))

def increment_retry(call_sid: str) -> int:
    session = get_or_create_session(call_sid)
    session['retry_count'] = session.get('retry_count', 0) + 1
    return session['retry_count']

# ---------- AI reply (OpenAI optional) ----------
def generate_ai_reply(prompt: str, session_history: Optional[List[Dict[str,str]]] = None) -> str:
    prompt = (prompt or "").strip()
    # If OpenAI is available, try to use it
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            messages = [{"role":"system","content":"You are a friendly Air India voice assistant. Keep replies short, helpful, and actionable."}]
            if session_history:
                messages.extend(session_history[-6:])
            messages.append({"role":"user","content":prompt})
            # use ChatCompletion if available, fallback to completion style
            resp = openai.ChatCompletion.create(model="gpt-4o" if "gpt-4o" in [m.id for m in openai.Model.list().data] else "gpt-4o-mini", messages=messages, max_tokens=150, temperature=0.3)
            ai_text = resp.choices[0].message.content.strip()
            logger.info("AI reply (OpenAI): %s", ai_text)
            return ai_text
        except Exception as e:
            logger.exception("OpenAI error, falling back to rules.")
    # fallback simple rules
    lower = prompt.lower()
    if "pnr" in lower or re.search(r'\b[a-z0-9]{6}\b', lower):
        return format_flight_status(prompt)
    if "agent" in lower or "representative" in lower or "human" in lower:
        return "Please hold while I connect you to an agent."
    if "booking" in lower or "domestic" in lower or "international" in lower:
        return "For booking, our team will contact you shortly. For international bookings please visit airindia.com."
    if "baggage" in lower or "refund" in lower:
        return "Please provide your booking reference and details—our baggage team will contact you within 24 hours."
    return "I'm not sure how to help with that right now. Would you like to check the main menu, or speak to an agent?"

# ---------- Twilio IVR endpoints (merged & stateful) ----------
@app.post("/ivr")
async def ivr_post(request: Request) -> Response:
    return await twilio_ivr(request)

    """
    Forward POST /ivr to the Twilio-style IVR entrypoint (twilio_ivr).
    This ensures the front-end fetch('/ivr', {method: 'POST'}) receives TwiML XML.
    """
    return await twilio_ivr(request)

# IMPORTANT: register the Twilio IVR handler route (was missing a decorator)
@app.post("/twilio/ivr")
async def twilio_ivr(request: Request) -> Response:
    """Entry point — proactive check then main menu."""
    form = await request.form()
    call_sid = form.get("CallSid") or f"LOCAL_{random.randint(1000,9999)}"
    session = get_or_create_session(call_sid)

    # Proactive
    proactive_data = get_proactive_offer(call_sid)
    if proactive_data:
        session["proactive_data"] = proactive_data
        update_session_state(call_sid, IvrConversationState.PROACTIVE_OFFER)
        proactive_msg = (
            f"Hello, {proactive_data['name']}. We show an active issue with your booking on flight {proactive_data['flight']}. "
            f"It appears {proactive_data['issue']} Would you like to address this issue now, or access the main menu?"
        )
        resp = VoiceResponse()
        gather = Gather(input="speech dtmf", timeout=7, action="/twilio/conversation", method="POST")
        gather.say(proactive_msg, voice="alice")
        resp.append(gather)
        resp.say("We did not receive any input. Goodbye.", voice="alice")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")

    # Default main menu
    update_session_state(call_sid, IvrConversationState.MAIN_MENU)
    resp = VoiceResponse()
    gather = Gather(num_digits=1, input="speech dtmf", timeout=5, action="/twilio/handle-key", method="POST")
    gather.say(IVR_MENU["main"]["prompt"], voice="alice")
    resp.append(gather)
    resp.say("We did not receive any input. Goodbye.", voice="alice")
    resp.hangup()
    return Response(content=str(resp), media_type="application/xml")

# ... rest of your routes unchanged ...
# (kept exactly as you had them: /handle-key, /twilio/booking, /twilio/conversation etc.)
@app.post("/handle-key")
async def twilio_handle_key(request: Request) -> Response:
    form = await request.form()
    call_sid = form.get("CallSid") or f"LOCAL_{random.randint(1000,9999)}"
    digits = form.get("Digits", "")
    speech = form.get("SpeechResult", "")
    logger.info(f"handle-key call_sid={call_sid} digits={digits!r} speech={speech!r}")

    session = get_or_create_session(call_sid)
    choice = digits or None
    if not choice and speech:
        s = speech.lower()
        if "booking" in s or "book" in s: choice = "1"
        elif "status" in s or "pnr" in s or "flight" in s: choice = "2"
        elif "baggage" in s or "refund" in s: choice = "3"
        elif "agent" in s or "representative" in s: choice = "9"

    resp = VoiceResponse()
    if choice == "1":
        session["menu"] = "booking"
        gather = Gather(input="speech dtmf", timeout=6, action="/twilio/booking", method="POST")
        gather.say(IVR_MENU["booking"]["prompt"], voice="alice")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/twilio/ivr")
        return Response(content=str(resp), media_type="application/xml")

    if choice == "2":
        session["menu"] = "status"
        resp.say("Please say your 6-digit P N R after the beep, or enter it on your keypad.", voice="alice")
        gather = Gather(input="speech dtmf", num_digits=6, timeout=8, action="/twilio/conversation", method="POST")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/twilio/ivr")
        return Response(content=str(resp), media_type="application/xml")

    if choice == "3":
        session["menu"] = "baggage"
        resp.say("Please describe your baggage or refund issue after the beep. I will note it.", voice="alice")
        gather = Gather(input="speech", timeout=8, action="/twilio/conversation", method="POST")
        resp.append(gather)
        resp.say("No input received. Returning to main menu.", voice="alice")
        resp.redirect("/twilio/ivr")
        return Response(content=str(resp), media_type="application/xml")

    if choice == "9":
        resp.say("Connecting you to an agent. Please hold.", voice="alice")
        resp.dial(AGENT_NUMBER)
        return Response(content=str(resp), media_type="application/xml")

    # fallback: if speech exists, pass to conversation endpoint
    if speech:
        return await twilio_conversation(request)

    resp.say("Sorry, I didn't understand that. Returning to main menu.", voice="alice")
    resp.redirect("/twilio/ivr")
    return Response(content=str(resp), media_type="application/xml")

@app.post("/twilio/booking")
async def twilio_booking(request: Request) -> Response:
    form = await request.form()
    call_sid = form.get("CallSid") or f"LOCAL_{random.randint(1000,9999)}"
    digits = form.get("Digits", "")
    speech = form.get("SpeechResult", "")
    logger.info(f"booking call_sid={call_sid} digits={digits!r} speech={speech!r}")

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
        resp.redirect("/twilio/ivr")
        return Response(content=str(resp), media_type="application/xml")

    resp.say("Sorry I didn't get that. Returning to main menu.", voice="alice")
    resp.redirect("/twilio/ivr")
    return Response(content=str(resp), media_type="application/xml")

@app.post("/twilio/conversation")
async def twilio_conversation(request: Request) -> Response:
    form = await request.form()
    call_sid = form.get("CallSid") or f"LOCAL_{random.randint(1000,9999)}"
    speech_result = form.get("SpeechResult", "") or ""
    digits = form.get("Digits", "") or ""
    user_input = (speech_result or digits or "").strip()

    logger.info(f"conversation call_sid={call_sid} input={user_input!r} digits={digits!r}")

    session = get_or_create_session(call_sid)
    current_state = get_current_state(call_sid)

    # NLU
    intent, confidence = intent_recognizer.recognize_intent(user_input)
    entities = intent_recognizer.extract_entities(user_input)
    if user_input:
        session["history"].append({"role":"user","content":user_input})

    reply_text = None
    end_flow = False

    # Universal navigation interrupts
    if current_state != IvrConversationState.MAIN_MENU and (intent == IvrIntent.SPEAK_AGENT or 'agent' in user_input.lower()):
        reply_text = "Connecting you to a human agent now. Please hold."
        end_flow = True
    elif intent == IvrIntent.GOODBYE:
        reply_text = "Thank you for calling Air India. Goodbye."
        end_flow = True
    elif 'main menu' in user_input.lower() or intent == IvrIntent.HELP:
        reply_text = "Redirecting you to the main menu. " + IVR_MENU["main"]["prompt"]
        update_session_state(call_sid, IvrConversationState.MAIN_MENU)

    # Contextual DTMF bypass: if user says PNR anywhere (and not in some states)
    elif current_state not in [IvrConversationState.COLLECTING_PNR_OR_ID, IvrConversationState.PROACTIVE_OFFER] and entities.get('pnr_or_id'):
        pnr = entities.get('pnr_or_id')
        reply_text = f"Thank you for providing {pnr}. Skipping to flight status now... {format_flight_status(pnr)} Would you like anything else?"
        update_session_state(call_sid, IvrConversationState.COMPLETED)

    # Dialogue state logic
    elif current_state == IvrConversationState.PROACTIVE_OFFER:
        proactive_data = session.get("proactive_data", {})
        if 'issue' in user_input.lower() or 'yes' in user_input.lower() or intent == IvrIntent.BAG_REFUND:
            update_session_state(call_sid, IvrConversationState.COLLECTING_ISSUE_DETAILS)
            reply_text = f"Let's resolve the issue for flight {proactive_data.get('flight')}. Please describe the situation in more detail now."
        else:
            update_session_state(call_sid, IvrConversationState.MAIN_MENU)
            reply_text = IVR_MENU["main"]["prompt"]

    elif current_state == IvrConversationState.MAIN_MENU:
        # digits preference
        if digits == "1" or intent == IvrIntent.BOOK_FLIGHT:
            update_session_state(call_sid, IvrConversationState.COLLECTING_BOOKING_TYPE)
            reply_text = IVR_MENU["booking"]["prompt"]
        elif digits == "2" or intent == IvrIntent.CHECK_STATUS:
            update_session_state(call_sid, IvrConversationState.COLLECTING_PNR_OR_ID)
            reply_text = "Certainly. Please say your 6-digit P N R or your flight number (like A I 1 0 1)."
        elif digits == "3" or intent == IvrIntent.BAG_REFUND:
            update_session_state(call_sid, IvrConversationState.COLLECTING_ISSUE_DETAILS)
            reply_text = "I can log a baggage or refund request. Please describe your issue after the tone, or say 'call back'."
        elif digits == "9":
            reply_text = "Connecting you to an agent now. Please hold."
            end_flow = True
        else:
            reply_text = IVR_MENU["main"]["prompt"]

    elif current_state == IvrConversationState.COLLECTING_PNR_OR_ID:
        pnr_or_id = entities.get('pnr_or_id', '')
        if pnr_or_id and len(pnr_or_id) >= 6:
            reply_text = format_flight_status(pnr_or_id)
            update_session_state(call_sid, IvrConversationState.COMPLETED)
            log_final_action(call_sid, "STATUS_CHECK_SUCCESS", {"ID": pnr_or_id})
        else:
            retry = increment_retry(call_sid)
            if retry >= 3:
                reply_text = "I'm unable to detect the ID after several attempts. I will now connect you to an agent."
                end_flow = True
            else:
                retry_msg = "Please say it clearly, for example, 'P N R A B C D'." if retry == 1 else "I'm having trouble. Please state the ID again clearly."
                reply_text = f"That ID seems invalid or too short. {retry_msg}"

    elif current_state == IvrConversationState.COLLECTING_ISSUE_DETAILS:
        # if user asked for callback or provided contact
        if "call back" in user_input.lower() or "call me" in user_input.lower() or entities.get('contact'):
            update_session_state(call_sid, IvrConversationState.COLLECTING_CALLBACK_CONTACT)
            reply_text = "Understood. What is the best phone number for us to call you back on?"
            if entities.get('contact'):
                reply_text = f"I see you provided a number. {reply_text}"
        else:
            # LLM-style slot elicitation (simulated)
            elicitation_reply = simulate_llm_issue_elicitation(user_input)
            reply_text = f"Thank you for describing the issue. {elicitation_reply}"
            session['meta']['issue_desc'] = user_input

    elif current_state in [IvrConversationState.COLLECTING_CALLBACK_CONTACT, IvrConversationState.CONFIRMING_CONTACT]:
        contact = entities.get('contact')
        pending = session.get('pending_contact')
        u = user_input.lower()
        if pending and re.search(r'\b(yes|yep|correct)\b', u):
            log_final_action(call_sid, "CALLBACK_REQUEST", {"Contact": pending, "Issue": session['meta'].get('issue_desc', 'N/A')})
            reply_text = f"Thank you. We will call {format_phone_for_speech(pending)} within 24 hours. Goodbye."
            session['pending_contact'] = None
            update_session_state(call_sid, IvrConversationState.COMPLETED)
            end_flow = True
        elif pending and re.search(r'\b(no|wrong|nope)\b', u):
            session['pending_contact'] = None
            retry = increment_retry(call_sid)
            if retry >= 2:
                reply_text = "I'm having difficulty confirming the number. I will connect you to an agent."
                end_flow = True
            else:
                reply_text = "No problem. Please say your phone number again, digit by digit."
        elif contact:
            contact_spoken = format_phone_for_speech(contact)
            reply_text = f"I have the number {contact_spoken}. Is that correct? Say 'yes' or 'no'."
            session['pending_contact'] = contact
            update_session_state(call_sid, IvrConversationState.CONFIRMING_CONTACT)
        else:
            retry = increment_retry(call_sid)
            if retry >= 2:
                reply_text = "I'm having trouble getting the number. I will connect you to an agent."
                end_flow = True
            else:
                reply_text = "I couldn't detect a valid contact number. Please say it slowly, digit by digit."

    # Final action
    if end_flow:
        resp = VoiceResponse()
        if reply_text and "agent" in (reply_text or "").lower():
            resp.say(reply_text, voice="alice")
            resp.dial(AGENT_NUMBER)
        else:
            resp.say(reply_text or "Goodbye.", voice="alice")
            resp.hangup()
        return Response(content=str(resp), media_type="application/xml")

    # Default: generate reply (AI or templated) and keep the call open
    if not reply_text:
        ai_text = generate_ai_reply(user_input or "Hello", session_history=session.get("history"))
        reply_text = ai_text
        session["history"].append({"role":"assistant","content":ai_text})

    # Speak reply and gather more input
    resp = VoiceResponse()
    session["history"].append({"role":"assistant","content":reply_text})
    resp.say(reply_text, voice="alice")
    gather = Gather(input="speech dtmf", timeout=7, action="/twilio/conversation", method="POST")
    resp.append(gather)
    return Response(content=str(resp), media_type="application/xml")

# ---------- Booking & flight APIs ----------
@app.get("/")
async def root_index():
    # If frontend index exists, serve it
    if os.path.isfile(STATIC_CHECK_PATH):
        return FileResponse(STATIC_CHECK_PATH)
    return JSONResponse({"message":"Air India IVR backend running", "version": API_VERSION})

@app.get("/home")
def home():
    return {"message": "Welcome to Air India support IVR — Press 1 Booking, 2 Status, 3 Baggage, 9 Agent."}

@app.get("/booking_menu")
def booking_menu():
    return {"menu": "Booking menu", "option": ["1. Domestic", "2. International"]}

@app.get("/status_menu")
def status_menu():
    return {"menu": "Status Menu", "option": ["Enter the flight id or PNR to check status"]}

@app.get("/domestic_booking")
def domestic_booking():
    return {"message": "Domestic booking flow started"}

@app.get("/international_booking")
def international_booking():
    return {"message": "International booking flow started"}

@app.post("/booking", status_code=201)
def create_booking(booking: Booking):
    booking_id = f"BKG{random.randint(1000,9999)}"
    rec = {"booking_id": booking_id, "booking": booking.dict(), "created_at": datetime.utcnow().isoformat()}
    booking_db.append(rec)
    return {"message": "Booked", "booking": rec}

@app.get("/flight/{flight_id}")
def get_flight(flight_id: str):
    statuses = {f["flights_id"]: f for f in flights}
    if flight_id not in statuses:
        raise HTTPException(status_code=404, detail="Flight not found")
    return statuses[flight_id]

@app.get("/active_flights")
def active_flights():
    return {"active_flights": flights}

@app.delete("/cancel_flight/{booking_id}")
def cancel_booking(booking_id: str):
    for b in booking_db:
        if b.get("booking_id") == booking_id:
            booking_db.remove(b)
            return {"message": "Booking cancelled"}
    raise HTTPException(status_code=404, detail="Booking not found")

# ---------- Diagnostic & status ----------
@app.get("/_status")
def status():
    return {"ok": True, "calls_in_memory": len(call_sessions), "bookings": len(booking_db)}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})
