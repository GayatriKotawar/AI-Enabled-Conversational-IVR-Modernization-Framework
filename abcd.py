# ================== IVR SIMULATOR BACKEND (UNIFIED) ==================
# Works locally and supports Twilio-like IVR flow
# Air India Airlines demo IVR system

from fastapi import FastAPI, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from twilio.twiml.voice_response import VoiceResponse, Gather
import random
import logging

app = FastAPI(title="Air India IVR System", version="2.0.0")
logging.basicConfig(level=logging.INFO)

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or restrict to localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DATA MODELS ==================
class CallStart(BaseModel):
    caller_number: str

class DTMFInput(BaseModel):
    call_id: str
    digit: str
    current_menu: str

# ================== STORAGE ==================
active_calls = {}
call_history = []

# ================== MENU STRUCTURE ==================
MENU = {
    "main": {
        "prompt": "Welcome to Air India Airlines. Press 1 for Booking Enquiry. Press 2 for Flight Status. Press 9 to speak with an agent.",
        "options": {
            "1": {"action": "goto", "target": "booking", "message": "Booking Enquiry selected."},
            "2": {"action": "goto", "target": "status", "message": "Flight Status selected."},
            "9": {"action": "agent", "message": "Connecting you to an agent. Please hold."}
        },
    },
    "booking": {
        "prompt": "Press 1 for Domestic Flights. Press 2 for International Flights. Press 0 to return to the Main Menu.",
        "options": {
            "1": {"action": "end", "message": "Domestic booking selected. Our agent will contact you shortly."},
            "2": {"action": "end", "message": "International booking selected. Please visit airindia.com."},
            "0": {"action": "goto", "target": "main", "message": "Returning to Main Menu."}
        },
    },
    "status": {
        "prompt": "Please enter your 6-digit PNR number followed by the hash (#) key.",
        "options": {
            "#": {"action": "pnr_lookup", "message": "Checking your PNR details..."}
        },
    },
}

# ================== ROUTES ==================

@app.get("/")
def health():
    return {
        "status": "IVR System Active",
        "active_calls": len(active_calls),
        "total_calls": len(call_history),
    }

# ---------- Simulated Call Start (local only) ----------
@app.post("/ivr/start")
def start_call(data: CallStart):
    call_id = f"CALL_{random.randint(100000, 999999)}"
    active_calls[call_id] = {
        "caller_number": data.caller_number,
        "start_time": datetime.now().isoformat(),
        "menu": "main",
        "pnr": "",
    }
    return {
        "call_id": call_id,
        "prompt": MENU["main"]["prompt"],
        "status": "connected",
    }

# ---------- Process DTMF for local simulation ----------
@app.post("/ivr/dtmf")
def handle_dtmf(input: DTMFInput):
    call_id = input.call_id
    digit = input.digit
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    call = active_calls[call_id]
    menu = call["menu"]
    structure = MENU[menu]

    # Handle PNR digits
    if menu == "status" and digit != "#":
        call["pnr"] += digit
        if len(call["pnr"]) < 6:
            return {"prompt": f"You entered {digit}. Continue entering your PNR."}

    # Handle menu options
    if digit not in structure["options"]:
        return {"prompt": "Invalid input. Please try again."}

    opt = structure["options"][digit]
    action = opt["action"]

    if action == "goto":
        call["menu"] = opt["target"]
        return {"prompt": MENU[opt["target"]]["prompt"]}
    elif action == "end":
        end_call(call_id)
        return {"prompt": opt["message"], "status": "ended"}
    elif action == "agent":
        end_call(call_id)
        return {"prompt": "Please wait while we connect you to an agent.", "status": "transfer"}
    elif action == "pnr_lookup":
        if len(call["pnr"]) == 6:
            msg = f"Your PNR {call['pnr']} is confirmed. Flight AI101 from Mumbai to Delhi."
        else:
            msg = "Invalid PNR entered. Please try again."
        end_call(call_id)
        return {"prompt": msg, "status": "ended"}

# ---------- End Call ----------
@app.post("/ivr/end")
def end_call(call_id: str):
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not active")
    call = active_calls.pop(call_id)
    call["end_time"] = datetime.now().isoformat()
    call_history.append(call)
    return {"message": f"Call {call_id} ended successfully"}

# ================== TWILIO INTEGRATION ==================

@app.post("/twilio/ivr")
async def twilio_ivr(request: Request):
    """Twilio webhook for incoming call"""
    resp = VoiceResponse()
    gather = Gather(num_digits=1, action="/twilio/handle-key", method="POST")
    gather.say(MENU["main"]["prompt"])
    resp.append(gather)
    resp.say("No input received. Goodbye.")
    return Response(content=str(resp), media_type="application/xml")

@app.post("/twilio/handle-key")
async def twilio_handle_key(request: Request):
    """Handle Twilio key press"""
    form = await request.form()
    digit = form.get("Digits", "")
    resp = VoiceResponse()

    if digit == "1":
        resp.say("You selected booking enquiry. Please press 1 for domestic or 2 for international.")
    elif digit == "2":
        resp.say("Please enter your 6-digit PNR followed by the hash key.")
    elif digit == "9":
        resp.say("Connecting you to an available agent. Please hold.")
    else:
        resp.say("Invalid input. Please try again.")
        resp.redirect("/twilio/ivr")

    return Response(content=str(resp), media_type="application/xml")

# ================== END ==================
