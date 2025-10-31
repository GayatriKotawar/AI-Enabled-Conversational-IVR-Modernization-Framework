from fastapi import FastAPI, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from twilio.twiml.voice_response import VoiceResponse, Gather
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/")
def read_root():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing API and data models remain unchanged

class Booking(BaseModel):
    flight_id: int
    passenger_name: str = Field(..., min_length=2)
    seats: int = Field(1, gt=0)

class BookingMenu(BaseModel):
    booking_id: str
    trsna_id: str
    passenger_fullname: str
    passenger_contact: str

booking_db = []

@app.get("/")
def read_root():
    return {"message": "HELLO! Welcome to flight booking system."}

@app.get("/home")
def first_home():
    return {
        "message": "Welcome to Air India customer support IVR, Press 1 for booking, press 2 for booking-menu "
    }

@app.get("/booking_menu")
def booking_menu():
    return {
        "menu": "Booking menu",
        "option": [
            "1. Domestic",
            "2. International"
        ]
    }

@app.get("/status_menu")
def status_menu():
    return {
        "menu": "Status Menu",
        "option": ["Enter the flight id to check the status"]
    }

@app.get("/domestic_booking")
def domestic_booking():
    return {"message": "Domestic booking flow started"}

@app.get("/international_booking")
def international_booking():
    return {"message": "International booking flow started"}

@app.get('/flights/{flight_id}')
def get_flight(flight_id: int, details: bool = False):
    return {"flight_id": flight_id, "details": details}

@app.post("/booking", status_code=201)
def create_booking(booking: Booking):
    return {"message": "Booked", "booking": booking.dict()}

# Twilio IVR endpoints

@app.post("/ivr")
async def ivr(request: Request):
    """Main IVR entry point - returns TwiML with menu prompt"""
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/handle-key", method="POST")
    gather.say("Welcome to Air India Airlines. Press 1 for booking enquiry. Press 2 for flight status.")
    response.append(gather)
    response.say("We did not receive any input. Goodbye!")
    return Response(content=str(response), media_type="application/xml")

@app.post("/handle-key")
async def handle_key(request: Request, menu: str = Form("main-menu")):
    """Handle DTMF input from Twilio gather"""
    form = await request.form()
    digits = form.get("Digits", "")
    if menu == "main-menu":
        if digits == "1":
            # Return booking menu response as JSON
            return booking_menu()
        elif digits == "2":
            return status_menu()
    elif menu == "booking-menu":
        if digits == "1":
            return domestic_booking()
        elif digits == "2":
            return international_booking()

    # Default fallback - if input invalid, replay main menu TwiML response
    response = VoiceResponse()
    response.say("Invalid input. Please try again.")
    response.redirect("/ivr")
    return Response(content=str(response), media_type="application/xml")

@app.put("/update_booking/{booking_id}")
def update_booking(booking_id: str, details: BookingMenu):
    return {
        "message": f"Booking {booking_id} updated",
        "data": details
    }
from fastapi import FastAPI, HTTPException, Request, Response, Form
from pydantic import BaseModel, Field
import logging
from twilio.twiml.voice_response import VoiceResponse, Gather

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Flights and booking data - unchanged
flights = (
    {"flights_id": "AI1", "origin": "Mumbai", "destination": "Chennai", "status": "confirmed"},
    {"flights_id": "AI2", "origin": "Chennai", "destination": "Kochi", "status": "Delayed"},
    {"flights_id": "AI3", "origin": "Delhi", "destination": "Bangalore", "status": "Cancelled"},
)

booking_db = []
call_sessions = {}


@app.get("/flight/{flight_id}")
def get_flight(flight_id: str):
    flight_statuses = {
        "AI1": "Confirmed",
        "AI2": "Cancelled",
        "AI3": "Delayed",
        "AI4": "Confirmed"
    }
    if flight_id not in flight_statuses:
        raise HTTPException(status_code=404, detail="Flight not found")
    return {
        "flight_id": flight_id,
        "status": flight_statuses[flight_id]
    }


@app.get("/status/{flight_id}")
def get_flight_status(flight_id: str):
    for f in flights:
        if f["flights_id"] == flight_id:
            return {
                "status": f["status"],
                "origin": f["origin"],
                "destination": f["destination"]
            }
    raise HTTPException(status_code=404, detail="Flight not found")


@app.get("/active_flights")
def active_flight():
    return {"active_flight": flights}


@app.delete("/cancel_flight/{booking_id}")
def cancel_booking(booking_id: str):
    for b in booking_db:
        if b.get("booking_id") == booking_id:
            booking_db.remove(b)
            return {"message": "Booking cancelled"}
    raise HTTPException(status_code=404, detail="Booking not found")


@app.exception_handler(Exception)
async def handle_exceptions(request: Request, exc: Exception):
    logging.error(f"Error occurred: {exc}")
    return Response(
        content='{"detail": "Internal server error"}',
        media_type="application/json",
        status_code=500
    )


@app.post("/ivr/step1")
def step1(calls_id: str = Form(...), origin: str = Form(...)):
    call_sessions[calls_id] = {"origin": origin}
    return {"message": "Origin saved, please provide destination"}


@app.post("/ivr/step2")
def step2(calls_id: str = Form(...), destination: str = Form(...)):
    session = call_sessions.get(calls_id)
    if session:
        session["destination"] = destination
        origin = session.get("origin", "")
        return {"message": f"You are flying from {origin} to {destination}"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# Twilio IVR endpoints replacing ACS logic

@app.post("/ivr")
async def ivr_call(request: Request):
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/handle-key", method="POST")
    gather.say("Welcome to Air India. Press 1 for booking enquiry. Press 2 for flight status.")
    response.append(gather)
    response.say("No input received. Goodbye!")
    return Response(content=str(response), media_type="application/xml")


@app.post("/handle-key")
async def handle_key(request: Request):
    form = await request.form()
    digits = form.get("Digits", "")

    response = VoiceResponse()
    if digits == "1":
        response.say("You selected booking enquiry.")
        # You can add further prompt or redirect here
    elif digits == "2":
        response.say("You selected flight status.")
        # You can prompt for flight id or redirect to status menu here
    else:
        response.say("Invalid input. Please try again.")
        response.redirect("/ivr")

    return Response(content=str(response), media_type="application/xml")
from fastapi import FastAPI, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

TWILIO_PHONE_NUMBER = "+18314805664"
YOUR_ENDPOINT_BASE_URL ="https://34Vlq1I1YUWhIM9Rhim9i0mlIFo-dfNnPWrBYY6FdaZ9YBux.ngrok-free.app/twilio/ivr" 
### Replace with your real domain for webhook configuration

active_calls = {}  # Keep track of calls as needed

@app.post("/ivr")
async def ivr_call(request: Request):
    """Twilio webhook endpoint for incoming calls"""
    logging.info("Incoming call received")
    
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/handle-key", method="POST")
    gather.say("Welcome to Air India Airlines. Press 1 for booking enquiry. Press 2 for flight status.")
    response.append(gather)
    response.say("No input received. Goodbye!")
    return Response(content=str(response), media_type="application/xml")


@app.post("/handle-key")
async def handle_key(request: Request):
    """Handle DTMF input from Twilio Gather"""
    form = await request.form()
    digits = form.get("Digits", "")
    call_sid = form.get("CallSid")  # Twilio unique call identifier

    logging.info(f"CallSid {call_sid} - User pressed: {digits}")

    response = VoiceResponse()
    
    if digits == "1":
        logging.info("User selected: Booking Enquiry")
        response.say("You have selected booking enquiry. Our booking team will call you back within 5 minutes. Thank you for calling Air India. Goodbye.")
        response.hangup()
        active_calls.pop(call_sid, None)
    elif digits == "2":
        logging.info("User selected: Flight Status")
        response.say("You have selected flight status. Please visit our website airindia dot com for real-time flight updates. Thank you. Goodbye.")
        response.hangup()
        active_calls.pop(call_sid, None)
    else:
        logging.info(f"Invalid input: {digits}")
        response.say("Sorry, that was not a valid option. Please press 1 for booking enquiry or 2 for flight status.")
        response.redirect("/ivr")

    return Response(content=str(response), media_type="application/xml")
