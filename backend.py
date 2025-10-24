## backend - logic + db + api - things that happen here as the behind the scenes 
# frontend- UI 

#backend- frameworks- Fast api, Flask - by python, Node.js , Django 

from fastapi import FastAPI
from fastapi import HTTPException
# from fastapi.responses import Response 

app= FastAPI() # app is instanced 

@app.get("/") ##a route. when someone visits this / endpoint, this function runs
def read_root(): ### this is my function
    return {"message" : "Hello!"}### This is my JSON response 

@app.get("/home")
def first_home():
    return {
        "message" : "Welcome to Air India Customer support IVR, Press 1 for main-menu, Press 2 for booking-menu"
    }

@app.get("/booking_menu")
def booking_menu():
    return{
        "menu" : "Booking Menu",
        "options" : [
            "1. Domestic",
            "2. International"
        ]
    }

@app.get("/status_menu")
def status_menu():
    return {
        "menu" : "Status Menu",
        "options": ["Enter the flight id to check the status"]
    }

@app.get("/domestic_booking")
def domestic_booking():
    return {
        "message" : "Domestic booking flow started"
    }
    
@app.get("/international_booking")
def international_booking():
    return {
        "message" : "International booking flow started"
    }
    
    
# booking_menu- booking_id, trans_id, passenger_full name, passenger_contact 
# status_menu: flight id, origin, destination, status: confirmed or not 
# domestic: origin, destination, date of travel, 
# international- origin, destination, travel, time,,.........

from pydantic import BaseModel ## Base model- data validation library

class BookingMenu(BaseModel): ## defines a data model for your API input 
    #schema for the BookingMenu
    booking_id: str
    trans_id : str
    passenger_fullname: str
    passenger_contact: str ## it return 422 unprocessable entity if the data is missing/wrong data type
    
booking_db= []

@app.post("/handle-key")
def handle_key(Digits: str = "", menu : str = "main"):
    if menu == "main-menu":
        if Digits == "1":
            return booking_menu()
        elif Digits == "2":
            return status_menu()
        
    elif menu == "booking-menu":
        if Digits == "1":
            return domestic_booking()
        elif Digits == "2":
            return international_booking()
    
@app.put("/update_booking/{booking_id}") ## Put method- modifies the existing resource
def update_booking(booking_id: str, details: BookingMenu): ## booking_id- path parameter
    return {
        "message": "Booking {booking_id} updated", 
        "data" : details ## validating the body from the BOOKING MENU base model class
    }



# HTTP Status code- 200, 404, 402, 500, 

@app.get("/flight/{flight_id}")
def get_flight(flight_id: str):
    flights= {
        "AI1": "Confirmed", 
        "AI2" : "Cancelled", 
        "AI3" : "Delayed",
        "AI4" : "Confirmed"
    }
    if flight_id not in flights:
        raise HTTPException(status_code= 404, detail= "Flight not found")## returns error with the given detail/code, status code is a numeric code, details- custom message
    return {
        "flight_id" : flight_id,
        "status" : flights[flight_id]
    }

flights={
    {"flight_id": "AI1", "origin": "Mumbai", "destination": "Chennai", "status": "Confirmed"},
    {"flight_id": "AI2", "origin": "Chennai", "destination": "Kochi", "status": "Delayed"},
    {"flight_id": "AI3", "origin": "Delhi", "destination": "Mumbai", "status": "Cancelled"},
    {"flight_id": "AI4", "origin": "Kochi", "destination": "Bengaluru", "status": "Confirmed"},
    {"flight_id": "AI5", "origin": "Hyderabad", "destination": "Goa", "status": "Delayed"}
}

@app.get("/status/{flight_id}")
def get_flight_status(flight_id:str):
    for f in flights:
        if f[flight_id] == flight_id:
            return {
                "status": f["status"],## if its AI5- delayed, AI3- cancelled
                "origin" : f["origin"],
                "destination" : f["destination"]
            }
    raise HTTPException(status_code=404, detail= "Flight no found")

@app.get("/active_flights")
def active_flights():
    return {
        "active_flights": flights
    }


### HTTP method- delete 

@app.delete("/cancel_flight/{flight_id}")
def cancel_booking(booking_id: str):
    for b in booking_db: 
        if b["booking_id"] == booking_id:
            booking_db.remove(b)
            return {
                "message": "Booking cancelled"
            }
        raise HTTPException(status_code= 404, detail= "Booking not found")
            
        
#####Error & Logging support--- this ensure the backend doesn't crash and logs your errors

import logging 
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)

@app.exception_handler(Exception)
def handle_exceptions(request, exc):
    logging.error(f" Error occurred: {exc}")
    return JSONResponse(status_code=500, content= {"detail": "Internal server error"})


####server-side session- session management, as we are working with the IVR, it;s stateful(conversation has context). HTTPS are statless(each request os independent)
###call session storage- 
###Problem- user makes multiple requests during one call. How do you remember their previous answers? 
#Solution

call_sessions= {}

@app.post("/ivr/step1")
def step1 (calls_id:str, origin: str):
    call_sessions[calls_id]= {"origin" : origin}
    return{
        "message" : "origin saved, please provide destination"
    }

@app.post("/ivr/step2")
def step2(calls_id: str, origin: str, destination: str):
    session= call_sessions.get(calls_id)
    session["destination"] = destination

    origin = session["origin"]
    destination = session["destination"]
    
    return {
        "message" : "You are flying from {origin} to {destination}"
    }
    


####Simple_ivr using the ACS services


# pip install azure-communication-callautomation
# pip install azure-identity
# pip install fastapi
# pip install pydantic 
# pip install mysql-connector-python

# Simple IVR Flow Overview:

# 1. User dials your Azure phone number (+1234567890)
#        ↓
# 2. Azure receives the call
#        ↓
# 3. Azure sends HTTP POST to our /acs/incoming-call endpoint
#        ↓
# 4. Our FastAPI code answers the call (call_automation_client.answer_call)
#        ↓
# 5. Azure connects the call - User can now hear you
#        ↓
# 6. Our code plays welcome message using TTS (Text-to-Speech)
#        ↓
# 7. Our code starts listening for DTMF (key presses)
#        ↓
# 8. User presses 1
#        ↓
# 9. Azure sends event to your /acs/callbacks endpoint
#        ↓
# 10. Our code receives "user pressed 1" event
#        ↓
# 11. Our code decides: "1 means booking menu"
#        ↓
# 12. Our code plays booking menu
#        ↓
# ... and so on



from fastapi import FastAPI, Request, Response

from azure.communication.callautomation import (
    CallAutomationClient,
    TextSource,
    RecognizeInputType
)

from azure.communication.callautomation.models import RecogineDtmfOptions, DtmfTone
from azure.core.credintials import AzureKeyCredential

# ==================== CONFIGURATION ====================

# Step 1: Get from Azure Portal → Communication Service → Keys
ACS_CONNECTION_STRING = "endpoint=https://YOUR_RESOURCE.communication.azure.com/;accesskey=YOUR_KEY"

# # Step 2: Get from Azure Portal → Communication Service → Phone Numbers
ACS_PHONE_NUMBER = "+1234567890"  # Your Azure phone number

# # Step 3: Get from ngrok after running: ngrok http 8000- https://127.0.0.1:8000
CALLBACK_URI = "https://YOUR_NGROK_URL.ngrok.io"
# =======================================================
## Initialize the all azure clients

#call_automation_client= CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)
## This is your remote control for all the calls

app = FastAPI()# normal fastapi app instance

active_calls= {} # to store simple in-memory call storage, store active calls (in-production, or us can also use a particular DB)

###Helper functions

def get_call_connection(call_id: str):# control panel for a specific call
    ## Getting a control of a specific calls
    ## Args: call_id: Unique identifier of the call
    ## returns: CallConnectionClient to control this call using that particular call_id
    return call_automation_client.get_call_connection(call_id)

def create_voice_prompt(text:str):
    ## Converts my text to speech 
    ## Args: text: what you want the system to say
    ## Return: TextSource Object that Azure can speak out loud
    return TextSource(
        text= "Welcome to Air India Customer Support",
        voice_name= "en-IN-NeerjaNeural" ## this you can get it from the azure portal, this is the indian english female voice
    )
# Other Available Voices:

# en-IN-NeerjaNeural - Female, Indian English
# en-IN-PrabhatNeural - Male, Indian English
# en-US-JennyNeural - Female, US English

###Endpoints

@app.get("/")
def home():
    ## Test endpoint to check if the server is running 
    return{
        "status" : "IVR system is running", 
        "active_calls": len(active_calls),
        "platform": "Azure Communication Service"
    }


####answer the incoming call

@app.post("/acs/incoming_call")
async def handle_incoming_call(request: Request):
    ## This is called by azure services when someone/user calls your number
    ###what happens here: 4 actions to do
    # 1. azure sends your call info
    # 2. you answer the call 
    # 3. you play the welcome message
    # 4. you start listening for key presses
    
    print("=" * 50)
    print("Incoming call received")
    print("=" * 50)
    
    #get the data azure sent us 
    event_data= await request.json()
    print("Event data: {event_data}")
    
    ## extract the important information 
    incoming_call_context= event_data["data"]["incomingCallContext"]
    caller_number = event_data["data"]["from"]["phoneNumber"]["value"]
    
    print("call from : {caller_number}")
    
    ## step 2: answer the call 
    
    try: 
        print("Answering call...")
        call_properties= call_automation_client.answer_call(
            incoming_call_context= incoming_call_context,
            callback_url= {CALLBACK_URI}/acs/callbacks
        )
        
        ## get the unique id for this call, to store it in our active calls
        call_id= call_properties.call_connection_id
        print("Call answered! Call_id: {call_id}")
        
        ## store the call information
        active_calls[call_id]={
            "caller_number": caller_number, 
            "status": "in-progress"
        }
        
        ##play welcome message and get the input 
        play_welcome_menu(call_id)
        
        return{
            "status": "call answered successfully",
            "call_id": call_id
        }
        
    except Exception as e:
        print("Error answering call: {e}")
        raise HTTPException(status_code=500, detail= "Error answering the call")
    
##########stopped yesterday here- 14/10
    
#### write/play the welcome message function/menu


def play_welcome_menu(call_id: str):
    """
    Play the main menu and listen for DTMF input
    
    This function:
    1. Creates a voice prompt
    2. Tells Azure to play it
    3. Tells Azure to listen for key presses (1 or 2 or 3 or 4 or so on...)
    """
    
    print("Playing welcome menu for call {call_id}")
    
    # Get control of this call
    call_connection = get_call_connection(call_id)
    
    # Create the welcome message
    welcome_text = """
    Welcome to Air India Airlines. 
    Press 1 for booking enquiry. 
    Press 2 for flight status.
    """
    
    voice_prompt = create_voice_prompt(welcome_text)
    
    # Configure how to collect DTMF input
    dtmf_options = RecognizeDtmfOptions(
        max_tones_to_collect=1,  # We want 1 digit (1 or 2)
        initial_silence_timeout_in_seconds=10,  # Wait 10 seconds for input
        inter_digit_timeout_in_seconds=5,  # Not needed (only 1 digit)
        interrupt_prompt=True,  # Stop talking if user presses key
        stop_dtmf_tones=[DtmfTone.POUND]  # Stop if user presses #
    )
    #In backend: the system plays the prompts, waits for the user to press a key, 
    # once user pressed, azure sends that event to /acs/callbacks endpoint with the digit
    
    # Tell Azure: Play this message and listen for key press
    call_connection.start_recognizing_media(
        input_type=RecognizeInputType.DTMF,
        target_participant=call_properties.target_participant,
        recognize_options=dtmf_options,
        play_prompt=voice_prompt,
        operation_context="main_menu"  # Tag so we know it's main menu
    )
    # This function returns immediately - the call is still live
    # Azure will send events to /acs/callbacks when something happens
    # doesnt wait for the user to press a key
    
    print("Welcome menu playing and listening for input...")

### Handling user input

@app.post("/acs/callbacks")
async def handle_callbacks(request: Request):
    
    # THIS IS CALLED BY AZURE WHEN SOMETHING HAPPENS- even if there is a user input or call ended or timeout
    
    # Events Azure sends here:
    # - User pressed a key (RecognizeCompleted)
    # - User didn't press anything (RecognizeFailed)
    # - Audio finished playing (PlayCompleted)
    # - Call ended (CallDisconnected)
    
    # This is the BRAIN of your IVR - routes user based on input
        
    print("\n" + "=" * 50)
    print("CALLBACK RECEIVED FROM AZURE")
    print("=" * 50)
    
    # Azure can send multiple events at once
    events = await request.json()
    
    for event in events:
        event_type = event.get("type")
        event_data = event.get("data", {})
        
        print("Event Type: {event_type}")
        
        # ========== USER PRESSED A KEY ==========
        if event_type == "Microsoft.Communication.RecognizeCompleted":
            
            call_id = event_data.get("callConnectionId")
            operation_context = event_data.get("operationContext")
            
            # Extract the key they pressed
            recognize_result = event_data.get("recognizeResult", {})
            dtmf_result = recognize_result.get("dtmfResult", {})
            tones = dtmf_result.get("tones", [])
            
            # Convert to digit (Azure sends "one", "two", etc.)
            user_input = convert_tone_to_digit(tones[0]) if tones else ""
            
            print("User pressed: {user_input}")
            print("Menu context: {operation_context}")
            
            # Route based on which menu they were in
            if operation_context == "main_menu":
                handle_main_menu_input(call_id, user_input)
        
        # ========== USER DIDN'T PRESS ANYTHING (TIMEOUT) ==========
        elif event_type == "Microsoft.Communication.RecognizeFailed":
            
            call_id = event_data.get("callConnectionId")
            print("Timeout - no input received for call {call_id}")
            
            # Play timeout message
            call_connection = get_call_connection(call_id)
            timeout_prompt = create_voice_prompt(
                "I didn't receive any input. Let me repeat the options."
            )
            call_connection.play_media(play_source=timeout_prompt)
            
            # Replay the welcome menu
            play_welcome_menu(call_id)
        
        # ========== CALL ENDED ==========
        elif event_type == "Microsoft.Communication.CallDisconnected":
            
            call_id = event_data.get("callConnectionId")
            print("Call {call_id} disconnected")
            
            # Clean up
            if call_id in active_calls:
                del active_calls[call_id]
                print("Session cleaned up")
    
    return Response(status_code=200)

## Handle main menu input

def handle_main_menu_input(call_id: str, user_input: str):
    """
    Process what user pressed in main menu
    
    1 = Booking enquiry
    2 = Flight status
    Other = Invalid
    """
    
    print("Processing main menu input: {user_input}")
    
    call_connection = get_call_connection(call_id)
    
    if user_input == "1":
        # User wants booking enquiry
        print("User selected: Booking Enquiry")
        
        response_text = """
        You have selected booking enquiry. 
        Our booking team will call you back within 5 minutes. 
        Thank you for calling Air India. Goodbye.
        """
        
        response_prompt = create_voice_prompt(response_text)
        call_connection.play_media(play_source=response_prompt)
        
        # Hang up after message plays (in production, add delay)
        call_connection.hang_up(is_for_everyone=True)
        
        # Clean up
        if call_id in active_calls:
            del active_calls[call_id]
    
    elif user_input == "2":
        # User wants flight status
        print("User selected: Flight Status")
        
        response_text = """
        You have selected flight status. 
        Please visit our website airindia dot com for real-time flight updates. 
        Thank you. Goodbye.
        """
        
        response_prompt = create_voice_prompt(response_text)
        call_connection.play_media(play_source=response_prompt)
        
        # Hang up
        call_connection.hang_up(is_for_everyone=True)
        
        # Clean up
        if call_id in active_calls:
            del active_calls[call_id]
    
    else:
        # Invalid input
        print("Invalid input: {user_input}")
        
        error_text = """
        Sorry, that was not a valid option. 
        Please press 1 for booking enquiry or 2 for flight status.
        """
        
        error_prompt = create_voice_prompt(error_text)
        call_connection.play_media(play_source=error_prompt)
        
        # Replay menu
        play_welcome_menu(call_id)
        
## utility function 
## converting the azure dtmf tine name to actual digit 
## azure sends: "one", "two", ..... but we need, 1, 2, 3,....
def convert_tone_to_digit(tone: str) -> str:
    tone_map={
        "zero": "0",
        "one": "1",
        "two": "2",
        # so on till nine
        "pound": "#",
        "asterisk": "*"
    }
    
    return tone_map.get(tone.lower(), "")