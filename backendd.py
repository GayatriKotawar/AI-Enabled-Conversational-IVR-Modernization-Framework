from fastapi import FastAPI, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field
from twilio.twiml.voice_response import VoiceResponse, Gather
import logging

# ... your model/classes and other endpoint definitions ...
app = FastAPI()
# ... CORS and logging middleware here ...

# Your existing routes here

@app.get("/vxml", response_class=FastAPIResponse)
def vxml_ivr():
    vxml = '''<?xml version="1.0" encoding="UTF-8"?>
<vxml version="2.1">
  <form id="main-menu">
    <block>
      <prompt>
        Hello, welcome to the Air India airline customer support IVR.
        Press 1 for booking enquiry. Press 2 for flight status.
      </prompt>
    </block>
    <field name="choice">
      <prompt>
        Press 1 for booking, Press 2 for flight status, Press 3 for luggage allowance.
      </prompt>
      <grammar type="application/grammar+xml" mode="dtmf">
        1|2|3
      </grammar>
      <filled>
        <if cond="choice==1">
          <goto next="#booking"/>
        <elseif cond="choice==2"/>
          <goto next="#flight-status"/>
        <elseif cond="choice==3"/>
          <goto next="#luggage"/>
        <else/>
          <prompt>Invalid choice, please try again.</prompt>
        </if>
      </filled>
    </field>
  </form>
  <!-- Extend with other forms -->
</vxml>
    '''
    return FastAPIResponse(content=vxml, media_type="application/xml")

from flask import Flask, request, redirect, url_for, Response
from twilio.twiml.voice_response import VoiceResponse, Gather

app = Flask(__name__)

@app.route("/ivr", methods=['GET', 'POST'])
def ivr_welcome():
    """IVR entry point: Plays main menu and gathers input"""
    response = VoiceResponse()
    gather = Gather(num_digits=1, action='/menu', method='POST')
    gather.say(
        "Welcome to Air India Airlines. "
        "Press 1 for booking enquiry. "
        "Press 2 for flight status.",
        voice='alice', language='en-IN'
    )
    response.append(gather)
    response.redirect('/ivr')  # If no input received, repeat menu
    return Response(str(response), mimetype="application/xml")

@app.route("/menu", methods=['POST'])
def ivr_menu():
    """Handles the first menu selection"""
    digits = request.form.get('Digits')
    response = VoiceResponse()

    if digits == "1":
        response.say(
            "You have selected booking enquiry. "
            "Our booking team will call you back within 5 minutes. "
            "Thank you for calling Air India. Goodbye.",
            voice='alice', language='en-IN'
        )
        response.hangup()
    elif digits == "2":
        response.say(
            "You have selected flight status. "
            "Please visit our website airindia dot com for real-time flight updates. "
            "Thank you. Goodbye.",
            voice='alice', language='en-IN'
        )
        response.hangup()
    else:
        response.say(
            "Sorry, that was not a valid option. "
            "Please press 1 for booking enquiry or 2 for flight status.",
            voice='alice', language='en-IN'
        )
        response.redirect('/ivr')
    return Response(str(response), mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
