import sys
import os
from fastapi.testclient import TestClient

# ✅ Add backend folder to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backendtwilio import app  # import now works

client = TestClient(app)

def test_home_route():
    response = client.get("/")
    assert response.status_code == 200

def test_handle_key_route():
    response = client.post("/handle-key", data={"Digits": "1"})
    assert response.status_code in [200, 201, 400]
