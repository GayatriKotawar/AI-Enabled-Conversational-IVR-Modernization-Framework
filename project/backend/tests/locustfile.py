from locust import HttpUser, task, between

class IVRUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_home(self):
        self.client.get("/")

    @task
    def test_handle_key(self):
        self.client.post("/handle-key", data={"Digits": "1"})
