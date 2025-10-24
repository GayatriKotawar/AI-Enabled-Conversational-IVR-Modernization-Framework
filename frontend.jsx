import React, { useState } from "react";

function App() {
  const [message, setMessage] = useState("");

  async function fetchHome() {
    const response = await fetch("http://127.0.0.1:8000/home");
    const data = await response.json();
    setMessage(data.message);
  }
return (
    <div style={{ padding: 20 }}>
      <h1>Air India IVR Frontend</h1>
      <button onClick={fetchHome}>Get Home Message</button>
      <p>Hello world</p>
    </div>
  );
}

export default App;