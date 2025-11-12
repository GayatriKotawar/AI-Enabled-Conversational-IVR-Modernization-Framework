import React, { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { PhoneCall, Headphones, Waves } from "lucide-react";

export default function App() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [callInProgress, setCallInProgress] = useState(false);

  const backendURL = "https://34VjF6WvHhamggrU50GoJxVZI2o_2ruF2UBmCWRuCTTA8Y5o3.ngrok-free.app";

  const fetchLogs = async () => {
    try {
      const response = await axios.get(`${backendURL}/twilio/logs`);
      setLogs(response.data);
    } catch (err) {
      console.error("Error fetching logs:", err);
    }
  };

  const triggerTestCall = async () => {
    setLoading(true);
    try {
      await axios.post(`${backendURL}/twilio/test-call`, {
        to: "+18314805664", // your Twilio number
      });
      setCallInProgress(true);
    } catch (err) {
      console.error("Error starting call:", err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-800 text-white font-sans">
      <header className="text-center py-10">
        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-5xl font-bold text-red-400 drop-shadow-lg"
        >
          Air India Voice Assistant
        </motion.h1>
        <p className="text-lg mt-3 text-gray-300">
          “Welcome aboard! Experience next-gen conversational flying.”
        </p>
      </header>

      <main className="max-w-4xl mx-auto p-6 space-y-8">
        {/* Control Section */}
        <div className="bg-white/10 p-6 rounded-2xl shadow-lg backdrop-blur-md">
          <h2 className="text-2xl mb-4 font-semibold text-red-300 flex items-center gap-2">
            <PhoneCall /> Call Controls
          </h2>
          <button
            onClick={triggerTestCall}
            disabled={loading}
            className="bg-gradient-to-r from-red-400 to-pink-500 px-6 py-3 rounded-xl font-bold hover:scale-105 transition"
          >
            {loading ? "Connecting..." : "Start IVR Test Call"}
          </button>

          {callInProgress && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 flex items-center gap-3 text-green-400"
            >
              <Waves className="animate-pulse" /> Call in progress with AI voice...
            </motion.div>
          )}
        </div>

        {/* Live Logs */}
        <div className="bg-white/10 p-6 rounded-2xl shadow-lg backdrop-blur-md">
          <h2 className="text-2xl mb-4 font-semibold text-blue-300 flex items-center gap-2">
            <Headphones /> Live IVR Logs
          </h2>
          <div className="max-h-64 overflow-y-auto bg-black/30 p-4 rounded-lg space-y-2">
            {logs.length > 0 ? (
              logs.map((log, idx) => (
                <p key={idx} className="text-sm text-gray-200">
                  {log.timestamp}: {log.message}
                </p>
              ))
            ) : (
              <p className="text-gray-400">No logs yet — waiting for call activity...</p>
            )}
          </div>
        </div>
      </main>

      <footer className="text-center text-gray-400 mt-12 py-6 text-sm">
        © 2025 Air India AI Systems — Powered by Twilio & OpenAI Voice.
      </footer>
    </div>
  );
}
