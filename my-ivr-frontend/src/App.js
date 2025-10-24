// Add Google Fonts and Material Symbols links to public/index.html, as before

import React, { useState } from 'react';

function Icon({ name, style = {}, ...props }) {
  return <span className="material-symbols-outlined" style={{ fontSize: 22, verticalAlign: "middle", ...style }} {...props}>{name}</span>;
}

const backendBase = "http://localhost:8000";

const today = new Date().toISOString().split('T')[0];

export default function App() {
  // State variables for search, booking, status, etc.
  const [tab, setTab] = useState("Search Flights");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [depart, setDepart] = useState(today);
  const [returnDate, setReturnDate] = useState("");
  const [adults, setAdults] = useState(1);
  const [classType, setClassType] = useState("Economy");
  const [payBy, setPayBy] = useState("Cash");
  const [concession, setConcession] = useState("None");
  const [promo, setPromo] = useState("");
  const [searchResult, setSearchResult] = useState(null);
  const [error, setError] = useState("");
  // From your previous logic (status check etc.)
  const [flightId, setFlightId] = useState("");
  const [status, setStatus] = useState(null);

  // Tabs
  const mainTabs = ["Search Flights", "Manage Booking", "Check In", "Flight Status"];

  // Handlers
  async function handleFlightSearch(e) {
    e.preventDefault();
    setError("");
    if (!origin || !destination) return setError("Please enter both origin and destination.");
    // Simulate search: In a real system, call backend and handle results dynamically
    setSearchResult({
      origin, destination, depart, returnDate, adults, classType, payBy, concession, promo,
      flights: [
        { id: "AI231", time: "10:00", fare: "₹8,200", avail: true },
        { id: "AI418", time: "15:20", fare: "₹6,490", avail: true }
      ]
    });
  }

  async function handleStatusCheck() {
    setError(""); setStatus(null);
    if (!flightId) return setError("Enter Flight ID (e.g. AI1)");
    try {
      const r = await fetch(`${backendBase}/status/${flightId}`);
      if (!r.ok) return setError("Flight not found");
      setStatus(await r.json());
    } catch {
      setError("Backend off or unreachable!");
    }
  }

  // UI
  return (
    <div style={{
      fontFamily: "'Open Sans', 'Lato', Arial, sans-serif",
      minHeight: "100vh",
      background: "linear-gradient(135deg,#fafbfc,#e8eefa 60%,#f4f7fc 100%)",
      margin: 0, padding: 0
    }}>
      {/* Air India Top Nav (Logo + Tabs) */}
      <nav style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "#fff", borderBottom: "2px solid #f6e6d9", padding: "0 0 0 32px"
      }}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <img src="https://www.airindia.com/o/airindia-theme/images/air-india-logo.svg"
            alt="air india" height={38} style={{ margin: "16px 0", marginRight: 18 }} />
          <div style={{ fontWeight: 800, fontSize: 21, color: "#d13a2f", letterSpacing: 1 }}>AIR INDIA</div>
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center", marginRight: 24 }}>
          {["BOOK & MANAGE", "WHERE WE FLY", "PREPARE TO TRAVEL", "AIR INDIA EXPERIENCE", "MAHARAJA CLUB"].map((t) =>
            <span key={t} style={{ color: "#253858", fontWeight: 500, fontSize: 15 }}>{t}</span>
          )}
          <Icon name="help" style={{ color: "#b91c1c" }} />
          <span style={{ color: "#d13a2f", fontWeight: 600 }}>SIGN IN</span>
        </div>
      </nav>
      {/* Page SubNav Tabs */}
      <div style={{
        display: "flex",
        background: "#fff",
        boxShadow: "0px 2px 8px #e6e4ee1f",
        borderBottom: "2px solid #f2efe2"
      }}>
        {mainTabs.map(t => (
          <button
            onClick={() => setTab(t)}
            key={t}
            style={{
              padding: "17px 24px", fontWeight: 600, color: tab === t ? "#d13a2f" : "#233154",
              border: "none", background: "none", fontSize: 16, borderBottom: tab === t ? "2.5px solid #d13a2f" : "2.5px solid transparent",
              outline: "none", cursor: "pointer"
            }}
          >{t}</button>
        ))}
      </div>
      {/* Search & Hero Content */}
      {tab === "Search Flights" && (
        <>
          <form onSubmit={handleFlightSearch} style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr 1fr auto",
            gap: 10,
            background: "#fff",
            borderRadius: 8,
            margin: "26px auto 0 auto",
            maxWidth: 1130,
            padding: "15px 15px 14px 15px",
            boxShadow: "0 2px 14px #eee"
          }}>
            <div style={{ display: "flex", flexDirection: "column", fontWeight: 500 }}>
              <span style={{ fontSize: "13px", color: "#54595d", marginBottom: 2 }}>FROM</span>
              <input value={origin} onChange={e => setOrigin(e.target.value)} placeholder="Origin" style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6 }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", fontWeight: 500 }}>
              <span style={{ fontSize: "13px", color: "#54595d", marginBottom: 2 }}>TO</span>
              <input value={destination} onChange={e => setDestination(e.target.value)} placeholder="Destination" style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6 }} />
            </div>
            <div>
              <div><span style={{ fontSize: 13, color: "#54595d", marginBottom: 2 }}>Depart</span></div>
              <input type="date" value={depart} onChange={e => setDepart(e.target.value)} style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6, width: "90%" }} />
            </div>
            <div>
              <div><span style={{ fontSize: 13, color: "#54595d" }}>Return</span></div>
              <input type="date" value={returnDate} onChange={e => setReturnDate(e.target.value)} style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6, width: "90%" }} />
            </div>
            <div>
              <div><span style={{ fontSize: 13, color: "#54595d" }}>Passenger(s)</span></div>
              <select value={adults} onChange={e => setAdults(e.target.value)} style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6 }}>
                {[1, 2, 3, 4, 5].map(a => <option key={a} value={a}>Adult {a}</option>)}
              </select>
            </div>
            <div>
              <div><span style={{ fontSize: 13, color: "#54595d" }}>Class</span></div>
              <select value={classType} onChange={e => setClassType(e.target.value)} style={{ padding: 7, border: "1px solid #ddd", borderRadius: 6 }}>
                <option>Economy</option>
                <option>Business</option>
              </select>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              <input value={promo} onChange={e => setPromo(e.target.value)} placeholder="Add Promo Code" style={{ border:"1px solid #d6b8b7", borderRadius:6, padding:7, width:118, marginBottom:4}} />
              <button type="submit" style={{
                background:"#d13a2f", color:"#fff", padding:"11px 0", border:"none", borderRadius:7,
                fontWeight:700, cursor:"pointer", fontSize:17, width: 118, marginTop:2
              }}>Search</button>
            </div>
          </form>
          <div style={{ color: "#b91c1c", fontWeight: 600, marginTop: 10, textAlign: "center" }}>{error}</div>
          {/* Results */}
          {searchResult && (
            <div style={{maxWidth:840,margin:"18px auto",background:"#fff",padding:24,borderRadius:12,boxShadow:"0 2px 8px #eee"}}>
              <b>Flights from {searchResult.origin} to {searchResult.destination}:</b>
              {searchResult.flights.map(f =>
                <div key={f.id} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  marginTop: 10, padding: "10px 0", borderBottom: "1px solid #f2eee8"
                }}>
                  <span style={{ fontWeight: 600, color: "#d13a2f", fontSize: 18 }}>{f.id}</span>
                  <span>{f.time}</span>
                  <span style={{ color: "#0a5", fontWeight: 600 }}>{f.fare}</span>
                  <button style={{
                    background: "#185a9d", color: "#fff", border: "none", borderRadius: 7, padding: "6px 18px",
                    fontWeight: 700, cursor: "pointer"
                  }}>Book</button>
                </div>
              )}
            </div>
          )}
          {/* Hero and Banner */}
          <div style={{
            display: "flex", flexDirection: "row", alignItems: "center", margin: "60px 0 36px 0", justifyContent: "center", gap:32
          }}>
            <div style={{
              background: "linear-gradient(95deg,#e83933 10%,#ec9a22 100%)",
              color: "#fff", padding: "38px 40px", borderRadius: 23,
              fontWeight: 800, fontSize: 30, width: 510, maxWidth: "90%", boxShadow: "0 10px 48px #d13a2f22",
              display: "flex", flexDirection: "column", gap: 10
            }}>
              ENJOY UP TO 15% OFF <br />AND ZERO CONVENIENCE FEE
              <span style={{ fontWeight: 400, fontSize: 15, marginTop: 15 }}>Exclusively for logged in Maharaja Club members</span>
              <button style={{
                background: "#fff", color: "#d13a2f", fontWeight: 700, borderRadius: 8, padding: "10px 30px",
                border: "none", marginTop: 18, fontSize: 16, cursor: "pointer"
              }}>Learn More</button>
            </div>
            <img src="https://www.airindia.com/o/airindia-theme/images/slider/600x400-air-india12.jpg" alt="hero" style={{ height: 220, borderRadius: 14, boxShadow: "0 2px 16px #0002" }} />
          </div>
          {/* Info Banner and slider dots */}
          <div style={{
            background: "#22243b",
            color: "#fff", fontSize: 17, fontWeight: 500, maxWidth: 570,
            margin: "0 auto", borderRadius: 13, padding: "10px 30px 18px 30px", 
            display:"flex", flexDirection:"column", alignItems:"center"
          }}>
            INFO    <span style={{margin:"0 12px",fontWeight:600}}>5/12</span>
            <div style={{display:"flex",justifyContent:"center",marginTop:9,gap:7}}>
              {[0,1,2,3,4].map((i)=><div key={i} style={{
                width:18,height:7,background:i===1?"#d13a2f":"#fff", borderRadius:9
              }}/>)}
            </div>
          </div>
        </>
      )}

      {/* Other Tab Contents */}
      {tab === "Flight Status" && (
        <div style={{
          background: "#fff", borderRadius: 20,  margin: "55px auto 32px auto",
          maxWidth: 530, padding: "20px 35px", boxShadow: "0 2px 16px #8fdbe811"
        }}>
          <h2 style={{color:"#185a9d",fontSize:25,marginTop:8,marginBottom:".8em"}}><Icon name="search" style={{fontSize:30,marginRight:10}}/>Check Flight Status</h2>
          <div style={{marginBottom:12,fontSize:16}}>Enter Flight ID (e.g. AI1), and instantly get flight info below.</div>
          <input
            value={flightId}
            onChange={e=>setFlightId(e.target.value.toUpperCase())}
            placeholder="Flight ID"
            style={{border:"1.5px solid #ace",borderRadius:11,padding:10,marginRight:6,fontSize:17}}
            onKeyDown={e=>e.key==="Enter"&&handleStatusCheck()}
          />
          <button onClick={handleStatusCheck} style={{color:"#fff",background:"#d13a2f",padding:"9px 28px",border:"none",borderRadius:9,fontWeight:600,fontSize:17,letterSpacing:1}}>
            Check
          </button>
          {error && <div style={{color:"#dc2626",marginTop:8,fontWeight:500}}>{error}</div>}
          {status && <div style={{marginTop:15,background:'#f3faf9',padding:16,borderRadius:12,fontSize:17}}>
            <b>Status:</b> {status.status}<br/>
            <b>Origin:</b> {status.origin}<br/>
            <b>Destination:</b> {status.destination}
          </div>}
        </div>
      )}

      {/* Footer */}
      <footer style={{
        textAlign:"center",
        fontWeight:600, color:"#fff", fontSize:20,
        marginTop:64, background:"linear-gradient(90deg,#43cea2,#185a9d)",
        padding:"2.2rem 0 .9rem 0", letterSpacing:".04em"
      }}>
        Inspired by Air India &mdash; FastAPI + Twilio IVR Case &mdash; {new Date().getFullYear()}
      </footer>
    </div>
  );
}
