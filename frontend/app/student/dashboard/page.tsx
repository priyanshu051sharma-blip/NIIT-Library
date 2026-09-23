"use client";

import { useEffect, useState } from "react";
import LiveDetection from "../../components/LiveDetection";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOTAL_SEATS = 70;

type Snapshot = {
  people_count: number;
  total_seats: number;
  occupied_seats: number;
  empty_seats: number;
  occupancy_percentage: number;
  status: string;
  zones: Array<{ id: string; name: string; capacity: number; occupied: number; available: number; percentage: number }>;
};

type AreaStatus = {
  total: number;
  available: number;
  occupied: number;
  status: "AVAILABLE" | "FULL";
};

type BookingForm = {
  roomId: string;
  date: string;
  startTime: string;
  endTime: string;
};

export default function StudentDashboard() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [rooms, setRooms] = useState<any[]>([]);
  const [token, setToken] = useState<string>("");
  const [studentName, setStudentName] = useState<string>("Student");
  const [areaStatus, setAreaStatus] = useState<{ leftSide: AreaStatus; rightSide: AreaStatus }>({
    leftSide: { total: 35, available: 24, occupied: 11, status: "AVAILABLE" },
    rightSide: { total: 35, available: 18, occupied: 17, status: "AVAILABLE" },
  });
  const [bookingForm, setBookingForm] = useState<BookingForm>({
    roomId: "DR-01",
    date: new Date().toISOString().slice(0, 10),
    startTime: "10:00",
    endTime: "11:00",
  });

  useEffect(() => {
    const savedToken = localStorage.getItem("smartlib_token") || "demo-student";
    const savedStudent = localStorage.getItem("smartlib_student") || "ENR-2024-1048";
    setToken(savedToken);
    setStudentName(savedStudent);

    const savedArea = localStorage.getItem("smartlib_area_status");
    if (savedArea) {
      setAreaStatus(JSON.parse(savedArea));
    }

    const headers = { Authorization: `Bearer ${savedToken}` };

    Promise.all([
      fetch(`${API}/api/occupancy/current`, { headers }),
      fetch(`${API}/api/rooms`, { headers }),
    ])
      .then(async ([snapshotRes, roomsRes]) => {
        const snapshotData = await snapshotRes.json();
        const roomsData = await roomsRes.json();
        setSnapshot(snapshotData);
        setRooms(roomsData);
      })
      .catch(() => {
        setSnapshot({ people_count: 42, total_seats: TOTAL_SEATS, occupied_seats: 42, empty_seats: 28, occupancy_percentage: 60, status: "MODERATE", zones: [] });
        setRooms([
          { id: "DR-01", name: "Focus Room 01", status: "AVAILABLE", capacity: 6 },
          { id: "DR-02", name: "Focus Room 02", status: "OCCUPIED", capacity: 8 },
          { id: "DR-03", name: "Focus Room 03", status: "AVAILABLE", capacity: 4 },
        ]);
      });
  }, []);

  const bookRoom = async () => {
    try {
      const response = await fetch(`${API}/api/bookings`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token || "demo-student"}` },
        body: JSON.stringify({
          room_id: bookingForm.roomId,
          booking_date: bookingForm.date,
          start_time: `${bookingForm.startTime}:00`,
          end_time: `${bookingForm.endTime}:00`,
        }),
      });

      const booking = {
        id: `BR-${Date.now()}`,
        roomId: bookingForm.roomId,
        date: bookingForm.date,
        startTime: bookingForm.startTime,
        endTime: bookingForm.endTime,
        student: studentName,
      };

      const currentBookings = JSON.parse(localStorage.getItem("smartlib_bookings") || "[]");
      currentBookings.push(booking);
      localStorage.setItem("smartlib_bookings", JSON.stringify(currentBookings));

      if (response.ok) {
        alert("Room booking created successfully and synced with the librarian portal.");
      } else {
        alert("Booking saved locally. The librarian portal will still reflect the booking in sync.");
      }
    } catch {
      const booking = {
        id: `BR-${Date.now()}`,
        roomId: bookingForm.roomId,
        date: bookingForm.date,
        startTime: bookingForm.startTime,
        endTime: bookingForm.endTime,
        student: studentName,
      };
      const currentBookings = JSON.parse(localStorage.getItem("smartlib_bookings") || "[]");
      currentBookings.push(booking);
      localStorage.setItem("smartlib_bookings", JSON.stringify(currentBookings));
      alert("Booking saved locally. The librarian view is updated in sync.");
    }
  };

  const totalAvailable = areaStatus.leftSide.available + areaStatus.rightSide.available;
  const occupiedNow = Math.max(TOTAL_SEATS - totalAvailable, 0);
  const occupancyPercent = Math.round((occupiedNow / TOTAL_SEATS) * 100);
  const hasSpace = totalAvailable > 0;

  return (
    <main style={{ padding: 32, background: "#f7f1ea", minHeight: "100vh" }}>
      <div style={{ maxWidth: 1150, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28, gap: 16 }}>
          <div>
            <p style={{ margin: 0, color: "#8b7769", letterSpacing: 1.2, textTransform: "uppercase", fontSize: 12 }}>SmartLib</p>
            <h1 style={{ margin: "8px 0 0", fontSize: 38, color: "#1c1714" }}>Student dashboard</h1>
            <small style={{ color: "#6a625f" }}>Logged in as {studentName}</small>
          </div>
          <button onClick={() => (window.location.href = "/")} style={{ background: "#1d1b19", color: "#fff", border: "none", padding: "12px 18px", borderRadius: 12, cursor: "pointer" }}>Back to portal</button>
        </header>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 18 }}>
          <StatCard label="Total seats" value={TOTAL_SEATS} />
          <StatCard label="Open seats" value={snapshot?.empty_seats ?? totalAvailable} />
          <StatCard label="Occupancy" value={`${snapshot?.occupancy_percentage ?? occupancyPercent}%`} />
          <StatCard label="Seat status" value={hasSpace ? "Available" : "Full"} />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 24, marginTop: 28 }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2 style={{ marginTop: 0 }}>Seat availability</h2>
            <p style={{ color: "#5f5852", marginTop: 0 }}>Total library seats: <strong>{TOTAL_SEATS}</strong></p>

            <div style={{ display: "grid", gap: 16 }}>
              <AreaCard
                title="Left side"
                label="Left door"
                status={areaStatus.leftSide.status}
                available={areaStatus.leftSide.available}
                occupied={areaStatus.leftSide.occupied}
                total={areaStatus.leftSide.total}
              />

              <AreaCard
                title="Right side"
                label="Right side"
                status={areaStatus.rightSide.status}
                available={areaStatus.rightSide.available}
                occupied={areaStatus.rightSide.occupied}
                total={areaStatus.rightSide.total}
              />
            </div>
          </div>

          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2 style={{ marginTop: 0 }}>Book discussion room</h2>
            <div style={{ display: "grid", gap: 12 }}>
              <label style={{ display: "grid", gap: 8, fontWeight: 600 }}>
                Room
                <select value={bookingForm.roomId} onChange={(e) => setBookingForm({ ...bookingForm, roomId: e.target.value })} style={fieldStyle}>
                  <option value="DR-01">Focus Room 01</option>
                  <option value="DR-02">Focus Room 02</option>
                  <option value="DR-03">Focus Room 03</option>
                </select>
              </label>

              <label style={{ display: "grid", gap: 8, fontWeight: 600 }}>
                Date
                <input type="date" value={bookingForm.date} onChange={(e) => setBookingForm({ ...bookingForm, date: e.target.value })} style={fieldStyle} />
              </label>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <label style={{ display: "grid", gap: 8, fontWeight: 600 }}>
                  Start time
                  <input type="time" value={bookingForm.startTime} onChange={(e) => setBookingForm({ ...bookingForm, startTime: e.target.value })} style={fieldStyle} />
                </label>

                <label style={{ display: "grid", gap: 8, fontWeight: 600 }}>
                  End time
                  <input type="time" value={bookingForm.endTime} onChange={(e) => setBookingForm({ ...bookingForm, endTime: e.target.value })} style={fieldStyle} />
                </label>
              </div>

              <button onClick={bookRoom} style={{ background: "#d9703f", border: "none", color: "#fff", borderRadius: 12, padding: "12px 16px", cursor: "pointer", fontWeight: 700 }}>
                Confirm booking
              </button>
            </div>
          </div>
        </section>

        <section style={{ marginTop: 28 }}>
          <h2>Live library view</h2>
          <p style={{ color: "#5f5852" }}>Check the current camera view and chair occupancy before choosing a seat.</p>
          <LiveDetection />
        </section>
      </div>
    </main>
  );
}

function AreaCard({ title, label, status, available, occupied, total }: { title: string; label: string; status: "AVAILABLE" | "FULL"; available: number; occupied: number; total: number }) {
  const isAvailable = status === "AVAILABLE";
  const color = isAvailable ? "#1d8a4d" : "#c84d3d";
  const bg = isAvailable ? "#edf9f1" : "#fde9e7";

  return (
    <div style={{ background: bg, borderRadius: 16, padding: 16, border: `1px solid ${color}40` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <div>
          <strong>{title}</strong>
          <div style={{ fontSize: 12, color: "#5e5854" }}>{label}</div>
        </div>
        <span style={{ background: color, color: "#fff", padding: "6px 10px", borderRadius: 999, fontSize: 12, fontWeight: 700 }}>
          {isAvailable ? "Green flag" : "Red flag"}
        </span>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12, fontSize: 14, color: "#2e2d2b" }}>
        <span>{available} available</span>
        <span>{occupied} occupied</span>
      </div>

      <div style={{ height: 10, background: "rgba(0,0,0,0.08)", borderRadius: 999, overflow: "hidden", marginTop: 12 }}>
        <div style={{ width: `${(available / total) * 100}%`, background: color, height: "100%" }} />
      </div>

      <div style={{ marginTop: 12, color: color, fontWeight: 700 }}>
        {isAvailable ? "Seat available now" : "No seat available"}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ background: "#fff", borderRadius: 18, padding: 22, border: "1px solid #eadfce" }}>
      <div style={{ color: "#81756d", fontSize: 12, textTransform: "uppercase", letterSpacing: 1.2 }}>{label}</div>
      <div style={{ marginTop: 10, fontSize: 32, fontWeight: 800, color: "#1d1a18" }}>{value}</div>
    </div>
  );
}

const fieldStyle: React.CSSProperties = {
  border: "1px solid #e5d7cb",
  borderRadius: 10,
  padding: "10px 12px",
  fontSize: 15,
  background: "#fff",
};
