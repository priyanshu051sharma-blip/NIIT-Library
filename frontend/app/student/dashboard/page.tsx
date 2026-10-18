"use client";

import { useEffect, useState } from "react";
import LiveDetection from "../../components/LiveDetection";
import { useOccupancy } from "../../context/OccupancyContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BookingForm = {
  roomId: string;
  date: string;
  startTime: string;
  endTime: string;
};

type DiscussionRoom = {
  id: string;
  name: string;
  capacity: number;
  location: string;
  status: string;
};

export default function StudentDashboard() {
  const [rooms, setRooms] = useState<DiscussionRoom[]>([]);
  const [roomError, setRoomError] = useState<string | null>(null);
  const [token, setToken] = useState<string>("");
  const [studentName, setStudentName] = useState<string>("Student");
  const { snapshot, error: loadError } = useOccupancy();
  const [bookingForm, setBookingForm] = useState<BookingForm>({
    roomId: "DR-01",
    date: "",
    startTime: "10:00",
    endTime: "11:00",
  });

  useEffect(() => {
    const today = new Date().toISOString().slice(0, 10);
    setBookingForm((prev) => ({ ...prev, date: prev.date || today }));

    const savedToken = localStorage.getItem("smartlib_token") || "demo-student";
    const savedStudent = localStorage.getItem("smartlib_student") || "ENR-2024-1048";
    setToken(savedToken);
    setStudentName(savedStudent);

    const headers = { Authorization: `Bearer ${savedToken}` };

    fetch(`${API}/api/rooms`, { headers })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not load room availability.");
        setRooms(await response.json());
        setRoomError(null);
      })
      .catch((requestError: unknown) => {
        setRoomError(requestError instanceof Error ? requestError.message : "Could not load room availability.");
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
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not create the room booking.");
      alert("Room booking confirmed and shared with the staff portal.");
    } catch (requestError) {
      alert(requestError instanceof Error ? requestError.message : "Could not create the room booking.");
    }
  };

  const availableSeats = snapshot?.empty_seats ?? 0;
  const seatStatus = !snapshot
    ? "Loading"
    : snapshot.mode === "waiting" || snapshot.total_seats === 0
      ? "No camera data"
      : availableSeats > 0 ? "Available" : "Full";

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

        {loadError && <p role="alert" style={{ color: "#a63125" }}>{loadError}</p>}
        {roomError && <p role="alert" style={{ color: "#a63125" }}>{roomError}</p>}
        {snapshot?.mode === "database-fallback" && (
          <p role="status" style={{ color: "#8a5b22" }}>
            Camera AI is unavailable; the counts below are database fallback values, not live detections.
          </p>
        )}
        {snapshot?.mode === "waiting" && (
          <p role="status" style={{ color: "#8a5b22" }}>Waiting for a connected camera to detect seats.</p>
        )}
        {snapshot && (
          <p style={{ color: "#81756d", fontSize: 13, marginBottom: 12 }}>
            Shared live snapshot · updated {new Date(snapshot.timestamp).toLocaleTimeString()} · {snapshot.camera_groups.library.detected_cameras}/{snapshot.camera_groups.library.connected_cameras} library cameras reporting
          </p>
        )}

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 18 }}>
          <StatCard label="Total detected seats" value={snapshot?.total_seats ?? "—"} />
          <StatCard label="People detected" value={snapshot?.people_count ?? "—"} />
          <StatCard label="Occupied seats" value={snapshot?.occupied_seats ?? "—"} />
          <StatCard label="Available seats" value={snapshot?.empty_seats ?? "—"} />
          <StatCard label="Occupancy" value={snapshot ? `${snapshot.occupancy_percentage}%` : "—"} />
          <StatCard label="Seat status" value={seatStatus} />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 18, marginTop: 18 }}>
          <GroupCard label="Discussion rooms · Cabin-1 and Cabin-2" group={snapshot?.camera_groups.discussion} />
          <GroupCard label="Entrance · Library Main and Library-3" group={snapshot?.camera_groups.entrance} peopleOnly />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24, marginTop: 28 }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2 style={{ marginTop: 0 }}>Live seat availability</h2>
            <p style={{ color: "#5f5852", marginTop: 0 }}>
              Occupied: <strong>{snapshot?.occupied_seats ?? "—"}</strong>
              {" · "}
              Vacant / available: <strong>{snapshot?.empty_seats ?? "—"}</strong>
            </p>
            <p style={{ color: "#5f5852" }}>
              Library load uses Library-1, Library-2 and Library-4 only. Cabin cameras are counted separately as discussion rooms; Library Main and Library-3 are entrance cameras.
            </p>
            <p style={{ color: "#5f5852" }}>
              People detected and occupied seats are separate measurements. Occupied seats are supported by seat detections and people inside the calibrated seating area.
            </p>
          </div>

          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2 style={{ marginTop: 0 }}>Book discussion room</h2>
            <div style={{ display: "grid", gap: 12 }}>
              <label style={{ display: "grid", gap: 8, fontWeight: 600 }}>
                Room
                <select value={bookingForm.roomId} onChange={(e) => setBookingForm({ ...bookingForm, roomId: e.target.value })} style={fieldStyle}>
                  {rooms.map((room) => (
                    <option key={room.id} value={room.id} disabled={room.status !== "AVAILABLE"}>
                      {room.name} · {room.capacity} seats · {room.status}
                    </option>
                  ))}
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

              <button onClick={bookRoom} disabled={!rooms.length} style={{ background: "#d9703f", border: "none", color: "#fff", borderRadius: 12, padding: "12px 16px", cursor: rooms.length ? "pointer" : "not-allowed", fontWeight: 700, opacity: rooms.length ? 1 : 0.6 }}>
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

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ background: "#fff", borderRadius: 18, padding: 22, border: "1px solid #eadfce" }}>
      <div style={{ color: "#81756d", fontSize: 12, textTransform: "uppercase", letterSpacing: 1.2 }}>{label}</div>
      <div style={{ marginTop: 10, fontSize: 32, fontWeight: 800, color: "#1d1a18" }}>{value}</div>
    </div>
  );
}

function GroupCard({
  label,
  group,
  peopleOnly = false,
}: {
  label: string;
  group?: { people_count: number; occupied_seats: number; empty_seats: number; detected_cameras: number };
  peopleOnly?: boolean;
}) {
  return (
    <div style={{ background: "#fff", borderRadius: 18, padding: 22, border: "1px solid #eadfce" }}>
      <div style={{ color: "#81756d", fontSize: 12, textTransform: "uppercase", letterSpacing: 1.1 }}>{label}</div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 14, color: "#514d4a" }}>
        <span>People <strong>{group?.people_count ?? "—"}</strong></span>
        {!peopleOnly && <>
          <span>Occupied <strong>{group?.occupied_seats ?? "—"}</strong></span>
          <span>Available <strong>{group?.empty_seats ?? "—"}</strong></span>
        </>}
      </div>
      <div style={{ color: "#81756d", fontSize: 12, marginTop: 12 }}>
        {group ? `${group.detected_cameras} camera feeds reporting` : "Waiting for camera data"}
      </div>
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
