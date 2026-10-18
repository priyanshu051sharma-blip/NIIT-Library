"use client";

import { useEffect, useState } from "react";
import LiveDetection from "../../components/LiveDetection";
import { useOccupancy } from "../../context/OccupancyContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function AdminDashboard() {
  const [bookings, setBookings] = useState<any[]>([]);
  const [bookingsError, setBookingsError] = useState<string | null>(null);
  const { snapshot, error: loadError } = useOccupancy();

  useEffect(() => {
    const role = localStorage.getItem("smartlib_role");
    const token = role === "ADMIN" || role === "LIBRARIAN" ? localStorage.getItem("smartlib_token") || "demo-admin" : "demo-admin";
    let active = true;
    let requestInProgress = false;
    const refreshBookings = async () => {
      if (requestInProgress) return;
      requestInProgress = true;
      try {
        const response = await fetch(`${API}/api/bookings`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Could not load shared room bookings.");
        const savedBookings = await response.json();
        if (active) {
          setBookings(savedBookings.filter((booking: any) => booking.status === "CONFIRMED").map((booking: any) => ({
            id: booking.id,
            roomId: booking.room_id,
            date: booking.booking_date,
            startTime: booking.start_time,
            endTime: booking.end_time,
            student: booking.student_id,
          })));
          setBookingsError(null);
        }
      } catch (requestError) {
        if (active) setBookingsError(requestError instanceof Error ? requestError.message : "Could not load shared room bookings.");
      } finally {
        requestInProgress = false;
      }
    };
    void refreshBookings();
    const timer = window.setInterval(() => void refreshBookings(), 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f5efe9" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24, gap: 16, flexWrap: "wrap" }}>
          <div>
            <p style={{ margin: 0, color: "#8e796d", letterSpacing: 1.2, textTransform: "uppercase", fontSize: 12 }}>Operations</p>
            <h1 style={{ margin: "8px 0 0", fontSize: 38 }}>Library operations</h1>
          </div>
          <button onClick={() => (window.location.href = "/")} style={{ background: "#1d1b19", color: "#fff", border: "none", padding: "12px 18px", borderRadius: 12, cursor: "pointer" }}>Return to portal</button>
        </header>

        {loadError && <p role="alert" style={{ color: "#a63125" }}>{loadError}</p>}
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
          <StatCard label="Library load" value={snapshot ? `${snapshot.occupancy_percentage}%` : "—"} />
          <StatCard label="Current people" value={snapshot?.people_count ?? "—"} />
          <StatCard label="Occupied seats" value={snapshot?.occupied_seats ?? "—"} />
          <StatCard label="Available seats" value={snapshot?.empty_seats ?? "—"} />
          <StatCard label="Total detected seats" value={snapshot?.total_seats ?? "—"} />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 18, marginTop: 18 }}>
          <GroupCard label="Discussion rooms · Cabin-1 and Cabin-2" group={snapshot?.camera_groups.discussion} />
          <GroupCard label="Entrance · Library Main and Library-3" group={snapshot?.camera_groups.entrance} peopleOnly />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24, marginTop: 28 }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2>Library seating load</h2>
            <p style={{ color: "#605c58" }}>Based on Library-1, Library-2 and Library-4 only.</p>
            <p style={{ color: "#605c58" }}>People detected and occupied seats are separate model measurements; they are not forced to match.</p>
            <p style={{ color: "#605c58" }}>Current status: <strong>{snapshot?.status ?? "WAITING"}</strong></p>
            <p style={{ color: "#605c58" }}>Detection source: <strong>{snapshot?.mode ?? "—"}</strong></p>
            <ul style={{ color: "#514d4a", lineHeight: 2 }}>
              <li>Occupied seats: {snapshot?.occupied_seats ?? "—"}</li>
              <li>Vacant / available seats: {snapshot?.empty_seats ?? "—"}</li>
              <li>Total detected seats: {snapshot?.total_seats ?? "—"}</li>
            </ul>
          </div>

          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2>Staff tools</h2>
            <div style={{ display: "grid", gap: 12 }}>
              <a href="/admin/live-monitor" style={{ display: "block", textDecoration: "none", background: "#faf3ed", borderRadius: 12, padding: 14, color: "#1d1b19" }}>Live monitor</a>
              <a href="/admin/analytics" style={{ display: "block", textDecoration: "none", background: "#faf3ed", borderRadius: 12, padding: 14, color: "#1d1b19" }}>Analytics</a>
              <a href="/admin/cameras" style={{ display: "block", textDecoration: "none", background: "#faf3ed", borderRadius: 12, padding: 14, color: "#1d1b19" }}>Camera feed</a>
            </div>

            <div style={{ marginTop: 22 }}>
              <h3 style={{ marginBottom: 10 }}>Booked discussion rooms</h3>
              {bookingsError && <p role="alert" style={{ color: "#a63125" }}>{bookingsError}</p>}
              {bookings.length ? (
                <div style={{ display: "grid", gap: 10 }}>
                  {bookings.map((booking) => (
                    <div key={booking.id} style={{ background: "#faf4ee", borderRadius: 12, padding: 12, border: "1px solid #efdccc" }}>
                      <strong>{booking.roomId}</strong>
                      <div style={{ color: "#655d58", fontSize: 13, marginTop: 6 }}>
                        {booking.student} • {booking.date} • {booking.startTime} to {booking.endTime}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: "#6c6662" }}>No room bookings yet.</div>
              )}
            </div>
          </div>
        </section>

        <section style={{ marginTop: 28 }}>
          <h2>Live library view</h2>
          <p style={{ color: "#5f5852" }}>Check the live camera feed and chair occupancy from the admin dashboard.</p>
          <LiveDetection />
        </section>
      </div>
    </main>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ background: "#fff", borderRadius: 18, padding: 22, border: "1px solid #eadfce" }}>
      <div style={{ fontSize: 12, color: "#82756d", textTransform: "uppercase", letterSpacing: 1.2 }}>{label}</div>
      <div style={{ fontSize: 32, fontWeight: 800, marginTop: 10 }}>{value}</div>
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
      <div style={{ fontSize: 12, color: "#82756d", textTransform: "uppercase", letterSpacing: 1.1 }}>{label}</div>
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
