"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOTAL_SEATS = 70;

export default function AdminDashboard() {
  const [snapshot, setSnapshot] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);
  const [areaStatus, setAreaStatus] = useState({
    leftSide: { total: 35, available: 24, occupied: 11, status: "AVAILABLE" },
    rightSide: { total: 35, available: 18, occupied: 17, status: "AVAILABLE" },
  });
  const [bookings, setBookings] = useState<any[]>([]);
  const hourlyLogs = [
    { hour: "08:00", occupied: 15, available: 55, capacity: TOTAL_SEATS, occupancy: 21 },
    { hour: "09:00", occupied: 18, available: 52, capacity: TOTAL_SEATS, occupancy: 26 },
    { hour: "10:00", occupied: 24, available: 46, capacity: TOTAL_SEATS, occupancy: 34 },
    { hour: "11:00", occupied: 26, available: 44, capacity: TOTAL_SEATS, occupancy: 37 },
    { hour: "12:00", occupied: 30, available: 40, capacity: TOTAL_SEATS, occupancy: 43 },
    { hour: "13:00", occupied: 28, available: 42, capacity: TOTAL_SEATS, occupancy: 40 },
    { hour: "14:00", occupied: 25, available: 45, capacity: TOTAL_SEATS, occupancy: 36 },
    { hour: "15:00", occupied: 22, available: 48, capacity: TOTAL_SEATS, occupancy: 31 },
  ];

  useEffect(() => {
    const role = localStorage.getItem("smartlib_role");
    const token = role === "ADMIN" || role === "LIBRARIAN" ? localStorage.getItem("smartlib_token") || "demo-admin" : "demo-admin";
    const storedArea = localStorage.getItem("smartlib_area_status");
    const storedBookings = localStorage.getItem("smartlib_bookings");

    if (storedArea) {
      setAreaStatus(JSON.parse(storedArea));
    }
    if (storedBookings) {
      setBookings(JSON.parse(storedBookings));
    }

    const headers = { Authorization: `Bearer ${token}` };

    Promise.all([
      fetch(`${API}/api/occupancy/current`, { headers }),
      fetch(`${API}/api/admin/analytics`, { headers }),
      fetch(`${API}/api/cameras`, { headers }),
    ])
      .then(async ([snapshotRes, analyticsRes, camerasRes]) => {
        setSnapshot(await snapshotRes.json());
        setAnalytics(await analyticsRes.json());
        await camerasRes.json();
      })
      .catch(() => {
        setSnapshot({ people_count: 42, total_seats: TOTAL_SEATS, occupied_seats: 42, empty_seats: 28, occupancy_percentage: 60, status: "MODERATE" });
        setAnalytics({ peak_occupancy: 72, average_occupancy: 55.6, hourly: [] });
      });
  }, []);

  const leftStatus = areaStatus.leftSide.available > 0 ? "Green flag" : "Red flag";
  const rightStatus = areaStatus.rightSide.available > 0 ? "Green flag" : "Red flag";

  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f5efe9" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
          <div>
            <p style={{ margin: 0, color: "#8e796d", letterSpacing: 1.2, textTransform: "uppercase", fontSize: 12 }}>Operations</p>
            <h1 style={{ margin: "8px 0 0", fontSize: 38 }}>Library operations</h1>
          </div>
          <button onClick={() => (window.location.href = "/")} style={{ background: "#1d1b19", color: "#fff", border: "none", padding: "12px 18px", borderRadius: 12, cursor: "pointer" }}>Return to portal</button>
        </header>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 18 }}>
          <StatCard label="Library load" value={`${snapshot?.occupancy_percentage ?? 60}%`} />
          <StatCard label="Current people" value={snapshot?.people_count ?? 42} />
          <StatCard label="Occupied" value={snapshot?.occupied_seats ?? 42} />
          <StatCard label="Total seats" value={TOTAL_SEATS} />
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 28 }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 24, border: "1px solid #eadfce" }}>
            <h2>Live occupancy summary</h2>
            <p style={{ color: "#605c58" }}>Current status: <strong>{snapshot?.status ?? "MODERATE"}</strong></p>
            <ul style={{ color: "#514d4a", lineHeight: 2 }}>
              <li>Occupied seats: {snapshot?.occupied_seats ?? 42}</li>
              <li>Available seats: {snapshot?.empty_seats ?? 28}</li>
              <li>Total seats: {snapshot?.total_seats ?? TOTAL_SEATS}</li>
            </ul>

            <div style={{ display: "grid", gap: 12, marginTop: 18 }}>
              <AreaFlag title="Left door" status={leftStatus} available={areaStatus.leftSide.available} total={areaStatus.leftSide.total} />
              <AreaFlag title="Right side" status={rightStatus} available={areaStatus.rightSide.available} total={areaStatus.rightSide.total} />
            </div>

            <div style={{ marginTop: 24 }}>
              <h3 style={{ marginBottom: 10 }}>Hourly capacity log</h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: "left", color: "#6c6662" }}>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #eae0d6" }}>Hour</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #eae0d6" }}>Used</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #eae0d6" }}>Available</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #eae0d6" }}>Occupancy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hourlyLogs.map((log) => (
                      <tr key={log.hour}>
                        <td style={{ padding: "8px 10px", borderBottom: "1px solid #f0e5dc" }}>{log.hour}</td>
                        <td style={{ padding: "8px 10px", borderBottom: "1px solid #f0e5dc" }}>{log.occupied}</td>
                        <td style={{ padding: "8px 10px", borderBottom: "1px solid #f0e5dc" }}>{log.available}</td>
                        <td style={{ padding: "8px 10px", borderBottom: "1px solid #f0e5dc" }}>{log.occupancy}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
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
      </div>
    </main>
  );
}

function AreaFlag({ title, status, available, total }: { title: string; status: string; available: number; total: number }) {
  const isGreen = status === "Green flag";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 12px", borderRadius: 12, background: isGreen ? "#edf9f1" : "#fde9e7", border: `1px solid ${isGreen ? "#c9ecd4" : "#f4c1bb"}` }}>
      <span>{title}</span>
      <span style={{ fontWeight: 700, color: isGreen ? "#1d8a4d" : "#c84d3d" }}>{status}</span>
      <span>{available} / {total} seats</span>
    </div>
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
