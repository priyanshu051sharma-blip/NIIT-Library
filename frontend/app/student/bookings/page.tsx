"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Booking = {
  id: string;
  room_id: string;
  booking_date: string;
  start_time: string;
  end_time: string;
  status: string;
};

export default function StudentBookings() {
  const [bookings, setBookings] = useState<Booking[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("smartlib_token") || "demo-student";
    fetch(`${API}/api/bookings`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not load bookings from the server.");
        const serverBookings = await response.json();
        setBookings(serverBookings.filter((booking: Booking) => booking.status === "CONFIRMED"));
        setError(null);
      })
      .catch((requestError: unknown) => {
        setError(requestError instanceof Error ? requestError.message : "Could not load bookings from the server.");
      });
  }, []);

  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f7f1ea" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", borderRadius: 20, padding: 28, border: "1px solid #eadfce" }}>
        <p style={{ margin: 0, color: "#8b7769", letterSpacing: 1.2, textTransform: "uppercase", fontSize: 12 }}>SmartLib</p>
        <h1>My bookings</h1>
        <p style={{ color: "#645d58" }}>Bookings sync from the server and are shared with library staff.</p>
        {error && <p role="alert" style={{ color: "#a63125" }}>{error}</p>}
        {bookings.length ? (
          <div style={{ display: "grid", gap: 12 }}>
            {bookings.map((booking) => (
              <article key={booking.id} style={{ background: "#faf5ef", border: "1px solid #eadfce", borderRadius: 12, padding: 16 }}>
                <strong>{booking.room_id}</strong>
                <div style={{ color: "#645d58", marginTop: 6 }}>
                  {booking.booking_date} · {booking.start_time}–{booking.end_time} · {booking.status}
                </div>
              </article>
            ))}
          </div>
        ) : !error ? (
          <p style={{ color: "#645d58" }}>No shared bookings yet.</p>
        ) : null}
        <a href="/student/dashboard" style={{ display: "inline-block", marginTop: 20, textDecoration: "none", color: "#fff", background: "#d9703f", padding: "10px 16px", borderRadius: 10 }}>Back to dashboard</a>
      </div>
    </main>
  );
}
