export default function StudentBookings() {
  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f7f1ea" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", borderRadius: 20, padding: 28, border: "1px solid #eadfce" }}>
        <h1>My bookings</h1>
        <p style={{ color: "#645d58" }}>Booking data is served by the backend API and is available after the student signs in.</p>
        <a href="/student/dashboard" style={{ display: "inline-block", marginTop: 12, textDecoration: "none", color: "#fff", background: "#d9703f", padding: "10px 16px", borderRadius: 10 }}>Back to dashboard</a>
      </div>
    </main>
  );
}
