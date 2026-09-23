export default function AdminBookings() {
  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f5efe9" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", borderRadius: 20, padding: 28, border: "1px solid #eadfce" }}>
        <h1>Bookings overview</h1>
        <p style={{ color: "#645d58" }}>Room bookings are tracked via the SmartLib backend and available to staff after login.</p>
        <a href="/admin/dashboard" style={{ display: "inline-block", marginTop: 12, textDecoration: "none", color: "#fff", background: "#d9703f", padding: "10px 16px", borderRadius: 10 }}>Back to dashboard</a>
      </div>
    </main>
  );
}
