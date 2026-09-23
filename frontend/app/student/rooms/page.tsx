export default function StudentRooms() {
  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f7f1ea" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", borderRadius: 20, padding: 28, border: "1px solid #eadfce" }}>
        <h1>Room availability</h1>
        <p style={{ color: "#645d58" }}>The room directory is available through the student dashboard and the live backend API.</p>
        <a href="/student/dashboard" style={{ display: "inline-block", marginTop: 12, textDecoration: "none", color: "#fff", background: "#d9703f", padding: "10px 16px", borderRadius: 10 }}>Open dashboard</a>
      </div>
    </main>
  );
}
