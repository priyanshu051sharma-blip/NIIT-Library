import LiveDetection from "../../components/LiveDetection";

export default function AdminLiveMonitor() {
  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f5efe9" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <p style={{ margin: 0, color: "#9a5a36", fontWeight: 700, letterSpacing: 1 }}>VISION / CHAIR OCCUPANCY</p>
        <h1 style={{ marginBottom: 8 }}>Live monitor</h1>
        <p style={{ color: "#645d58", marginTop: 0 }}>Upload a camera frame to see Empty Chair, Guest on Chair, and Occupied Chair detections.</p>
        <LiveDetection />

        <a href="/admin/dashboard" style={{ display: "inline-block", marginTop: 16, textDecoration: "none", color: "#645d58" }}>Back to dashboard</a>
      </div>
    </main>
  );
}
