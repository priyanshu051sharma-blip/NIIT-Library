export default function AdminUsers() {
  return (
    <main style={{ padding: 32, minHeight: "100vh", background: "#f5efe9" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", borderRadius: 20, padding: 28, border: "1px solid #eadfce" }}>
        <h1>Users</h1>
        <p style={{ color: "#645d58" }}>This view can be connected to the authentication and user management database once the production service is enabled.</p>
        <a href="/admin/dashboard" style={{ display: "inline-block", marginTop: 12, textDecoration: "none", color: "#fff", background: "#d9703f", padding: "10px 16px", borderRadius: 10 }}>Back to dashboard</a>
      </div>
    </main>
  );
}
