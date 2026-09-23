"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowRight, BookOpen, ShieldCheck, UserRound } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const router = useRouter();
  const [busy, setBusy] = useState<"STUDENT" | "LIBRARIAN" | "ADMIN" | null>(null);
  const [studentForm, setStudentForm] = useState({ enrollment: "", password: "" });

  const login = async (role: "STUDENT" | "LIBRARIAN" | "ADMIN") => {
    setBusy(role);
    try {
      const response = await fetch(`${API}/api/auth/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Login failed");
      localStorage.setItem("smartlib_token", data.access_token);
      localStorage.setItem("smartlib_role", data.role);
      router.push(role === "STUDENT" ? "/student/dashboard" : "/admin/dashboard");
    } catch (error) {
      console.error(error);
      alert("Unable to sign in right now. Check that the backend is running on localhost:8000.");
    } finally {
      setBusy(null);
    }
  };

  const studentLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    const enrollment = studentForm.enrollment.trim();
    const password = studentForm.password.trim();

    if (!enrollment || !password) {
      alert("Please enter your enrollment number and password.");
      return;
    }

    localStorage.setItem("smartlib_student", enrollment);
    localStorage.setItem("smartlib_token", "demo-student");
    localStorage.setItem("smartlib_role", "STUDENT");

    const defaultSeats = {
      leftSide: { total: 35, available: 24, occupied: 11, status: "AVAILABLE" },
      rightSide: { total: 35, available: 18, occupied: 17, status: "AVAILABLE" },
      lastUpdated: new Date().toISOString(),
    };
    localStorage.setItem("smartlib_area_status", JSON.stringify(defaultSeats));

    router.push("/student/dashboard");
  };

  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", background: "#f4efe9", padding: 24 }}>
      <div style={{ maxWidth: 980, width: "100%", background: "#fffaf4", border: "1px solid #e9dfd3", borderRadius: 24, boxShadow: "0 24px 60px rgba(41,28,20,0.08)", overflow: "hidden" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 0 }}>
          <section style={{ padding: 48, background: "linear-gradient(135deg, #f8f1ea 0%, #fff 100%)" }}>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 10, padding: "8px 12px", borderRadius: 999, background: "#f5e5d9", color: "#9a4f2d", fontWeight: 700, fontSize: 12 }}>
              <BookOpen size={15} /> SmartLib
            </div>
            <h1 style={{ margin: "24px 0 12px", fontSize: 52, lineHeight: 1.05, letterSpacing: -2, color: "#1a1714" }}>AI-powered library occupancy.</h1>
            <p style={{ margin: 0, maxWidth: 500, color: "#5f5852", fontSize: 17, lineHeight: 1.7 }}>
              Track real-time library occupancy, book discussion rooms, and check seat availability across the left and right library zones.
            </p>
            <div style={{ marginTop: 28, display: "flex", gap: 16, flexWrap: "wrap" }}>
              <div style={{ background: "#fff", border: "1px solid #eee3d7", borderRadius: 16, padding: "14px 18px", minWidth: 150 }}>
                <strong style={{ display: "block", fontSize: 28, color: "#1a1714" }}>70</strong>
                <span style={{ color: "#706b65", fontSize: 13 }}>total library seats</span>
              </div>
              <div style={{ background: "#fff", border: "1px solid #eee3d7", borderRadius: 16, padding: "14px 18px", minWidth: 150 }}>
                <strong style={{ display: "block", fontSize: 28, color: "#1a1714" }}>2</strong>
                <span style={{ color: "#706b65", fontSize: 13 }}>area zones</span>
              </div>
            </div>
          </section>

          <aside style={{ padding: 40, background: "#fffdfb", borderLeft: "1px solid #efe5db" }}>
            <div style={{ marginBottom: 18 }}>
              <p style={{ margin: 0, color: "#9b8575", fontSize: 12, letterSpacing: 1.5, textTransform: "uppercase" }}>Portal access</p>
              <h2 style={{ margin: "10px 0 0", color: "#1f1b18", fontSize: 30 }}>Student login</h2>
            </div>

            <form onSubmit={studentLogin} style={{ display: "grid", gap: 14, marginBottom: 22 }}>
              <label style={{ display: "grid", gap: 8, color: "#3a352f", fontWeight: 600 }}>
                Enrollment number
                <input
                  type="text"
                  value={studentForm.enrollment}
                  onChange={(event) => setStudentForm({ ...studentForm, enrollment: event.target.value })}
                  placeholder="e.g. ENR-2024-1048"
                  style={{ border: "1px solid #e4d7cc", borderRadius: 12, padding: "12px 14px", fontSize: 15 }}
                />
              </label>

              <label style={{ display: "grid", gap: 8, color: "#3a352f", fontWeight: 600 }}>
                Password
                <input
                  type="password"
                  value={studentForm.password}
                  onChange={(event) => setStudentForm({ ...studentForm, password: event.target.value })}
                  placeholder="Enter your password"
                  style={{ border: "1px solid #e4d7cc", borderRadius: 12, padding: "12px 14px", fontSize: 15 }}
                />
              </label>

              <button type="submit" disabled={busy !== null} style={{ ...buttonStyle, justifyContent: "center", background: "#d9703f", color: "#fff" }}>
                {busy === "STUDENT" ? "Signing in..." : "Login as student"}
              </button>
            </form>

            <div style={{ display: "grid", gap: 12 }}>
              <button onClick={() => login("LIBRARIAN")} disabled={busy !== null} style={buttonStyle}>
                <span style={{ display: "flex", alignItems: "center", gap: 12 }}><ShieldCheck size={18} /> Librarian portal</span>
                {busy === "LIBRARIAN" ? "Signing in..." : <ArrowRight size={18} />}
              </button>

              <button onClick={() => login("ADMIN")} disabled={busy !== null} style={{ ...buttonStyle, background: "#1d1b19", color: "#fff" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 12 }}><ShieldCheck size={18} /> Admin portal</span>
                {busy === "ADMIN" ? "Signing in..." : <ArrowRight size={18} />}
              </button>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}

const buttonStyle: React.CSSProperties = {
  border: "1px solid #e8ddd2",
  background: "#fff",
  borderRadius: 14,
  padding: "18px 18px",
  fontSize: 16,
  color: "#1e1b18",
  fontWeight: 600,
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  cursor: "pointer",
  transition: "all 0.2s ease",
};
