from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any
import json
import math
import os
import secrets
from urllib import request
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from .database import Booking, Camera, OccupancyLog, Room, Seat, SessionLocal, User, Zone

app = FastAPI(title="SmartLib API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
bearer = HTTPBearer(auto_error=False)

class Role(str, Enum):
    STUDENT = "STUDENT"
    LIBRARIAN = "LIBRARIAN"
    ADMIN = "ADMIN"

class BookingRequest(BaseModel):
    room_id: str = Field(min_length=1, max_length=32)
    booking_date: date
    start_time: time
    end_time: time

    @field_validator("end_time")
    @classmethod
    def valid_window(cls, value: time, info):
        start = info.data.get("start_time")
        if start and value <= start:
            raise ValueError("end_time must be after start_time")
        return value

class ConnectionManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]):
        for connection in list(self.connections):
            try:
                await connection.send_json(payload)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")
TOTAL_SEATS = int(os.getenv("TOTAL_SEATS", "70"))
BOOKINGS: list[dict[str, Any]] = []


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_library_data(db):
    if db.query(User).filter(User.id == "demo-student").first() is None:
        db.add(User(id="demo-student", name="Demo Student", email="demo.student@smartlib.local", password_hash="demo-only", role="STUDENT", student_id="ENR-2024-1048"))
        db.commit()
    if db.query(Zone).count() > 0:
        return
    left = Zone(id="left-zone", name="Left side", capacity=35)
    right = Zone(id="right-zone", name="Right side", capacity=35)
    db.add_all([left, right])
    db.commit()

    for index in range(1, 36):
        db.add(Seat(id=f"L-{index:02d}", seat_number=f"L-{index:02d}", zone_id="left-zone", status="empty" if index % 4 else "occupied", coordinates=[]))
    for index in range(1, 36):
        db.add(Seat(id=f"R-{index:02d}", seat_number=f"R-{index:02d}", zone_id="right-zone", status="empty" if index % 3 else "occupied", coordinates=[]))

    db.add_all([
        Camera(id="cam-main", name="Main reading floor", location="North wing", rtsp_url=None, zone_id="left-zone", status="SIMULATED"),
        Camera(id="cam-right", name="Right reading floor", location="South wing", rtsp_url=None, zone_id="right-zone", status="SIMULATED"),
        Room(id="DR-01", name="Focus Room 01", capacity=6, location="North wing", status="AVAILABLE"),
        Room(id="DR-02", name="Focus Room 02", capacity=8, location="North wing", status="OCCUPIED"),
        Room(id="DR-03", name="Focus Room 03", capacity=4, location="South wing", status="AVAILABLE"),
    ])
    db.commit()


with SessionLocal() as db:
    seed_library_data(db)


def load_seat_data(db=None):
    owns_db = db is None
    db = db or SessionLocal()
    try:
        seats = db.query(Seat).order_by(Seat.seat_number).all()
        return [{
            "id": seat.id,
            "seat_number": seat.seat_number,
            "zone_id": seat.zone_id,
            "status": seat.status,
            "coordinates": seat.coordinates or [],
        } for seat in seats]
    finally:
        if owns_db:
            db.close()


SEATS = load_seat_data()
BOOKINGS = []
with SessionLocal() as db:
    existing = db.query(Booking).order_by(Booking.created_at).all()
    for booking in existing:
        BOOKINGS.append({
            "id": booking.id,
            "room_id": booking.room_id,
            "booking_date": booking.booking_date,
            "start_time": booking.start_time,
            "end_time": booking.end_time,
            "student_id": booking.student_id,
            "status": booking.status,
            "created_at": booking.created_at,
        })
ROOMS = [
    {"id": "DR-01", "name": "Focus Room 01", "capacity": 6, "location": "North wing", "status": "AVAILABLE"},
    {"id": "DR-02", "name": "Focus Room 02", "capacity": 8, "location": "North wing", "status": "OCCUPIED"},
    {"id": "DR-03", "name": "Focus Room 03", "capacity": 4, "location": "South wing", "status": "AVAILABLE"},
]


def read_ai_snapshot() -> dict[str, Any] | None:
    try:
        with request.urlopen(f"{AI_SERVICE_URL}/ai/occupancy", timeout=2) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception:
        return None


def current_snapshot() -> dict[str, Any]:
    occupied = sum(seat["status"] == "occupied" for seat in SEATS)
    total = max(len(SEATS), TOTAL_SEATS)
    people = occupied
    ai_snapshot = read_ai_snapshot()
    if ai_snapshot:
        people = int(ai_snapshot.get("people_count", people))
        occupied = int(ai_snapshot.get("occupied_seats", occupied))
        total = max(TOTAL_SEATS, int(ai_snapshot.get("empty_seats", total - occupied)) + occupied)
    empty = max(total - occupied, 0)
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "people_count": people,
        "total_seats": total,
        "occupied_seats": occupied,
        "empty_seats": empty,
        "occupancy_percentage": round((occupied / total) * 100, 1) if total else 0,
        "entries": int(ai_snapshot.get("entries", 86)) if ai_snapshot else 86,
        "exits": int(ai_snapshot.get("exits", 61)) if ai_snapshot else 61,
        "status": occupancy_status(occupied / total if total else 0),
        "zones": zone_snapshots(),
        "privacy": "No facial recognition or identity data is processed.",
    }

def occupancy_status(ratio: float) -> str:
    if ratio < 0.4:
        return "LOW"
    if ratio < 0.75:
        return "MODERATE"
    if ratio < 0.9:
        return "BUSY"
    return "VERY BUSY"

def zone_snapshots():
    zone_map = {
        "left-zone": "Left side",
        "right-zone": "Right side",
    }
    result = []
    for zone_id, label in zone_map.items():
        seats = [seat for seat in SEATS if seat["zone_id"] == zone_id]
        used = sum(seat["status"] == "occupied" for seat in seats)
        result.append({"id": zone_id, "name": label, "capacity": len(seats), "occupied": used, "available": len(seats) - used, "percentage": round(used / len(seats) * 100, 1) if seats else 0})
    if not result:
        result = [{"id": "left-zone", "name": "Left side", "capacity": 35, "occupied": 14, "available": 21, "percentage": 40}, {"id": "right-zone", "name": "Right side", "capacity": 35, "occupied": 16, "available": 19, "percentage": 46}]
    return result

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    # Demo token keeps the local app usable; replace with JWT verification in deployment.
    if credentials and credentials.credentials in {"demo-student", "demo-librarian", "demo-admin"}:
        if credentials.credentials == "demo-admin":
            return Role.ADMIN
        if credentials.credentials == "demo-librarian":
            return Role.LIBRARIAN
        return Role.STUDENT
    return Role.STUDENT

def require_staff(user: Role = Depends(current_user)):
    if user not in {Role.ADMIN, Role.LIBRARIAN}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Staff access required")
    return user

@app.get("/health")
def health():
    return {"status": "ok", "service": "smartlib-backend", "mode": os.getenv("APP_MODE", "mock")}

@app.post("/api/process/frame")
async def process_browser_frame(frame_data: str = Form(...), _: Role = Depends(current_user)):
    payload = urlencode({"frame_data": frame_data}).encode()
    vision_request = request.Request(
        f"{AI_SERVICE_URL}/ai/process-frame/base64",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with request.urlopen(vision_request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"AI frame processing unavailable: {error}") from error
    return {"success": True, "occupancy": result, "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.api_route("/api/auth/demo", methods=["GET", "POST"])
async def demo_login(request: Request, role: str | None = None):
    payload = None
    if request.method == "POST":
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    if payload:
        role = str(payload.get("role") or payload.get("role_name") or role or Role.STUDENT)
    elif role is None:
        role = str(request.query_params.get("role") or Role.STUDENT)
    try:
        parsed_role = Role(role.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")
    tokens = {Role.STUDENT: "demo-student", Role.LIBRARIAN: "demo-librarian", Role.ADMIN: "demo-admin"}
    return {"access_token": tokens[parsed_role], "token_type": "bearer", "role": parsed_role}

@app.get("/api/occupancy/current")
def occupancy_current(_: Role = Depends(current_user)):
    return current_snapshot()

@app.get("/api/occupancy/hourly")
def occupancy_hourly(_: Role = Depends(current_user)):
    return [
        {"hour": "08:00", "people_count": 15, "occupied_seats": 15, "entries": 6, "exits": 3},
        {"hour": "09:00", "people_count": 18, "occupied_seats": 18, "entries": 8, "exits": 4},
        {"hour": "10:00", "people_count": 24, "occupied_seats": 24, "entries": 10, "exits": 5},
        {"hour": "11:00", "people_count": 26, "occupied_seats": 26, "entries": 11, "exits": 6},
        {"hour": "12:00", "people_count": 30, "occupied_seats": 30, "entries": 12, "exits": 7},
        {"hour": "13:00", "people_count": 28, "occupied_seats": 28, "entries": 8, "exits": 5},
        {"hour": "14:00", "people_count": 25, "occupied_seats": 25, "entries": 9, "exits": 4},
        {"hour": "15:00", "people_count": 22, "occupied_seats": 22, "entries": 7, "exits": 4},
    ]

@app.get("/api/zones")
def zones(_: Role = Depends(current_user)):
    return zone_snapshots()

@app.get("/api/seats")
def seats(_: Role = Depends(current_user)):
    return SEATS

@app.get("/api/rooms")
def rooms(_: Role = Depends(current_user)):
    return ROOMS

@app.post("/api/bookings", status_code=201)
async def create_booking(request: BookingRequest, user: Role = Depends(current_user)):
    if user != Role.STUDENT:
        raise HTTPException(status_code=403, detail="Only students can create bookings")
    with SessionLocal() as db:
        room = db.query(Room).filter(Room.id == request.room_id).first()
        if room is None:
            raise HTTPException(status_code=404, detail="Room not found")

        for booking in db.query(Booking).filter(Booking.room_id == request.room_id, Booking.booking_date == request.booking_date.isoformat(), Booking.status == "CONFIRMED").all():
            if request.start_time < time.fromisoformat(booking.end_time) and request.end_time > time.fromisoformat(booking.start_time):
                raise HTTPException(status_code=409, detail="This slot overlaps an existing booking")

        booking = Booking(
            id=secrets.token_hex(6),
            room_id=request.room_id,
            student_id="demo-student",
            booking_date=request.booking_date.isoformat(),
            start_time=request.start_time.isoformat(timespec="minutes"),
            end_time=request.end_time.isoformat(timespec="minutes"),
            status="CONFIRMED",
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        db.add(booking)
        db.commit()
        booking_payload = {
            "id": booking.id,
            "room_id": booking.room_id,
            "booking_date": booking.booking_date,
            "start_time": booking.start_time,
            "end_time": booking.end_time,
            "student_id": booking.student_id,
            "status": booking.status,
            "created_at": booking.created_at,
        }
        BOOKINGS.append(booking_payload)
        await manager.broadcast({"type": "booking.created", "booking": booking_payload})
        return booking_payload

@app.get("/api/bookings")
def bookings(_: Role = Depends(current_user)):
    return BOOKINGS

@app.delete("/api/bookings/{booking_id}")
def cancel_booking(booking_id: str, _: Role = Depends(current_user)):
    with SessionLocal() as db:
        booking = db.query(Booking).filter(Booking.id == booking_id).first()
        if booking:
            booking.status = "CANCELLED"
            db.commit()
            for item in BOOKINGS:
                if item["id"] == booking_id:
                    item["status"] = "CANCELLED"
                    return item
            return {"id": booking.id, "room_id": booking.room_id, "booking_date": booking.booking_date, "start_time": booking.start_time, "end_time": booking.end_time, "student_id": booking.student_id, "status": booking.status, "created_at": booking.created_at}
    raise HTTPException(status_code=404, detail="Booking not found")

@app.get("/api/cameras")
def cameras(_: Role = Depends(require_staff)):
    return [{"id": "cam-main", "name": "Main reading floor", "location": "North wing", "status": "SIMULATED", "rtsp_url": None}]

@app.get("/api/admin/analytics")
def analytics(_: Role = Depends(require_staff)):
    hourly = occupancy_hourly(Role.ADMIN)
    return {
        "peak_occupancy": max(point["people_count"] for point in hourly),
        "average_occupancy": round(sum(point["people_count"] for point in hourly) / len(hourly), 1),
        "hourly": hourly,
        "zones": zone_snapshots(),
        "total_seats": TOTAL_SEATS,
    }

@app.websocket("/ws/occupancy")
async def occupancy_socket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json(current_snapshot())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
