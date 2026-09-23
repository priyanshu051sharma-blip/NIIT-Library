import os
import secrets
from typing import Any

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, Time, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartlib.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="STUDENT")
    student_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[str] = mapped_column(String(50), default=lambda: __import__("datetime").datetime.utcnow().isoformat() + "Z")


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    location: Mapped[str] = mapped_column(String(160), nullable=False)
    rtsp_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("zones.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="OFFLINE")


class Seat(Base):
    __tablename__ = "seats"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    seat_number: Mapped[str] = mapped_column(String(40), nullable=False)
    zone_id: Mapped[str] = mapped_column(String(64), ForeignKey("zones.id"), nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("cameras.id"), nullable=True)
    coordinates: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=lambda: [])
    status: Mapped[str] = mapped_column(String(20), default="empty")
    last_updated: Mapped[str] = mapped_column(String(50), default=lambda: __import__("datetime").datetime.utcnow().isoformat() + "Z")


class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(50), default=lambda: __import__("datetime").datetime.utcnow().isoformat() + "Z")
    camera_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("cameras.id"), nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("zones.id"), nullable=True)
    people_count: Mapped[int] = mapped_column(Integer, default=0)
    occupied_seats: Mapped[int] = mapped_column(Integer, default=0)
    empty_seats: Mapped[int] = mapped_column(Integer, default=0)
    entries: Mapped[int] = mapped_column(Integer, default=0)
    exits: Mapped[int] = mapped_column(Integer, default=0)


class Room(Base):
    __tablename__ = "discussion_rooms"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    location: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="AVAILABLE")


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: secrets.token_urlsafe(12))
    room_id: Mapped[str] = mapped_column(String(64), ForeignKey("discussion_rooms.id"), nullable=False)
    student_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False)
    booking_date: Mapped[str] = mapped_column(String(20), nullable=False)
    start_time: Mapped[str] = mapped_column(String(20), nullable=False)
    end_time: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="CONFIRMED")
    created_at: Mapped[str] = mapped_column(String(50), default=lambda: __import__("datetime").datetime.utcnow().isoformat() + "Z")


Base.metadata.create_all(bind=engine)
