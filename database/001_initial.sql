CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE TYPE user_role AS ENUM ('STUDENT', 'LIBRARIAN', 'ADMIN');
CREATE TYPE occupancy_status AS ENUM ('empty', 'occupied');

CREATE TABLE users (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role user_role NOT NULL DEFAULT 'STUDENT', student_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE zones (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, capacity INTEGER NOT NULL CHECK (capacity >= 0));
CREATE TABLE cameras (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, location TEXT NOT NULL, rtsp_url TEXT NOT NULL, zone_id UUID REFERENCES zones(id), status TEXT NOT NULL DEFAULT 'offline');
CREATE TABLE seats (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), seat_number TEXT NOT NULL, zone_id UUID NOT NULL REFERENCES zones(id), camera_id UUID REFERENCES cameras(id), coordinates JSONB NOT NULL DEFAULT '[]', status occupancy_status NOT NULL DEFAULT 'empty', last_updated TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE occupancy_logs (id BIGSERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL DEFAULT now(), camera_id UUID REFERENCES cameras(id), zone_id UUID REFERENCES zones(id), people_count INTEGER NOT NULL, occupied_seats INTEGER NOT NULL, empty_seats INTEGER NOT NULL, entries INTEGER NOT NULL DEFAULT 0, exits INTEGER NOT NULL DEFAULT 0);
CREATE TABLE discussion_rooms (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, capacity INTEGER NOT NULL, location TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'AVAILABLE');
CREATE TABLE bookings (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), room_id UUID NOT NULL REFERENCES discussion_rooms(id), student_id UUID NOT NULL REFERENCES users(id), booking_date DATE NOT NULL, start_time TIME NOT NULL, end_time TIME NOT NULL, status TEXT NOT NULL DEFAULT 'CONFIRMED', created_at TIMESTAMPTZ NOT NULL DEFAULT now(), CHECK (end_time > start_time));
CREATE TABLE audit_logs (id BIGSERIAL PRIMARY KEY, user_id UUID REFERENCES users(id), action TEXT NOT NULL, resource TEXT NOT NULL, timestamp TIMESTAMPTZ NOT NULL DEFAULT now(), ip_address INET);
CREATE INDEX bookings_room_date_idx ON bookings (room_id, booking_date, start_time, end_time);
