"""
memory/journal_store.py
-------------------------
Penyimpanan metadata entri jurnal menggunakan SQLite.

Kenapa butuh ini selain vector store? Vector store (Chroma) bagus
untuk pencarian berbasis makna, tapi untuk keperluan UI seperti
"tampilkan semua entri urut tanggal" atau "hitung berapa kali mood
sedih bulan ini", database relasional sederhana lebih cocok dan cepat.

Library: standar Python (sqlite3) - bukan bagian dari 3 library wajib,
murni untuk kebutuhan penyimpanan data aplikasi.
"""

import sqlite3
import uuid
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "cerita.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Membuat tabel entries jika belum ada."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                mood TEXT,
                tags TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )


def save_entry(text: str, mood: str, tags: list[str]) -> dict:
    """Menyimpan entri baru ke SQLite dan mengembalikan datanya."""
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO entries (id, text, mood, tags, timestamp) VALUES (?, ?, ?, ?, ?)",
            (entry_id, text, mood, ", ".join(tags), timestamp),
        )
    return {
        "id": entry_id,
        "text": text,
        "mood": mood,
        "tags": tags,
        "timestamp": timestamp,
    }


def get_all_entries() -> list[dict]:
    """Mengambil semua entri jurnal, terbaru di atas."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def count_entries() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM entries").fetchone()
    return row["c"]
