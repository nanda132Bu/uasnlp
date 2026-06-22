"""
graph/state.py
----------------
Definisi State untuk LangGraph.

Di LangGraph, "State" adalah objek (biasanya TypedDict) yang dibawa
dan diperbarui oleh setiap node selagi data mengalir melalui graph.
Setiap node menerima state saat ini, melakukan sesuatu, lalu
mengembalikan dict berisi field yang ingin diperbarui.

Library: LangGraph (TypedDict-based state)
"""

from typing import TypedDict, Optional


class JurnalState(TypedDict, total=False):
    # --- Input dari user ---
    user_input: str  # teks mentah yang diketik user

    # --- Hasil klasifikasi intent ---
    intent: str  # "tulis" atau "tanya"

    # --- Field khusus mode "tulis" ---
    mood: Optional[str]
    tags: Optional[list[str]]
    entry_saved: Optional[bool]

    # --- Field khusus mode "tanya" (RAG) ---
    retrieved_context: Optional[str]  # gabungan entri jurnal relevan

    # --- Output akhir yang ditampilkan ke user ---
    response: str

    # --- Riwayat percakapan (memory jangka pendek dalam 1 sesi) ---
    chat_history: list[dict]
