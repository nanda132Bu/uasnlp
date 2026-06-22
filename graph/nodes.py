"""
graph/nodes.py
----------------
Semua "node" (langkah) yang membentuk alur kerja LangGraph Jurnal AI.

Setiap fungsi node punya signature yang sama: menerima JurnalState,
mengembalikan dict (partial state update). Ini adalah konvensi
standar LangGraph.

Library yang dipakai di sini:
- LangChain: ChatGroq, prompt invocation
- (vector store Chroma dipanggil lewat memory/vectorstore.py)
"""

from langchain_groq import ChatGroq

import config
from graph.state import JurnalState
from chains.prompts import (
    INTENT_CLASSIFIER_PROMPT,
    JOURNAL_RESPONSE_PROMPT,
    RAG_ANSWER_PROMPT,
)
from chains.mood_classifier import classify_mood_and_tags
from memory.vectorstore import get_vectorstore, add_journal_entry, search_journal
from memory.journal_store import save_entry


def _get_llm(temperature: float = 0.7) -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
    )


# ---------------------------------------------------------------
# NODE 1: Klasifikasi intent
# ---------------------------------------------------------------
def classify_intent_node(state: JurnalState) -> dict:
    """
    Node pertama di graph. Menentukan apakah user sedang menulis
    entri jurnal baru ("tulis") atau bertanya/refleksi soal jurnal
    lama ("tanya"). Hasil node ini menentukan cabang mana yang akan
    dilalui (lihat conditional edge di graph/builder.py).
    """
    llm = _get_llm(temperature=0.0)
    chain = INTENT_CLASSIFIER_PROMPT | llm
    result = chain.invoke({"input": state["user_input"]})

    intent_raw = result.content.strip().lower()
    intent = "tanya" if "tanya" in intent_raw else "tulis"

    return {"intent": intent}


# ---------------------------------------------------------------
# NODE 2a (cabang "tulis"): Simpan entri jurnal baru
# ---------------------------------------------------------------
def write_journal_node(state: JurnalState) -> dict:
    """
    Cabang ketika user menulis entri jurnal baru:
    1. Analisis mood & tags dari teks (pakai chain LangChain)
    2. Simpan metadata ke SQLite
    3. Simpan embedding teks ke vector store (Chroma) -> bagian RAG
    """
    text = state["user_input"]

    analysis = classify_mood_and_tags(text)
    mood = analysis["mood"]
    tags = analysis["tags"]

    # Simpan ke SQLite (metadata terstruktur)
    saved = save_entry(text=text, mood=mood, tags=tags)

    # Simpan ke vector store (untuk pencarian semantik nanti)
    vectorstore = get_vectorstore()
    add_journal_entry(
        vectorstore=vectorstore,
        entry_id=saved["id"],
        text=text,
        mood=mood,
        tags=tags,
        timestamp=saved["timestamp"],
    )

    return {"mood": mood, "tags": tags, "entry_saved": True}


# ---------------------------------------------------------------
# NODE 2b (cabang "tanya"): Retrieval dari vector store (RAG)
# ---------------------------------------------------------------
def retrieve_context_node(state: JurnalState) -> dict:
    """
    Cabang ketika user bertanya soal jurnal lama. Melakukan
    similarity search di Chroma vector store untuk menemukan
    entri-entri jurnal yang paling relevan dengan pertanyaan user.
    Inilah inti dari RAG (Retrieval-Augmented Generation).
    """
    vectorstore = get_vectorstore()
    docs = search_journal(vectorstore, query=state["user_input"], k=3)

    if not docs:
        context = "(Belum ada entri jurnal yang tersimpan.)"
    else:
        parts = []
        for doc in docs:
            ts = doc.metadata.get("timestamp", "?")
            mood = doc.metadata.get("mood", "?")
            parts.append(f"[{ts}] (mood: {mood}) {doc.page_content}")
        context = "\n\n".join(parts)

    return {"retrieved_context": context}


# ---------------------------------------------------------------
# NODE 3a: Generate respons untuk mode "tulis"
# ---------------------------------------------------------------
def journal_response_node(state: JurnalState) -> dict:
    """Menghasilkan respons hangat & reflektif setelah entri disimpan."""
    llm = _get_llm(temperature=0.7)
    chain = JOURNAL_RESPONSE_PROMPT | llm

    result = chain.invoke(
        {
            "text": state["user_input"],
            "mood": state.get("mood", "netral"),
            "tags": ", ".join(state.get("tags", [])) or "umum",
        }
    )
    return {"response": result.content}


# ---------------------------------------------------------------
# NODE 3b: Generate jawaban untuk mode "tanya" (pakai konteks RAG)
# ---------------------------------------------------------------
def rag_answer_node(state: JurnalState) -> dict:
    """Menghasilkan jawaban berbasis entri jurnal lama yang ditemukan."""
    llm = _get_llm(temperature=0.5)
    chain = RAG_ANSWER_PROMPT | llm

    result = chain.invoke(
        {
            "context": state.get("retrieved_context", ""),
            "question": state["user_input"],
        }
    )
    return {"response": result.content}


# ---------------------------------------------------------------
# ROUTER: dipakai oleh conditional edge di builder.py
# ---------------------------------------------------------------
def route_by_intent(state: JurnalState) -> str:
    """
    Fungsi router murni (tanpa LLM call) yang membaca hasil
    klasifikasi intent dan mengarahkan graph ke node berikutnya
    yang sesuai. Ini contoh "conditional edge" di LangGraph.
    """
    return state["intent"]
