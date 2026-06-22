"""
config.py
---------
Pusat konfigurasi aplikasi Tukang Cerita.

File ini bertanggung jawab untuk:
1. Memuat environment variables dari .env
2. Mengaktifkan LangSmith tracing secara otomatis
3. Menyediakan konstanta konfigurasi (nama model, direktori, dll)
   yang dipakai di seluruh bagian aplikasi.
"""

import os
from dotenv import load_dotenv

# Muat file .env (kalau ada) ke environment variables
load_dotenv()

# ---------------------------------------------------------------
# LangSmith Configuration
# ---------------------------------------------------------------
# LangSmith akan otomatis aktif jika environment variable di bawah
# ini ter-set. Tidak perlu kode tambahan di chain/graph kita -
# LangChain & LangGraph akan otomatis mengirim trace ke LangSmith
# selama variable ini ada.
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "tukang-cerita")

# ---------------------------------------------------------------
# Groq Configuration (LLM, gratis)
# ---------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------
# Embedding Configuration (lokal, lewat sentence-transformers)
# ---------------------------------------------------------------
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "cerita_entries"


def is_langsmith_enabled() -> bool:
    """Cek apakah API key LangSmith sudah diisi (bukan default kosong)."""
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def is_groq_configured() -> bool:
    """Cek apakah API key Groq sudah diisi."""
    return bool(GROQ_API_KEY)
