"""
memory/vectorstore.py
----------------------
Modul ini mengatur Vector Store (ChromaDB) yang menjadi "otak memory"
dari Tukang Cerita. Setiap entri cerita/jurnal yang ditulis user akan
diubah menjadi embedding (representasi numerik makna teks) lalu
disimpan di sini. Saat user bertanya soal cerita lama, kita akan
mencari entri yang paling relevan secara makna (bukan cuma kata
kunci) - inilah yang disebut RAG (Retrieval-Augmented Generation).

Embedding dihitung secara lokal di komputer (lewat library
sentence-transformers), jadi tidak butuh API key/token HuggingFace
dan tidak akan kena masalah izin/rate-limit dari Inference API.
Model akan otomatis terdownload sekali saat pertama kali dijalankan
lalu disimpan di cache lokal.

Library yang dipakai: LangChain (untuk wrapper embedding & vector store)
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Membuat objek embedding model yang berjalan lokal di komputer
    (CPU), tanpa perlu memanggil API HuggingFace. Model ini mengubah
    teks menjadi vector (list angka) yang merepresentasikan makna
    teks tersebut.
    """
    return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    """
    Membuat / membuka koneksi ke ChromaDB yang tersimpan di disk
    (persistent), sehingga data cerita tidak hilang saat aplikasi
    di-restart.
    """
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def add_journal_entry(
    vectorstore: Chroma,
    entry_id: str,
    text: str,
    mood: str,
    tags: list[str],
    timestamp: str,
) -> None:
    """
    Menyimpan satu entri cerita ke vector store beserta metadata-nya
    (mood, tag, waktu) agar bisa difilter/ditelusuri nanti.
    """
    doc = Document(
        page_content=text,
        metadata={
            "entry_id": entry_id,
            "mood": mood,
            "tags": ", ".join(tags),
            "timestamp": timestamp,
        },
    )
    vectorstore.add_documents(documents=[doc], ids=[entry_id])


def search_journal(vectorstore: Chroma, query: str, k: int = 3) -> list[Document]:
    """
    Mencari entri cerita yang paling relevan secara makna dengan
    pertanyaan/topik yang diberikan (similarity search).

    Args:
        query: pertanyaan atau topik yang ingin dicari, misal
               "gimana mood aku soal kerjaan minggu lalu?"
        k: jumlah entri teratas yang ingin diambil
    """
    return vectorstore.similarity_search(query, k=k)
