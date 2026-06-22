"""
chains/mood_classifier.py
---------------------------
Contoh konkret penggunaan LangChain Expression Language (LCEL) untuk
menyusun chain: prompt -> LLM -> output parser.

Chain ini mengambil teks cerita mentah dan mengembalikan dict berisi
mood + tags dalam bentuk Python object (bukan string JSON mentah).

Library: LangChain (ChatGroq, StrOutputParser, LCEL pipe operator)
"""

import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

import config
from chains.prompts import MOOD_ANALYSIS_PROMPT


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Membuat instance ChatGroq. Temperature rendah (0.3) dipakai
    untuk tugas klasifikasi agar hasil lebih konsisten/deterministik,
    berbeda dengan temperature lebih tinggi untuk tugas generatif
    seperti respons reflektif.
    """
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
    )


def build_mood_classifier_chain():
    """
    Menyusun chain LCEL: prompt -> llm -> string parser.
    Inilah pola khas LangChain: menyambungkan komponen dengan
    operator pipe (|) menjadi satu alur yang bisa di-invoke.
    """
    llm = get_llm(temperature=0.3)
    parser = StrOutputParser()
    chain = MOOD_ANALYSIS_PROMPT | llm | parser
    return chain


def classify_mood_and_tags(text: str) -> dict:
    """
    Menjalankan chain klasifikasi mood dan mem-parsing hasil JSON
    dari LLM menjadi dict Python. Ada fallback jika LLM tidak
    mengembalikan JSON valid.
    """
    chain = build_mood_classifier_chain()
    raw_output = chain.invoke({"text": text})

    try:
        # Bersihkan kemungkinan markdown code fence dari output LLM
        cleaned = raw_output.strip().strip("```").replace("json", "", 1).strip()
        result = json.loads(cleaned)
        mood = result.get("mood", "netral")
        tags = result.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        return {"mood": mood, "tags": tags[:3]}
    except (json.JSONDecodeError, AttributeError):
        # Fallback aman kalau parsing gagal
        return {"mood": "netral", "tags": []}
