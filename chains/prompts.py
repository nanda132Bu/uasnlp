"""
chains/prompts.py
-------------------
Kumpulan PromptTemplate dari LangChain yang dipakai di berbagai node
LangGraph. Memisahkan prompt ke file sendiri membuat kode lebih rapi
dan gampang di-tweak tanpa mengubah logic.

Library: LangChain (ChatPromptTemplate)
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------
# 1. Prompt untuk MENGKLASIFIKASIKAN intent user
#    (apakah dia mau nulis jurnal baru, atau mau tanya/ngobrol soal
#    jurnal lama)
# ---------------------------------------------------------------
INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Kamu adalah pengklasifikasi intent untuk aplikasi journaling. "
            "Tugasmu HANYA menjawab dengan satu kata: 'tulis' atau 'tanya'.\n\n"
            "Jawab 'tulis' jika user sedang menceritakan hari/perasaannya "
            "(ingin membuat entri jurnal baru).\n"
            "Jawab 'tanya' jika user mengajukan pertanyaan tentang entri "
            "jurnal sebelumnya, atau mengobrol reflektif soal masa lalu.\n\n"
            "Jawab HANYA dengan satu kata, tanpa penjelasan tambahan.",
        ),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------
# 2. Prompt untuk ANALISIS MOOD & TAG dari entri jurnal baru
# ---------------------------------------------------------------
MOOD_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Kamu adalah asisten yang menganalisis entri jurnal pribadi. "
            "Dari teks yang diberikan, tentukan:\n"
            "1. mood: SATU kata mood dominan (contoh: senang, sedih, cemas, "
            "lelah, bersyukur, marah, netral)\n"
            "2. tags: maksimal 3 kata kunci/tema singkat (contoh: kerjaan, "
            "keluarga, kesehatan)\n\n"
            "Jawab HANYA dalam format JSON persis seperti ini, tanpa "
            "teks lain:\n"
            '{{"mood": "...", "tags": ["...", "..."]}}',
        ),
        ("human", "{text}"),
    ]
)

# ---------------------------------------------------------------
# 3. Prompt untuk merespons user setelah entri jurnal disimpan
#    (mode 'tulis')
# ---------------------------------------------------------------
JOURNAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Kamu adalah teman jurnal yang hangat, suportif, dan reflektif. "
            "User baru saja menulis entri jurnal dengan mood '{mood}' "
            "dan topik {tags}. Berikan respons singkat (2-4 kalimat) yang "
            "validasi perasaan mereka dan, jika relevan, ajukan satu "
            "pertanyaan reflektif ringan. Gunakan Bahasa Indonesia yang "
            "natural dan hangat.",
        ),
        ("human", "{text}"),
    ]
)

# ---------------------------------------------------------------
# 4. Prompt untuk menjawab PERTANYAAN user berdasarkan konteks
#    entri jurnal lama yang ditemukan lewat RAG (mode 'tanya')
# ---------------------------------------------------------------
RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Kamu adalah asisten jurnal pribadi yang membantu user "
            "merefleksikan entri-entri jurnal mereka di masa lalu.\n\n"
            "Berikut adalah entri-entri jurnal relevan yang ditemukan:\n"
            "{context}\n\n"
            "Jawab pertanyaan user berdasarkan entri-entri di atas. "
            "Jika entri tidak cukup untuk menjawab, katakan dengan jujur. "
            "Jawab dengan empatik dan dalam Bahasa Indonesia.",
        ),
        ("human", "{question}"),
    ]
)
