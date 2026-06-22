"""
app.py
-------
Antarmuka Streamlit untuk Tukang Cerita.

Aplikasi ini punya 2 mode interaksi dalam satu chat box:
1. Ketik cerita harian -> otomatis dideteksi sebagai entri jurnal baru
   (mood & tag dianalisis, lalu disimpan ke memory).
2. Ketik pertanyaan soal masa lalu -> otomatis dideteksi sebagai mode
   tanya, lalu dijawab berdasarkan entri jurnal lama (RAG).

Routing antara dua mode ini sepenuhnya ditangani oleh LangGraph
(lihat graph/builder.py), bukan oleh logic Streamlit.
"""

import streamlit as st

import config  # noqa: F401  (memastikan env & LangSmith ter-setup di awal)
from graph.builder import jurnal_graph
from memory.journal_store import init_db, get_all_entries, count_entries

# ---------------------------------------------------------------
# Setup awal
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Tukang Cerita",
    page_icon="📔",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------
# Custom CSS — Elegant Dark Mode Design
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary:    #0d0f14;
        --bg-secondary:  #13161f;
        --bg-card:       rgba(255,255,255,0.04);
        --bg-card-hover: rgba(255,255,255,0.07);
        --border:        rgba(255,255,255,0.08);
        --border-accent: rgba(139,92,246,0.40);
        --accent-1:      #8b5cf6;
        --accent-2:      #06b6d4;
        --accent-3:      #f59e0b;
        --text-primary:  #f1f5f9;
        --text-muted:    #94a3b8;
        --text-dim:      #64748b;
        --user-bubble:   linear-gradient(135deg,#8b5cf6,#6d28d9);
        --bot-bubble:    rgba(255,255,255,0.05);
        --glow:          0 0 30px rgba(139,92,246,0.15);
        --radius:        14px;
        --radius-sm:     8px;
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 2rem 2.5rem 5rem !important;
        max-width: 900px !important;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--accent-1); border-radius: 99px; }

    /* ══════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #13161f 100%) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }

    /* Sidebar brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    .sidebar-brand .brand-icon {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 15px rgba(139,92,246,0.4);
        flex-shrink: 0;
    }
    .sidebar-brand .brand-text h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .sidebar-brand .brand-text p {
        font-size: 0.72rem !important;
        color: var(--text-dim) !important;
        margin: 0 !important;
    }

    /* Sidebar divider */
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.2rem 0;
    }

    /* Sidebar section title */
    .sidebar-section-title {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.8rem;
    }

    /* Metric card */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.2s;
    }
    .metric-card:hover { background: var(--bg-card-hover); border-color: var(--border-accent); }
    .metric-card .metric-icon { font-size: 1.5rem; }
    .metric-card .metric-info .metric-value {
        font-size: 1.5rem; font-weight: 700;
        color: var(--accent-1);
        line-height: 1;
    }
    .metric-card .metric-info .metric-label {
        font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;
    }

    /* Stack badge */
    .stack-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 99px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 3px 2px;
        transition: all 0.2s;
    }
    .stack-badge:hover { border-color: var(--border-accent); color: var(--text-primary); }

    /* Status pill */
    .status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        border-radius: var(--radius-sm);
        padding: 8px 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-pill.success { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
    .status-pill.warning { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
    .status-pill.error   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
    .status-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    /* Journal entry card */
    .entry-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    .entry-card:hover { background: var(--bg-card-hover); border-color: var(--border-accent); }
    .entry-card .entry-meta {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.72rem; color: var(--text-dim); margin-bottom: 4px;
    }
    .entry-card .entry-mood {
        background: rgba(139,92,246,0.15); color: var(--accent-1);
        border-radius: 99px; padding: 1px 8px;
        font-weight: 500;
    }
    .entry-card .entry-text {
        font-size: 0.8rem; color: var(--text-muted); line-height: 1.4;
    }

    /* ══════════════════════════════════════
       MAIN HEADER
    ══════════════════════════════════════ */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 2rem;
        position: relative;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 50%; transform: translateX(-50%);
        width: 300px; height: 200px;
        background: radial-gradient(ellipse, rgba(139,92,246,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header .header-emoji {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 20px rgba(139,92,246,0.5));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-8px); }
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #f1f5f9 30%, var(--accent-1));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        line-height: 1.15 !important;
    }
    .main-header p {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.5rem !important;
    }

    /* Mode indicator strip */
    .mode-strip {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-top: 1rem;
    }
    .mode-chip {
        display: flex; align-items: center; gap: 6px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 99px;
        padding: 5px 14px;
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .mode-chip .chip-dot { width: 6px; height: 6px; border-radius: 50%; }
    .mode-chip.write .chip-dot { background: var(--accent-1); }
    .mode-chip.ask .chip-dot   { background: var(--accent-2); }

    /* ══════════════════════════════════════
       CHAT MESSAGES
    ══════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
    }

    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
    }

    /* Avatar */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, var(--accent-1), #6d28d9) !important;
        border-radius: 50% !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, var(--accent-2), #0891b2) !important;
        border-radius: 50% !important;
    }

    /* Bubble wrapper */
    [data-testid="stChatMessageContent"] {
        background: var(--bot-bubble) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.9rem 1.2rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        line-height: 1.65 !important;
    }

    /* ══════════════════════════════════════
       CHAT INPUT
    ══════════════════════════════════════ */
    [data-testid="stChatInput"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border-accent) !important;
        background: rgba(139,92,246,0.05) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: var(--glow) !important;
        transition: all 0.3s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.2), var(--glow) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, var(--accent-1), #6d28d9) !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* ══════════════════════════════════════
       BADGE / CAPTION TWEAKS
    ══════════════════════════════════════ */
    .intent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 99px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }
    .intent-badge.write {
        background: rgba(139,92,246,0.15); color: var(--accent-1);
        border: 1px solid rgba(139,92,246,0.3);
    }
    .intent-badge.ask {
        background: rgba(6,182,212,0.12); color: var(--accent-2);
        border: 1px solid rgba(6,182,212,0.3);
    }

    .mood-tag-row {
        display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
    }
    .mood-chip {
        background: rgba(245,158,11,0.12); color: var(--accent-3);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 99px; padding: 2px 10px;
        font-size: 0.72rem; font-weight: 500;
    }
    .tag-chip {
        background: rgba(255,255,255,0.06); color: var(--text-muted);
        border: 1px solid var(--border);
        border-radius: 99px; padding: 2px 10px;
        font-size: 0.72rem;
    }

    /* Streamlit caption override */
    small, .stCaption, [data-testid="stCaption"] {
        color: var(--text-dim) !important;
        font-size: 0.75rem !important;
    }

    /* ══════════════════════════════════════
       EXPANDER
    ══════════════════════════════════════ */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
    }

    /* ══════════════════════════════════════
       SPINNER
    ══════════════════════════════════════ */
    [data-testid="stSpinner"] > div {
        border-color: var(--accent-1) transparent transparent transparent !important;
    }

    /* ══════════════════════════════════════
       MISC
    ══════════════════════════════════════ */
    .stMarkdown a { color: var(--accent-1) !important; }
    .stMarkdown code {
        background: rgba(139,92,246,0.12) !important;
        color: var(--accent-1) !important;
        border-radius: 4px !important;
        padding: 1px 6px !important;
    }

    /* Fade-in animation for chat */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
        animation: fadeSlideUp 0.3s ease forwards;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    total_entries = count_entries()

    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">📔</div>
            <div class="brand-text">
                <h1>Tukang Cerita</h1>
                <p>AI Journaling Assistant</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="sidebar-section-title">Statistik</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📖</div>
            <div class="metric-info">
                <div class="metric-value">{total_entries}</div>
                <div class="metric-label">Total cerita tersimpan</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="sidebar-section-title">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
            <span class="stack-badge">⛓️ LangChain</span>
            <span class="stack-badge">🕸️ LangGraph</span>
            <span class="stack-badge">🔍 LangSmith</span>
            <span class="stack-badge">🤖 Groq LLM</span>
            <span class="stack-badge">🧠 ChromaDB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Status
    st.markdown('<div class="sidebar-section-title">Status</div>', unsafe_allow_html=True)
    if config.is_langsmith_enabled():
        st.markdown(
            '<div class="status-pill success"><span class="dot"></span>LangSmith tracing aktif</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill warning"><span class="dot"></span>LangSmith belum dikonfigurasi</div>',
            unsafe_allow_html=True,
        )

    if not config.is_groq_configured():
        st.markdown(
            '<div class="status-pill error" style="margin-top:6px"><span class="dot"></span>GROQ_API_KEY belum diisi</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Journal entries
    st.markdown('<div class="sidebar-section-title">Riwayat Cerita</div>', unsafe_allow_html=True)
    with st.expander("📚 Lihat semua cerita"):
        entries = get_all_entries()
        if not entries:
            st.markdown(
                '<p style="font-size:0.8rem;color:var(--text-dim);text-align:center;padding:8px 0">Belum ada cerita tersimpan.</p>',
                unsafe_allow_html=True,
            )
        for e in entries:
            ts = e["timestamp"][:16].replace("T", " ")
            preview = e["text"][:100] + ("..." if len(e["text"]) > 100 else "")
            st.markdown(
                f"""
                <div class="entry-card">
                    <div class="entry-meta">
                        <span>🕐 {ts}</span>
                        <span class="entry-mood">✨ {e['mood']}</span>
                        <span>{e['tags']}</span>
                    </div>
                    <div class="entry-text">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <span class="header-emoji">📔</span>
        <h1>Tukang Cerita</h1>
        <p>Ceritakan harimu, atau tanya soal kenangan lamamu</p>
        <div class="mode-strip">
            <div class="mode-chip write">
                <span class="chip-dot"></span>
                Mode Tulis — ceritakan harimu
            </div>
            <div class="mode-chip ask">
                <span class="chip-dot"></span>
                Mode Tanya — gali kenangan lama (RAG)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------
# Chat input & LangGraph invocation
# ---------------------------------------------------------------
user_input = st.chat_input("Tulis cerita hari ini, atau tanya sesuatu tentang masa lalumu...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Memproses lewat LangGraph..."):
            result = jurnal_graph.invoke({"user_input": user_input})

            intent   = result.get("intent", "?")
            response = result.get("response", "(tidak ada respons)")

        # Intent badge
        if intent == "tulis":
            badge_html = '<span class="intent-badge write">📝 Mode Tulis Cerita</span>'
        else:
            badge_html = '<span class="intent-badge ask">🔍 Mode Tanya (RAG)</span>'

        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(response)

        # Mood & tags (hanya untuk mode tulis)
        if intent == "tulis":
            mood = result.get("mood", "-")
            tags = result.get("tags", [])
            tag_chips = "".join(f'<span class="tag-chip">#{t}</span>' for t in tags)
            st.markdown(
                f"""
                <div class="mood-tag-row">
                    <span class="mood-chip">✨ {mood}</span>
                    {tag_chips}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.session_state.chat_history.append({"role": "assistant", "content": response})
