"""
graph/builder.py
------------------
Di sinilah graph LangGraph benar-benar dirangkai: node-node dari
nodes.py disambungkan dengan edges (alur biasa) dan conditional
edges (percabangan berdasarkan kondisi).

Visualisasi alur:

    START
      |
      v
  classify_intent
      |
      |--(intent == "tulis")--> write_journal --> journal_response --> END
      |
      |--(intent == "tanya")--> retrieve_context --> rag_answer --> END

Library: LangGraph (StateGraph, START, END)
"""

from langgraph.graph import StateGraph, START, END

from graph.state import JurnalState
from graph.nodes import (
    classify_intent_node,
    write_journal_node,
    retrieve_context_node,
    journal_response_node,
    rag_answer_node,
    route_by_intent,
)


def build_jurnal_graph():
    """
    Menyusun dan meng-compile StateGraph untuk Jurnal AI.
    Mengembalikan graph yang sudah siap di-invoke (.invoke(...)).
    """
    graph = StateGraph(JurnalState)

    # Daftarkan semua node
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("write_journal", write_journal_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("journal_response", journal_response_node)
    graph.add_node("rag_answer", rag_answer_node)

    # Edge awal: dari START selalu mulai dengan klasifikasi intent
    graph.add_edge(START, "classify_intent")

    # Conditional edge: percabangan berdasarkan hasil classify_intent
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "tulis": "write_journal",
            "tanya": "retrieve_context",
        },
    )

    # Cabang "tulis": simpan entri -> beri respons reflektif -> selesai
    graph.add_edge("write_journal", "journal_response")
    graph.add_edge("journal_response", END)

    # Cabang "tanya": retrieve konteks -> jawab dengan RAG -> selesai
    graph.add_edge("retrieve_context", "rag_answer")
    graph.add_edge("rag_answer", END)

    return graph.compile()


# Instance graph siap pakai, di-import oleh app.py
jurnal_graph = build_jurnal_graph()
