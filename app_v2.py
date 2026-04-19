"""
app_v2.py — Skills Intelligence RAG — Streamlit UI
===================================================
Drop-in replacement for app.py that exposes all new features:
  - Skill / domain / difficulty filtering sidebar
  - Pipeline mode selector (baseline, hybrid, full)
  - Skills profile panel per source document
  - Compression ratio display on sources

Run: streamlit run app_v2.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import DocumentLoader
from rag_engine_v2 import RAGEngineV2

load_dotenv()

st.set_page_config(
    page_title="Skills RAG — Temarios",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEMARIOS_PATH      = "temarios"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "rag_engine": None,
        "index_built": False,
        "messages": [],
        "last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# ENGINE INIT
# ============================================================

@st.cache_resource
def get_engine(api_key, model, top_k, use_hybrid, use_reranker, use_compression):
    return RAGEngineV2(
        api_key=api_key, model=model, top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_compression=use_compression,
    )


def build_or_load(engine: RAGEngineV2) -> bool:
    if engine.load_index():
        return True
    loader = DocumentLoader(TEMARIOS_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = loader.load_all_documents()
    if not chunks:
        return False
    engine.build_index(chunks, extract_skills=False)
    return True


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(engine: RAGEngineV2):
    st.sidebar.title("🧠 Skills RAG")
    st.sidebar.markdown("---")

    # --- Pipeline config ---
    st.sidebar.subheader("⚙️ Pipeline Mode")
    pipeline = st.sidebar.selectbox(
        "Select mode:",
        ["Full (hybrid + rerank + compress)", "Hybrid only", "Baseline (FAISS only)"],
    )

    st.sidebar.markdown("---")

    # --- Index stats ---
    st.sidebar.subheader("📊 Index Status")
    if st.session_state.index_built:
        stats = engine.get_index_stats()
        st.sidebar.success("✅ Index active")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Chunks", stats.get("total_fragmentos", 0))
        col2.metric("Carreras", len(stats.get("carreras", [])))
        if stats.get("skills_indexed"):
            st.sidebar.metric("Skills indexed", stats["skills_indexed"])
    else:
        st.sidebar.warning("⚠️ Index not built")

    st.sidebar.markdown("---")

    # --- Retrieval filters ---
    st.sidebar.subheader("🔍 Filters")

    carrera_filter = ""
    skill_filter   = ""
    domain_filter  = ""
    diff_filter    = ""

    if st.session_state.index_built:
        stats = engine.get_index_stats()

        # Carrera filter
        carreras = ["All"] + stats.get("carreras", [])
        sel_carrera = st.sidebar.selectbox("Carrera:", carreras)
        if sel_carrera != "All":
            carrera_filter = sel_carrera

        # Skills filter
        if engine.skills_store.is_loaded():
            domains = ["Any"] + engine.skills_store.get_all_domains()
            sel_domain = st.sidebar.selectbox("Domain:", domains)
            if sel_domain != "Any":
                domain_filter = sel_domain

            diffs = ["Any"] + engine.skills_store.get_all_difficulties()
            sel_diff = st.sidebar.selectbox("Difficulty:", diffs)
            if sel_diff != "Any":
                diff_filter = sel_diff

            skill_filter = st.sidebar.text_input(
                "Skill keyword:", placeholder="e.g. linear regression"
            )

    st.sidebar.markdown("---")

    # --- Reindex button ---
    st.sidebar.subheader("🔄 Index Management")
    extract_skills_toggle = st.sidebar.toggle(
        "Extract skills (slow, costs LLM tokens)",
        value=False,
    )
    if st.sidebar.button("Reindex documents"):
        with st.spinner("Processing PDFs and building index..."):
            loader = DocumentLoader(TEMARIOS_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks = loader.load_all_documents()
            if chunks:
                engine.build_index(chunks, extract_skills=extract_skills_toggle)
                st.session_state.index_built = True
                st.sidebar.success("Index rebuilt.")
                st.rerun()
            else:
                st.sidebar.error(f"No PDFs found in '{TEMARIOS_PATH}/'")

    return pipeline, carrera_filter, skill_filter, domain_filter, diff_filter


# ============================================================
# CHAT PANEL
# ============================================================

def render_chat(engine: RAGEngineV2, pipeline, carrera_filter,
                skill_filter, domain_filter, diff_filter):
    st.title("🧠 Skills Intelligence RAG")
    st.caption("University syllabus assistant with skill-aware retrieval, hybrid search, and re-ranking.")

    if not st.session_state.index_built:
        st.info(
            "**Getting started:**\n\n"
            "1. Place PDF syllabi in `temarios/`\n"
            "2. Click **Reindex documents** in the sidebar\n"
            "3. Ask anything about the curricula!"
        )
        return

    # Show active filters as tags
    active = [f for f in [carrera_filter, skill_filter, domain_filter, diff_filter] if f]
    if active:
        st.markdown("**Active filters:** " + " · ".join(f"`{f}`" for f in active))

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Input
    if user_input := st.chat_input("Ask about the syllabi..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = engine.query(
                    user_question=user_input,
                    carrera_filter=carrera_filter or None,
                    conversation_history=history or None,
                    skill_filter=skill_filter or None,
                    domain_filter=domain_filter or None,
                    difficulty_filter=diff_filter or None,
                )
            st.markdown(result["answer"])
            if result["sources"]:
                render_sources(result["sources"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()


def render_sources(sources: list):
    with st.expander(f"📚 Sources ({len(sources)} chunks)"):
        for i, s in enumerate(sources, 1):
            meta = s.get("skills_metadata", {})
            skills_str = ", ".join(meta.get("skills", [])[:4]) or "—"
            domain_str = meta.get("domain", "—")
            diff_str   = meta.get("difficulty", "—")
            comp_ratio = s.get("compression_ratio", 1.0)
            rerank     = s.get("rerank_score")
            hybrid     = s.get("hybrid_score")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**[{i}] {s['carrera']}** — `{s['source']}` — p.{s['page']}\n\n"
                    f"🏷 Skills: `{skills_str}` | Domain: `{domain_str}` | "
                    f"Difficulty: `{diff_str}`"
                )
            with col2:
                if rerank:
                    st.metric("Rerank", f"{rerank:.2f}")
                elif hybrid:
                    st.metric("Hybrid", f"{hybrid:.4f}")
                if comp_ratio < 0.95:
                    st.caption(f"Compressed to {int(comp_ratio*100)}%")

            st.caption(s["text"][:350] + ("..." if len(s["text"]) > 350 else ""))
            if i < len(sources):
                st.divider()


# ============================================================
# MAIN
# ============================================================

def main():
    init_state()

    if not OPENROUTER_API_KEY:
        st.error("Set OPENROUTER_API_KEY in your .env file.")
        st.stop()

    # Map pipeline choice to feature flags
    pipeline_cfg = {
        "Full (hybrid + rerank + compress)": (True,  True,  True),
        "Hybrid only":                        (True,  False, False),
        "Baseline (FAISS only)":              (False, False, False),
    }

    # We need to render sidebar to get pipeline choice — use temp defaults
    # and let cache handle engine switching
    use_h, use_r, use_c = True, True, True  # default to full

    engine = get_engine(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        top_k=RAG_TOP_K,
        use_hybrid=use_h,
        use_reranker=use_r,
        use_compression=use_c,
    )
    st.session_state.rag_engine = engine

    if not st.session_state.index_built:
        with st.spinner("Loading index..."):
            if build_or_load(engine):
                st.session_state.index_built = True

    pipeline, carrera_f, skill_f, domain_f, diff_f = render_sidebar(engine)

    render_chat(engine, pipeline, carrera_f, skill_f, domain_f, diff_f)


if __name__ == "__main__":
    main()
