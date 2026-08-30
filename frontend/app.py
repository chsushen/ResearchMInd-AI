import os
import io
import json
import time
import requests
import streamlit as st
import pandas as pd

# Streamlit Page Config
st.set_page_config(
    page_title="Research Mind AI — Academic Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")

# Load Custom CSS
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None


# Helper API Client Functions
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


def fetch_indexed_documents():
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=4)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def upload_pdf_file(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        r = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
        if r.status_code == 201:
            return True, r.json()
        return False, r.json().get("detail", "Upload failed.")
    except Exception as e:
        return False, str(e)


def delete_document_api(doc_id):
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{doc_id}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def query_rag_api(query_text, doc_ids=None, api_key=None, history=None):
    try:
        payload = {
            "query": query_text,
            "doc_ids": doc_ids or None,
            "gemini_api_key": api_key or None,
            "conversation_history": history or [],
            "top_k": 6,
        }
        r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=45)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json().get("detail", "Query failed.")
    except Exception as e:
        return False, str(e)


def compare_documents_api(doc_ids, dimensions=None, api_key=None):
    try:
        payload = {
            "doc_ids": doc_ids,
            "dimensions": dimensions,
            "gemini_api_key": api_key or None,
        }
        r = requests.post(f"{BACKEND_URL}/compare", json=payload, timeout=60)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json().get("detail", "Comparison failed.")
    except Exception as e:
        return False, str(e)


def extract_bibtex_api(doc_id):
    try:
        r = requests.post(f"{BACKEND_URL}/extract-bibtex", json={"doc_id": doc_id}, timeout=15)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json().get("detail", "BibTeX extraction failed.")
    except Exception as e:
        return False, str(e)


# ==============================================================================
# SIDEBAR: Document Management & System Health
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔬 Research Mind AI")
    st.caption("FAANG-Caliber Academic Research Engine")

    # Backend Connection Status
    is_healthy, health_info = check_backend_health()
    if is_healthy:
        st.success(f"🟢 Backend Online (Chunks: {health_info.get('total_chunks', 0)})", icon="✅")
    else:
        st.warning("🟠 Backend Offline or Initializing...", icon="⚠️")
        st.caption(f"Target: `{BACKEND_URL}`")

    # Gemini API Key Configuration
    st.markdown("---")
    st.markdown("##### 🔑 Google Gemini API Key")
    key_input = st.text_input(
        "API Key (Optional for fallback)",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Obtain a free key at https://aistudio.google.com/. If blank, deterministic grounded fallback is used.",
    )
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input

    # Document Upload Section
    st.markdown("---")
    st.markdown("##### 📄 Ingest Academic Literature")
    uploaded_file = st.file_uploader(
        "Upload PDF Paper",
        type=["pdf"],
        help="Upload standard research papers (arXiv, IEEE, ACM, Nature). 100+ pages supported.",
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Paper", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}' with page chunking & hybrid vectors..."):
                success, res = upload_pdf_file(uploaded_file)
                if success:
                    st.success(f"Indexed: {res.get('title')}", icon="🎉")
                    st.caption(f"Processed {res.get('pages_processed')} pages ({res.get('chunks_created')} chunks).")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {res}")

    # Indexed Papers List
    st.markdown("---")
    docs = fetch_indexed_documents()
    st.markdown(f"##### 📚 Indexed Papers ({len(docs)})")

    if not docs:
        st.info("No papers indexed yet. Upload a PDF above to get started.")
    else:
        for doc in docs:
            with st.container():
                st.markdown(
                    f"""
                    <div class="doc-item-card">
                        <div class="doc-item-title" title="{doc.get('title')}">📄 {doc.get('title')}</div>
                        <div class="doc-item-meta">
                            {doc.get('total_pages')} pages • {doc.get('chunk_count')} chunks<br>
                            ID: <code>{doc.get('doc_id')}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_del, col_info = st.columns([1, 1])
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{doc.get('doc_id')}", use_container_width=True):
                        if delete_document_api(doc.get("doc_id")):
                            st.toast("Document deleted successfully.")
                            time.sleep(0.5)
                            st.rerun()
                with col_info:
                    if doc.get("abstract"):
                        with st.popover("Abstract"):
                            st.caption(doc.get("abstract"))


# ==============================================================================
# MAIN VIEW: Tabs for Grounded Chat, Multi-Paper Comparison, BibTeX Generator
# ==============================================================================
st.markdown('<div class="main-header">Research Mind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Document Academic Copilot with Hybrid Retrieval (ChromaDB + BM25) and Strict Page-Level Grounding</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_compare, tab_bibtex = st.tabs(
    ["💬 Grounded Research Chat", "📊 Multi-Paper Comparison", "📝 One-Click BibTeX"]
)

# ------------------------------------------------------------------------------
# TAB 1: Multi-Turn Grounded Chat
# ------------------------------------------------------------------------------
with tab_chat:
    # Scope Filter Controls
    col_scope, col_clear = st.columns([4, 1])
    with col_scope:
        doc_options = {d["doc_id"]: d["title"] for d in docs}
        selected_doc_ids = st.multiselect(
            "Filter Literature Scope (Default: All Papers)",
            options=list(doc_options.keys()),
            format_func=lambda x: f"{doc_options.get(x, x)[:50]}...",
            placeholder="Select specific papers to query or leave empty for global library search",
        )
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Render Multi-turn Conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 Grounded Sources ({len(msg['citations'])} page citations verified)", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-card-header">
                                    <span>📄 {c.get('doc_title')}</span>
                                    <span class="citation-pill">Page {c.get('page_number')}</span>
                                </div>
                                <div class="source-card-body">"{c.get('snippet')}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # User Input
    if user_query := st.chat_input("Ask a scientific question, request proof analysis, or compare empirical findings..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Grounded RAG Answer
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing answer with hybrid RRF retrieval and page citation grounding..."):
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-6:]
                ]
                success, response = query_rag_api(
                    query_text=user_query,
                    doc_ids=selected_doc_ids if selected_doc_ids else None,
                    api_key=st.session_state.api_key,
                    history=history_payload,
                )

                if success:
                    answer_text = response.get("answer", "")
                    citations = response.get("citations", [])
                    latency = response.get("latency_ms", 0)

                    st.markdown(answer_text)

                    if citations:
                        with st.expander(f"📚 Grounded Sources ({len(citations)} page citations verified • {latency}ms)", expanded=True):
                            for c in citations:
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-card-header">
                                            <span>📄 {c.get('doc_title')}</span>
                                            <span class="citation-pill">Page {c.get('page_number')}</span>
                                        </div>
                                        <div class="source-card-body">"{c.get('snippet')}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save Assistant Message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                    })
                else:
                    st.error(f"Synthesis failed: {response}")


# ------------------------------------------------------------------------------
# TAB 2: Multi-Paper Comparison Dashboard
# ------------------------------------------------------------------------------
with tab_compare:
    st.markdown("#### 📊 Multi-Paper Comparative Synthesis Matrix")
    st.caption("Select two or more academic papers to cross-analyze methodologies, datasets, benchmark metrics, and limitations.")

    if len(docs) < 2:
        st.info("Please index at least two papers in the sidebar to generate a comparative analysis matrix.")
    else:
        compare_selected = st.multiselect(
            "Select Papers to Compare",
            options=[d["doc_id"] for d in docs],
            format_func=lambda did: next((d["title"] for d in docs if d["doc_id"] == did), did),
            default=[d["doc_id"] for d in docs[:min(3, len(docs))]],
        )

        dim_options = [
            "Research Objective & Problem Statement",
            "Proposed Methodology & Architecture",
            "Empirical Datasets & Benchmarks",
            "Key Quantitative Findings",
            "Identified Limitations & Trade-offs",
        ]
        selected_dimensions = st.multiselect(
            "Comparison Dimensions",
            options=dim_options,
            default=dim_options,
        )

        if st.button("⚡ Generate Comparison Matrix", type="primary", disabled=len(compare_selected) < 2):
            with st.spinner("Extracting comparative dimensions across selected literature..."):
                success, comp_res = compare_documents_api(
                    doc_ids=compare_selected,
                    dimensions=selected_dimensions,
                    api_key=st.session_state.api_key,
                )

                if success:
                    st.session_state.comparison_data = comp_res
                else:
                    st.error(f"Comparison failed: {comp_res}")

        # Render Comparison Results
        if st.session_state.comparison_data:
            comp_res = st.session_state.comparison_data

            st.markdown("##### 📌 Executive Comparative Brief")
            st.info(comp_res.get("executive_summary", ""))

            st.markdown("##### 📋 Dimension Comparison Matrix")
            st.markdown(comp_res.get("markdown_table", ""))

            # Export Buttons
            col_export_md, col_export_csv = st.columns(2)
            with col_export_md:
                md_content = f"# Multi-Paper Comparison Matrix\n\n{comp_res.get('executive_summary')}\n\n{comp_res.get('markdown_table')}"
                st.download_button(
                    label="📥 Download Markdown (.md)",
                    data=md_content,
                    file_name="paper_comparison_matrix.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_export_csv:
                # Convert dimensions into DataFrame for CSV export
                dims = comp_res.get("dimensions", [])
                titles = comp_res.get("doc_titles", [])
                data_rows = []
                for d in dims:
                    row = {"Dimension": d["dimension"]}
                    row.update(d.get("evaluations", {}))
                    data_rows.append(row)
                df_comp = pd.DataFrame(data_rows)
                csv_buffer = io.StringIO()
                df_comp.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download CSV (.csv)",
                    data=csv_buffer.getvalue(),
                    file_name="paper_comparison_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ------------------------------------------------------------------------------
# TAB 3: One-Click BibTeX Generator
# ------------------------------------------------------------------------------
with tab_bibtex:
    st.markdown("#### 📝 One-Click BibTeX Citation Generator")
    st.caption("Generate standard, publication-ready BibTeX entries for your thesis or research manuscript.")

    if not docs:
        st.info("Upload and index research papers in the sidebar to generate BibTeX entries.")
    else:
        target_doc_id = st.selectbox(
            "Select Paper for Citation",
            options=[d["doc_id"] for d in docs],
            format_func=lambda did: next((d["title"] for d in docs if d["doc_id"] == did), did),
        )

        if target_doc_id:
            with st.spinner("Extracting BibTeX metadata..."):
                success, bib_res = extract_bibtex_api(target_doc_id)
                if success:
                    st.code(bib_res.get("bibtex", ""), language="latex")

                    col_copy, col_download = st.columns(2)
                    with col_download:
                        st.download_button(
                            label="📥 Download .bib File",
                            data=bib_res.get("bibtex", ""),
                            file_name=f"{bib_res.get('citation_key', 'citation')}.bib",
                            mime="text/plain",
                            use_container_width=True,
                        )
                else:
                    st.error(f"Failed to generate BibTeX: {bib_res}")
