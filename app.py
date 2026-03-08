import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile

from src.rag_pipeline import process_pdf, run_rag, load_vector_db

# PAGE CONFIG (ONLY ONCE)
st.set_page_config(
    page_title="AuditMind AI",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
text-align:center;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:30px;
}

.user-msg{
background:#ff4b4b;
padding:12px;
border-radius:10px;
color:white;
margin-bottom:10px;
}

.bot-msg{
background:#1e1e1e;
padding:12px;
border-radius:10px;
margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">🤖 AuditMind AI</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Upload audit reports and ask questions</div>',
unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
with st.sidebar:

    st.title("AuditMind AI")

    st.write("### Features")

    st.write("""
    📄 PDF Audit Analysis  
    🔎 Semantic Search (FAISS)  
    🤖 LLM Based Insights  
    📊 Source Page References
    """)

# ---------- LOAD VECTOR DB ----------
load_vector_db()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_files" not in st.session_state:
    st.session_state.saved_files = {}

# ---------- FILE UPLOAD ----------
uploaded_files = st.file_uploader(
    "📂 Upload Audit PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Save uploaded PDFs
if uploaded_files:

    for file in uploaded_files:

        if file.name not in st.session_state.saved_files:

            temp_file = tempfile.NamedTemporaryFile(delete=False)

            temp_file.write(file.read())

            st.session_state.saved_files[file.name] = temp_file.name

            process_pdf(file)

    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")

# ---------- CHAT INPUT ----------
question = st.chat_input("Ask something about the audit reports...")

if question:

    answer, sources = run_rag(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if msg["role"] == "assistant":

            st.write("**Sources:**")

            for i, s in enumerate(msg["sources"]):

                if st.button(
                    f"📄 Page {s['page']} – {s['file']} (Score: {s['score']})",
                    key=f"{s['file']}_{s['page']}_{i}"
                ):

                    pdf_path = st.session_state.saved_files.get(s["file"])

                    if pdf_path:

                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.write(f"Showing page {s['page']} from {s['file']}")

                        pdf_viewer(pdf_bytes)