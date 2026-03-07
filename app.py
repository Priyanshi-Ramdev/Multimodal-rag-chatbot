import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile

from src.rag_pipeline import process_pdf, run_rag, load_vector_db

st.set_page_config(page_title="AuditMind AI", page_icon="🤖")

st.title("🤖 AuditMind AI")
st.write("Upload audit reports and ask questions.")

# Load vector database
load_vector_db()

# session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_files" not in st.session_state:
    st.session_state.saved_files = {}

uploaded_files = st.file_uploader(
    "Upload Audit PDFs",
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

    st.success(f"{len(uploaded_files)} PDFs uploaded successfully!")

# Chat input
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

# Display chat
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if msg["role"] == "assistant":

            st.write("**Sources:**")

            for i, s in enumerate(msg["sources"]):

                if st.button(
                    f"Page {s['page']} – {s['file']} (Score: {s['score']})",
                    key=f"{s['file']}_{s['page']}_{i}"
                ):

                    pdf_path = st.session_state.saved_files.get(s["file"])

                    if pdf_path:
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.write(f"Showing page {s['page']} from {s['file']}")

                        pdf_viewer(pdf_bytes)