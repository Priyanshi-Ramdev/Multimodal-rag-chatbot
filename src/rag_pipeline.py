import faiss
import os
import pickle
import numpy as np
from pypdf import PdfReader

from src.reranker import rerank
from src.embedding import create_embeddings, create_query_embedding
from src.llm import ask_llm
from src.chunker import chunk_text


INDEX_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/chunks.pkl"

all_chunks = []
chunk_sources = []
index = None


def process_pdf(uploaded_file):

    global all_chunks, chunk_sources, index

    reader = PdfReader(uploaded_file)

    new_chunks = []
    new_sources = []

    for page_num, page in enumerate(reader.pages):

        extracted = page.extract_text()

        if extracted:

            page_chunks = chunk_text(extracted)

            for chunk in page_chunks:

                new_chunks.append(chunk)

                new_sources.append({
                    "file": uploaded_file.name,
                    "page": page_num + 1
                })

    embeddings = create_embeddings(new_chunks)

    if index is None:

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    all_chunks.extend(new_chunks)
    chunk_sources.extend(new_sources)

    save_vector_db()


def run_rag(question):

    global index

    query_embedding = create_query_embedding(question)

    distances, indices = index.search(np.array(query_embedding), k=10)

    retrieved_chunks = []
    sources = []
    scores = []

    for idx, dist in zip(indices[0], distances[0]):

        retrieved_chunks.append(all_chunks[idx])

        source = chunk_sources[idx].copy()
        source["score"] = round(float(dist), 3)

        sources.append(source)

    # rerank
    ranked_chunks, ranked_sources = rerank(question, retrieved_chunks, sources)

    best_chunks = ranked_chunks[:3]
    best_sources = ranked_sources[:3]

    context = "\n".join(best_chunks)

    answer = ask_llm(context, question)

    return answer, best_sources


def save_vector_db():

    global index, all_chunks, chunk_sources

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump((all_chunks, chunk_sources), f)


def load_vector_db():

    global index, all_chunks, chunk_sources

    if os.path.exists(INDEX_PATH):

        index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            all_chunks, chunk_sources = pickle.load(f)

        return True

    return False