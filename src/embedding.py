from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def create_embeddings(texts):
    model = _get_model()
    embeddings = model.encode(texts)
    return embeddings

def create_query_embedding(query):
    model = _get_model()
    embedding = model.encode([query])
    return embedding