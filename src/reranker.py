from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, chunks, sources):

    pairs = [[question, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, sources, scores),
        key=lambda x: x[2],
        reverse=True
    )

    ranked_chunks = [r[0] for r in ranked]
    ranked_sources = [r[1] for r in ranked]

    return ranked_chunks, ranked_sources