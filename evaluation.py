from src.rag_pipeline import run_rag, load_vector_db
import pandas as pd


dataset = [
    {
        "question": "What safety issues are mentioned?",
        "relevant_pages": [3,4]
    },
    {
        "question": "Which page mentions electrical risks?",
        "relevant_pages": [5]
    },
    {
        "question": "Where are compliance violations discussed?",
        "relevant_pages": [7]
    }
]


def calculate_metrics(k=3):

    load_vector_db()

    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for item in dataset:

        question = item["question"]
        relevant = item["relevant_pages"]

        answer, sources = run_rag(question)

        retrieved = [s["page"] for s in sources[:k]]

        correct = len(set(retrieved) & set(relevant))

        precision = correct / k
        recall = correct / len(relevant)

        precision_scores.append(precision)
        recall_scores.append(recall)

        # MRR
        rr = 0
        for i,page in enumerate(retrieved):
            if page in relevant:
                rr = 1/(i+1)
                break

        mrr_scores.append(rr)

    results = {
        "precision": sum(precision_scores)/len(precision_scores),
        "recall": sum(recall_scores)/len(recall_scores),
        "mrr": sum(mrr_scores)/len(mrr_scores)
    }

    # F1 score
    if results["precision"] + results["recall"] == 0:
        f1 = 0
    else:
        f1 = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])

    results["f1"] = f1


    return results