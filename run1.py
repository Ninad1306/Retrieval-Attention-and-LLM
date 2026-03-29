import json
from tqdm import tqdm

import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from utils import get_queries_and_items


def prepare_tools(tools_dict):
    tool_names = list(tools_dict.keys())
    tool_texts = list(tools_dict.values())

    tool_name_to_index = {
        name: idx for idx, name in enumerate(tool_names)
    }

    return tool_names, tool_texts, tool_name_to_index


class BM25Retriever:
    def __init__(self, tool_texts):
        self.tokenized_tools = [text.split() for text in tool_texts]
        self.bm25 = BM25Okapi(self.tokenized_tools)

    def score(self, query):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        return np.array(scores)

class DenseRetriever:
    def __init__(self, model_name, tool_texts):
        print(f"\nLoading model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        print("Encoding tools...")
        self.tool_embeddings = self.model.encode(
            tool_texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        )

    def score(self, query):
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )

        scores = cosine_similarity(
            query_embedding,
            self.tool_embeddings
        )

        return scores.cpu().numpy()


def compute_recall(ranked_indices, correct_index, k):
    return int(correct_index in ranked_indices[:k])


def evaluate(retriever, queries, tool_name_to_index):
    recall1 = 0
    recall5 = 0
    skipped = 0

    for q in tqdm(queries):
        query_text = q["text"]
        gold_tool = q["gold_tool_name"]

        # Handle missing tools (important edge case)
        if gold_tool not in tool_name_to_index:
            skipped += 1
            continue

        correct_index = tool_name_to_index[gold_tool]

        scores = retriever.score(query_text)
        ranked_indices = np.argsort(scores)[::-1]

        recall1 += compute_recall(ranked_indices, correct_index, 1)
        recall5 += compute_recall(ranked_indices, correct_index, 5)

    total = len(queries) - skipped

    return {
        "Recall@1": recall1 / total,
        "Recall@5": recall5 / total,
        "Skipped": skipped
    }


def main():
    _, queries, tools_dict = get_queries_and_items()

    tool_names, tool_texts, tool_name_to_index = prepare_tools(tools_dict)

    results = {}

    # -------- BM25 --------
    print("\n************* Running BM25 *************")
    bm25 = BM25Retriever(tool_texts)
    results["BM25"] = evaluate(bm25, queries, tool_name_to_index)

    print("\n************* Running MiniLM *************")
    minilm = DenseRetriever(
        "msmarco-MiniLM-L-6-v3",
        tool_texts
    )
    results["MiniLM"] = evaluate(minilm, queries, tool_name_to_index)

    print("\n************* Running UAE-large *************")
    uae = DenseRetriever(
        "WhereIsAI/UAE-Large-V1",
        tool_texts
    )
    results["UAE-large-v1"] = evaluate(uae, queries, tool_name_to_index)

    print("\n===== FINAL RESULTS =====")
    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  Recall@1: {metrics['Recall@1']:.4f}")
        print(f"  Recall@5: {metrics['Recall@5']:.4f}")
        print(f"  Skipped: {metrics['Skipped']}")
        print()


if __name__ == "__main__":
    main()