import os
import argparse
import numpy as np
import pandas as pd
import torch
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


os.environ["HF_HUB_ENABLE_XET"] = "0"


class SemanticSearchReranker:
    def __init__(
        self,
        documents,
        bi_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_path="jobby32/ms-marco-cybersecurity-MiniLM-L6-v2",
        device=None,
        bi_encoder_batch_size=64,
        cross_encoder_max_length=256
    ):
        self.documents = [str(doc) for doc in documents]
        self.bi_encoder_name = bi_encoder_name
        self.cross_encoder_path = cross_encoder_path
        self.bi_encoder_batch_size = bi_encoder_batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.bi_encoder = SentenceTransformer(
            self.bi_encoder_name,
            device=self.device
        )

        self.cross_encoder = CrossEncoder(
            self.cross_encoder_path,
            device=self.device,
            max_length=cross_encoder_max_length
        )

        self.index = None
        self.document_embeddings = None

    def build_vector_database(self):
        document_embeddings = self.bi_encoder.encode(
            self.documents,
            batch_size=self.bi_encoder_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False
        )

        document_embeddings = document_embeddings.astype("float32")
        faiss.normalize_L2(document_embeddings)

        dim = document_embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(document_embeddings)

        self.document_embeddings = document_embeddings
        self.index = index

    def search(self, q, n=15, k=5, rerank_batch_size=64):
        if self.index is None:
            raise RuntimeError("Векторная база не построена. Сначала вызовите build_vector_database().")

        q_emb = self.bi_encoder.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")

        faiss.normalize_L2(q_emb)

        _, candidate_indices = self.index.search(q_emb, n)
        candidate_indices = candidate_indices[0]

        candidates = [
            self.documents[int(idx)]
            for idx in candidate_indices
        ]

        pairs = [
            [q, doc]
            for doc in candidates
        ]

        scores = self.cross_encoder.predict(
            pairs,
            batch_size=rerank_batch_size,
            show_progress_bar=False
        )

        scored_documents = list(zip(candidates, scores))
        scored_documents = sorted(
            scored_documents,
            key=lambda x: x[1],
            reverse=True
        )

        return scored_documents[:k]


def load_documents_from_csv(path, document_column="positive"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[document_column]).reset_index(drop=True)
    return df[document_column].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--documents_csv",
        type=str,
        default="final_full_test_df.csv"
    )

    parser.add_argument(
        "--document_column",
        type=str,
        default="positive"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True
    )

    parser.add_argument(
        "--top_n_retrieval",
        type=int,
        default=15
    )

    parser.add_argument(
        "--top_k_output",
        type=int,
        default=5
    )

    parser.add_argument(
        "--bi_encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    parser.add_argument(
        "--cross_encoder",
        type=str,
        default="jobby32/ms-marco-cybersecurity-MiniLM-L6-v2"
    )

    args = parser.parse_args()

    documents = load_documents_from_csv(
        path=args.documents_csv,
        document_column=args.document_column
    )

    searcher = SemanticSearchReranker(
        documents=documents,
        bi_encoder_name=args.bi_encoder,
        cross_encoder_path=args.cross_encoder
    )

    searcher.build_vector_database()

    results = searcher.search(
        q=args.query,
        n=args.top_n_retrieval,
        k=args.top_k_output
    )

    print("\nРезультаты поиска:\n")

    for rank, (document, score) in enumerate(results, start=1):
        print(f"{rank}. score = {float(score):.6f}")
        print(document)
        print("-" * 100)


if __name__ == "__main__":
    main()