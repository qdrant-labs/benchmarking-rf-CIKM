"""
Train Relevance Feedback formula parameters on the first N queries of a BEIR dataset.

Reads beir_data_path.json (written by load_beir.py).
Saves trained params to formula_params.json under key '{dataset}-{retriever}-{feedback}'.
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

from beir.datasets.data_loader import GenericDataLoader
from qdrant_client import QdrantClient
from qdrant_relevance_feedback import RelevanceFeedback
from qdrant_relevance_feedback.retriever import QdrantRetriever

from logprobs_feedback import LogprobsFeedback
from logprobs_ordinal_feedback import LogprobsOrdinalFeedback
from logprobs_reasoning_feedback import LogprobsReasoningFeedback
from openai_feedback import OpenAIFeedback

MAX_RETRIES = 3
RETRY_DELAY = 2.0

RETRIEVERS: dict[str, tuple[str, str]] = {
    "jina":    ("jinaai/jina-embeddings-v2-base-en",   "jina"),
    "mxbread": ("mixedbread-ai/mxbai-embed-large-v1", "mxbread"),
}

FEEDBACK_MODELS = {
    "logprobs":           LogprobsFeedback,
    "logprobs-ordinal":   LogprobsOrdinalFeedback,
    "logprobs-reasoning": LogprobsReasoningFeedback,
    "openai-emb":         OpenAIFeedback,
}

PAYLOAD_KEY = "document"
TRAIN_LIMIT = 25  # documents retrieved per training query
FORMULA_PARAMS_FILE = "formula_params.json"
FEEDBACK_PROMPTS_FILE = "feedback_prompts.json"
COST_PER_CALL_FILE = "cost_per_call.json"


class FaultTolerantRelevanceFeedback(RelevanceFeedback):
    def prepare_train_data_query(self, query_idx, query, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return super().prepare_train_data_query(query_idx, query, **kwargs)
            except (LogprobsFeedback.ModelOutputError, LogprobsOrdinalFeedback.ModelOutputError, LogprobsReasoningFeedback.ModelOutputError) as e:
                print(f"  [skip] query {query_idx} — model output error: {e}")
                return []
            except Exception as e:
                print(f"  [retry {attempt}/{MAX_RETRIES}] query {query_idx} — {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        print(f"  [skip] query {query_idx} — all {MAX_RETRIES} retries failed")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Train RF formula params on first N queries of a BEIR dataset"
    )
    parser.add_argument(
        "--n-train-queries",
        type=int,
        required=True,
        help="Number of leading queries to use for training (e.g. 100)",
    )
    parser.add_argument(
        "--dataset",
        default="scidocs",
        help="BEIR dataset name, must be present in beir_data_path.json (default: scidocs)",
    )
    parser.add_argument(
        "--retriever",
        required=True,
        choices=list(RETRIEVERS.keys()),
        help="Embedding model to use for retrieval",
    )
    parser.add_argument(
        "--feedback-model",
        required=True,
        choices=list(FEEDBACK_MODELS.keys()),
        help="Feedback model to use for scoring relevance",
    )
    parser.add_argument(
        "--cloud-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qdrant cloud inference for embeddings (default: true). Pass --no-cloud-inference to embed locally.",
    )
    args = parser.parse_args()

    model_id, vector_name = RETRIEVERS[args.retriever]
    collection_name = f"{args.dataset}-relevance-feedback-{args.retriever}-docs"
    params_key = f"{args.dataset}-{args.retriever}-{args.feedback_model}"

    with open("beir_data_path.json") as f:
        data_path = json.load(f)[args.dataset]

    _corpus, queries, _qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    query_ids = list(queries.keys())
    train_query_ids = query_ids[: args.n_train_queries]
    print(f"Training on {len(train_query_ids)} '{args.dataset}' queries (indices 0–{len(train_query_ids) - 1})")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=args.cloud_inference)

    print("Initializing retriever and feedback model...")
    retriever = QdrantRetriever(
        model_id,
        modality="text",
        embed_options={"lazy_load": True},
    )
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if args.feedback_model in ("logprobs", "logprobs-ordinal", "logprobs-reasoning"):
        with open(FEEDBACK_PROMPTS_FILE) as f:
            prompts = json.load(f)
        model_prompts = prompts.get(args.feedback_model, {})
        if args.dataset not in model_prompts:
            raise ValueError(f"No prompts defined for '{args.feedback_model}/{args.dataset}' in {FEEDBACK_PROMPTS_FILE}")
        feedback_model = FEEDBACK_MODELS[args.feedback_model](api_key=openai_api_key, **model_prompts[args.dataset])
    else:
        feedback_model = FEEDBACK_MODELS[args.feedback_model](api_key=openai_api_key)

    rf_framework = FaultTolerantRelevanceFeedback(
        retriever=retriever,
        feedback=feedback_model,
        client=client,
        collection_name=collection_name,
        vector_name=vector_name,
        payload_key=PAYLOAD_KEY,
    )

    with open(COST_PER_CALL_FILE) as f:
        cost_data = json.load(f)
    cost_per_call = cost_data.get(args.feedback_model, {}).get(args.dataset)
    if cost_per_call is not None:
        est_cost = len(train_query_ids) * TRAIN_LIMIT * cost_per_call
        print(f"Estimated training cost: ${est_cost:.4f} ({len(train_query_ids)} queries × {TRAIN_LIMIT} docs × ${cost_per_call:.7f}/call)")
    else:
        print(f"Estimated training cost: unknown (no entry in {COST_PER_CALL_FILE} for '{args.feedback_model}/{args.dataset}')")

    print("Training RF formula...")
    train_texts = [queries[qid] for qid in train_query_ids]
    formula_params = rf_framework.train(queries=train_texts, limit=TRAIN_LIMIT)

    registry = {}
    if os.path.exists(FORMULA_PARAMS_FILE):
        with open(FORMULA_PARAMS_FILE) as f:
            registry = json.load(f)
    registry[params_key] = formula_params
    with open(FORMULA_PARAMS_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Formula params saved to {FORMULA_PARAMS_FILE} under key '{params_key}': {formula_params}")


if __name__ == "__main__":
    main()
