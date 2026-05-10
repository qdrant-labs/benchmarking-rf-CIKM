"""
Benchmark retrieval methods on a BEIR dataset, evaluated with Recall@N & NDCG@10 + custom disc@N.
https://castorini.github.io/pyserini/2cr/beir.html for reference

Reads beir_data_path.json (written by load_beir.py).
Reads formula_params.json (written by train_rf.py).

Methods
-------
1. Baseline  : retrieve --baseline-limit docs with the retriever model
2. rerank    : retrieve --rerank-limit docs, rescore with Feedback model
3. rf        : retrieve --rf-context-limit docs, rescore with Feedback model, run RF query
               (--rf-limit docs), rescore RF results with Feedback model, merge by score
4. pure-rf   : retrieve --rf-context-limit docs, rescore with Feedback model, run RF query
               (--rf-limit docs)
"""

import argparse
import json
import os
import time
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

import tqdm
from beir.datasets.data_loader import GenericDataLoader
from qdrant_client import QdrantClient, models
from qdrant_relevance_feedback.retriever import QdrantRetriever
from ranx import Qrels, Run, evaluate

from logprobs_feedback import LogprobsFeedback
from logprobs_ordinal_feedback import LogprobsOrdinalFeedback
from logprobs_reasoning_feedback import LogprobsReasoningFeedback
from openai_feedback import OpenAIFeedback
from qdrant_relevance_feedback.feedback import Feedback

PAYLOAD_KEY = "document"

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries

ALL_METHODS = ["rerank", "rf", "pure-rf"]

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

FORMULA_PARAMS_FILE = "formula_params.json"
COST_PER_CALL_FILE = "cost_per_call.json"
FEEDBACK_PROMPTS_FILE = "feedback_prompts.json"


def with_retries(fn: Callable, qid: str) -> tuple[dict | None, float]:
    """Call fn(), retrying up to MAX_RETRIES times on any exception.

    Returns (result, elapsed_seconds), or (None, 0.0) if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            result = fn()
            return result, time.perf_counter() - t0
        except (LogprobsFeedback.ModelOutputError, LogprobsOrdinalFeedback.ModelOutputError, LogprobsReasoningFeedback.ModelOutputError) as e:
            tqdm.tqdm.write(f"  [skip] qid={qid} — model output error: {e}")
            return None, 0.0
        except Exception as e:
            tqdm.tqdm.write(
                f"  [retry {attempt}/{MAX_RETRIES}] qid={qid} — {type(e).__name__}: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    tqdm.tqdm.write(f"  [skip] qid={qid} — all {MAX_RETRIES} retries failed")
    return None, 0.0


def baseline(
    client: QdrantClient, query_embedding, k: int, collection_name: str, vector_name: str
) -> dict:
    points = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=k,
        using=vector_name,
    ).points
    return {p.payload["document_id"]: p.score for p in points}


def rerank(
    client: QdrantClient,
    feedback_model: Feedback,
    query_text: str,
    query_embedding,
    rerank_limit: int,
    collection_name: str,
    vector_name: str,
) -> dict:
    """Retrieve rerank_limit docs, rescore all with feedback model, return {doc_id: score}."""
    points = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=rerank_limit,
        using=vector_name,
    ).points

    texts = [p.payload[PAYLOAD_KEY] for p in points]
    rescored = feedback_model.score(query_text, texts)
    return {
        p.payload["document_id"]: s
        for p, s in sorted(zip(points, rescored), key=lambda x: x[1], reverse=True)
    }


def relevance_feedback_retrieval(
    client: QdrantClient,
    feedback_model: Feedback,
    query_text: str,
    query_embedding,
    formula_params: dict,
    rf_context_limit: int,
    rf_limit: int,
    collection_name: str,
    vector_name: str,
) -> dict:
    """
    1. Retrieve rf_context_limit docs.
    2. Rescore with feedback model.
    3. Run RF query (limit=rf_limit) using feedback scores.
    4. Rescore RF results with feedback model.
    5. Merge all docs by feedback score, return as {doc_id: score}.
    """
    initial_points = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=rf_context_limit,
        using=vector_name,
    ).points

    initial_texts = [p.payload[PAYLOAD_KEY] for p in initial_points]
    fb_scores = feedback_model.score(query_text, initial_texts)

    # Point IDs used as examples → Qdrant excludes them from RF results automatically.
    rf_points = client.query_points(
        collection_name=collection_name,
        query=models.RelevanceFeedbackQuery(
            relevance_feedback=models.RelevanceFeedbackInput(
                target=query_embedding,
                feedback=[
                    models.FeedbackItem(example=p.id, score=score)
                    for p, score in zip(initial_points, fb_scores)
                ],
                strategy=models.NaiveFeedbackStrategy(
                    naive=models.NaiveFeedbackStrategyParams(**formula_params)
                ),
            )
        ),
        with_payload=True,
        limit=rf_limit,
        using=vector_name,
    ).points

    rf_texts = [p.payload[PAYLOAD_KEY] for p in rf_points]
    rf_rescored = feedback_model.score(query_text, rf_texts)

    all_pairs = list(zip(initial_points, fb_scores)) + list(zip(rf_points, rf_rescored))
    all_pairs.sort(key=lambda x: x[1], reverse=True)
    return {p.payload["document_id"]: s for p, s in all_pairs}


def pure_rf_retrieval(
    client: QdrantClient,
    feedback_model: Feedback,
    query_text: str,
    query_embedding,
    formula_params: dict,
    rf_context_limit: int,
    k: int,
    collection_name: str,
    vector_name: str,
) -> dict:
    """
    1. Retrieve rf_context_limit docs.
    2. Rescore with feedback model.
    3. Run RF query using feedback scores and return {doc_id: qdrant_score}.
    """
    initial_points = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        with_vectors=True,
        limit=rf_context_limit,
        using=vector_name,
    ).points

    initial_texts = [p.payload[PAYLOAD_KEY] for p in initial_points]
    fb_scores = feedback_model.score(query_text, initial_texts)

    rf_points = client.query_points(
        collection_name=collection_name,
        query=models.RelevanceFeedbackQuery(
            relevance_feedback=models.RelevanceFeedbackInput(
                target=query_embedding,
                feedback=[
                    models.FeedbackItem(example=p.vector[vector_name], score=score)
                    for p, score in zip(initial_points, fb_scores)
                ],
                strategy=models.NaiveFeedbackStrategy(
                    naive=models.NaiveFeedbackStrategyParams(**formula_params)
                ),
            )
        ),
        with_payload=True,
        limit=k,
        using=vector_name,
    ).points

    return {p.payload["document_id"]: p.score for p in rf_points}


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on a BEIR dataset")
    parser.add_argument(
        "--eval-start",
        type=int,
        required=True,
        help="Starting query index (inclusive); should equal --n-train-queries from train_rf.py",
    )
    parser.add_argument(
        "--n-eval-queries",
        type=int,
        required=True,
        help="Number of test queries to evaluate (starting from --eval-start)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        help="Methods to benchmark alongside baseline (default: all). "
             "Choices: rerank rf pure-rf",
    )
    parser.add_argument(
        "--baseline-limit",
        type=int,
        default=10,
        help="Number of docs retrieved for baseline (default: 10)",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=25,
        help="Number of docs retrieved for reranking (default: 25)",
    )
    parser.add_argument(
        "--rf-context-limit",
        type=int,
        default=5,
        help="Number of docs in the initial RF retrieval window (default: 5)",
    )
    parser.add_argument(
        "--rf-limit",
        type=int,
        default=20,
        help="Number of new docs additionally retrieved by the RF (default: 20)",
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
        help="Embedding model used for retrieval",
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

    selected_methods = set(args.methods)
    needs_feedback = bool(selected_methods)
    needs_rf = selected_methods & {"rf", "pure-rf"}

    with open("beir_data_path.json") as f:
        data_path = json.load(f)[args.dataset]

    formula_params = {}
    if needs_rf:
        with open(FORMULA_PARAMS_FILE) as f:
            formula_params = json.load(f)[params_key]
        print(f"Loaded formula params for '{params_key}': {formula_params}")

    _corpus, queries, qrels_dict = GenericDataLoader(data_folder=data_path).load(split="test")

    query_ids = list(queries.keys())
    eval_end = args.eval_start + args.n_eval_queries
    eval_query_ids = [
        qid for qid in query_ids[args.eval_start : eval_end]
        if qid in qrels_dict
    ]
    print(
        f"Evaluating indices {args.eval_start}–{eval_end - 1} "
        f"({len(eval_query_ids)} queries with qrels)"
    )

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=args.cloud_inference)

    print("Initializing retriever...")
    retriever = QdrantRetriever(
        model_id,
        modality="text",
        embed_options={"lazy_load": True},
    )

    feedback_model = None
    if needs_feedback:
        print("Initializing feedback model...")
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

    runs: dict[str, dict] = {"baseline": {}, **{m: {} for m in args.methods}}
    lats: dict[str, list[float]] = {"baseline": [], **{m: [] for m in args.methods}}
    skipped: set[str] = set()

    DISC_CUTOFFS = [1, 3, 5, 10]
    rf_methods = [m for m in ("rf", "pure-rf") if m in selected_methods]
    disc_counts: dict[str, dict[int, int]] = {m: {k: 0 for k in DISC_CUTOFFS} for m in rf_methods}

    print(f"\nRunning benchmark (methods: baseline + {', '.join(args.methods) or 'none'})...")
    for qid in tqdm.tqdm(eval_query_ids):
        tmp: dict[str, tuple[dict, float]] = {}

        query_text = queries[qid]
        query_embedding = retriever.embed_query(query_text)

        res, lat = with_retries(
            lambda: baseline(client, query_embedding, args.baseline_limit, collection_name, vector_name),
            qid,
        )
        if res is None:
            skipped.add(qid)
            continue
        tmp["baseline"] = (res, lat)

        # Build per-query method list in canonical order
        active_methods: list[tuple[str, Callable]] = []
        if "rerank" in selected_methods:
            active_methods.append(("rerank", lambda: rerank(
                client, feedback_model, query_text, query_embedding, args.rerank_limit,
                collection_name, vector_name,
            )))
        if "rf" in selected_methods:
            active_methods.append(("rf", lambda: relevance_feedback_retrieval(
                client, feedback_model, query_text, query_embedding,
                formula_params, args.rf_context_limit, args.rf_limit,
                collection_name, vector_name,
            )))
        if "pure-rf" in selected_methods:
            active_methods.append(("pure-rf", lambda: pure_rf_retrieval(
                client, feedback_model, query_text, query_embedding,
                formula_params, args.rf_context_limit, args.rf_limit,
                collection_name, vector_name,
            )))

        for name, fn in active_methods:
            res, lat = with_retries(fn, qid)
            if res is None:
                skipped.add(qid)
                break
            tmp[name] = (res, lat)
        else:
            # All methods succeeded — commit atomically
            for name, (res, lat) in tmp.items():
                runs[name][qid] = res
                lats[name].append(lat)

            if needs_rf:
                # Assumes ANN is deterministic for the same query embedding (holds for Qdrant HNSW in practice).
                rerank_pool_ids = {
                    p.payload["document_id"]
                    for p in client.query_points(
                        collection_name=collection_name,
                        query=query_embedding,
                        limit=args.rerank_limit,
                        using=vector_name,
                    ).points
                }
                unreachable_relevant = set(qrels_dict[qid].keys()) - rerank_pool_ids
                for m in rf_methods:
                    top10 = list(runs[m][qid].keys())[:10]
                    mask = [1 if doc_id in unreachable_relevant else 0 for doc_id in top10]
                    disc_counts[m][1]  += mask[0] if mask else 0
                    disc_counts[m][3]  += sum(mask[:3])
                    disc_counts[m][5]  += sum(mask[:5])
                    disc_counts[m][10] += sum(mask)

    if skipped:
        print(f"\nSkipped {len(skipped)} queries: {skipped}")

    evaluated_qids = [qid for qid in eval_query_ids if qid not in skipped]
    n_eval = len(evaluated_qids)

    RECALL_CUTOFFS = [1, 3, 5, 10]
    recall_metrics = [f"recall@{k}" for k in RECALL_CUTOFFS]
    metrics = recall_metrics + ["ndcg@10"]

    qrels_ranx = Qrels({qid: qrels_dict[qid] for qid in evaluated_qids})

    with open(COST_PER_CALL_FILE) as f:
        cost_per_call = json.load(f).get(args.feedback_model, {}).get(args.dataset, 0.0)

    def est_cost(calls_per_query: int) -> float:
        return n_eval * calls_per_query * cost_per_call

    method_configs = [
        ("baseline", f"1. Baseline  (retrieve {args.baseline_limit})",                                        0.0),
        ("rerank",   f"2. Rerank  (retrieve {args.rerank_limit} → Feedback Model rerank)",                    est_cost(args.rerank_limit)),
        ("rf",       f"3. RF+Rerank  ({args.rf_context_limit} fb → {args.rf_limit} RF → merge by rerank)",   est_cost(args.rf_context_limit + args.rf_limit)),
        ("pure-rf",  f"4. Pure RF  ({args.rf_context_limit} context → {args.rf_limit} RF docs)",       est_cost(args.rf_context_limit)),
    ]

    rows = []
    for name, label, cost in method_configs:
        if name != "baseline" and name not in selected_methods:
            continue
        scores = evaluate(qrels_ranx, Run(runs[name]), metrics)
        avg_lat = sum(lats[name]) / len(lats[name]) * 1000
        rows.append((label, scores, avg_lat, cost))

    col = 9
    lat_col = 12
    cost_col = 11
    method_col = max(len(label) for label, *_ in rows) + 2
    W = 2 + method_col + col * (len(RECALL_CUTOFFS) + 1) + lat_col + cost_col + 2

    recall_header = "".join(f"{'R@' + str(k):>{col}}" for k in RECALL_CUTOFFS)
    header = recall_header + f"{'nDCG@10':>{col}}"
    subtitle = f"BEIR {args.dataset}  |  {args.retriever}  |  {args.feedback_model}  |  Recall@[1,3,5,10] + nDCG@10  ({n_eval} queries)"
    print()
    print("=" * W)
    print(f"{'BENCHMARK RESULTS':^{W}}")
    print(f"{subtitle:^{W}}")
    print("=" * W)
    print(f"  {'Method':<{method_col}}{header} {'Latency':>{lat_col}} {'Est. cost':>{cost_col}}")
    print("-" * W)
    for label, scores, lat, cost in rows:
        row = "".join(f"{scores[m]:>{col}.4f}" for m in metrics)
        cost_str = f"${cost:.4f}" if cost > 0 else "—"
        print(f"  {label:<{method_col}}{row} {lat:>{lat_col - 3}.1f} ms {cost_str:>{cost_col}}")
    print("=" * W)
    print()

    if rf_methods:
        disc_labels = {
            "rf":      f"3. RF+Rerank  ({args.rf_context_limit} fb → {args.rf_limit} RF → merge by rerank)",
            "pure-rf": f"4. Pure RF  ({args.rf_context_limit} context → {args.rf_limit} RF docs)",
        }
        disc_col = 9
        disc_method_col = max(len(disc_labels[m]) for m in rf_methods) + 2
        DW = 2 + disc_method_col + disc_col * len(DISC_CUTOFFS) + 2
        disc_header = "".join(f"{'disc@' + str(k):>{disc_col}}" for k in DISC_CUTOFFS)
        disc_subtitle = f"RF DISCOVERY  |  rerank pool = {args.rerank_limit} docs  |  cumulative counts over {n_eval} queries"
        print("=" * DW)
        print(f"{disc_subtitle:^{DW}}")
        print("=" * DW)
        print(f"  {'Method':<{disc_method_col}}{disc_header}")
        print("-" * DW)
        for m in rf_methods:
            row = "".join(f"{disc_counts[m][k]:>{disc_col}}" for k in DISC_CUTOFFS)
            print(f"  {disc_labels[m]:<{disc_method_col}}{row}")
        print("=" * DW)
        print()


if __name__ == "__main__":
    main()
