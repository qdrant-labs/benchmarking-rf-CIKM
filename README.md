# Relevance Feedback Benchmarking

Benchmarking Qdrant's [Relevance Feedback](https://qdrant.tech/documentation/concepts/explore/#relevance-feedback) API on BEIR datasets.

> RF is not text-specific — the same approach applies to image or multimodal data. BEIR is used here for convenience.

All motivation & details in JOURNAL.md

Requires `.env` with `QDRANT_URL`, `QDRANT_API_KEY`, and `OPENAI_API_KEY`.

## Pipeline

```
1. load_beir.py          — download & register a BEIR dataset
2. indexing_to_qdrant.py — embed corpus and upload to Qdrant collection
3. train_rf.py           — learn RF formula params on first N queries
4. benchmark.py          — evaluate all methods on the remaining queries
```

Grid search over retrieval hyperparameters: `grid_search.sh [dataset]`

## Config files

| File | Purpose |
|------|---------|
| `beir_data_path.json` | local paths to downloaded datasets |
| `formula_params.json` | trained RF params, keyed by `{dataset}-{retriever}-{feedback}` |
| `feedback_prompts.json` | dataset-specific prompts for logprobs models |
| `cost_per_call.json` | estimated cost per (query, document) scoring call |

## Datasets & retrievers

Datasets: **SciDocs** (citation prediction, 25k corpus) · **FiQA** (financial QA, 57k corpus)
Retrievers: `jina` (`jina-embeddings-v2-base-en`, 768d) · `mxbread` (`mxbai-embed-large-v1`, 1024d)

## Metrics

measured with `ranx`
- **Recall@[1, 3, 5, 10]** & **nDCG@10**.
and
- **disc@N** — custom discovery metric. Counts how many documents RF-based methods ranked in the top-N that were *unreachable* by the reranker (i.e., outside its initial fetch window).

## Usage

```bash
# 1. Download dataset
uv run python load_beir.py --dataset fiqa

# 2. Index corpus
# mxbread uses Qdrant cloud inference by default; jina embeds locally (--no-cloud-inference)
uv run python indexing_to_qdrant.py --dataset fiqa --model-name mxbread
uv run python indexing_to_qdrant.py --dataset fiqa --model-name jina --no-cloud-inference

# 3. Train RF params
# --n-train-queries: how many leading queries to use for training (the rest go to eval)
# FiQA has 648 test queries total → use 148 for train, 500 for eval
# SciDocs has 1000 test queries total → use 200 for train, 500 for eval
uv run python train_rf.py --dataset fiqa --retriever mxbread --feedback-model openai-emb --n-train-queries 148
uv run python train_rf.py --dataset fiqa --retriever jina   --feedback-model openai-emb --n-train-queries 148 --no-cloud-inference

# 4. Benchmark
# --eval-start must equal --n-train-queries used in step 3
# --methods controls which methods to run alongside baseline (default: all three)
#   choices: rerank  rf  pure-rf
uv run python benchmark.py --dataset fiqa --retriever mxbread --feedback-model openai-emb --eval-start 148 --n-eval-queries 500
uv run python benchmark.py --dataset fiqa --retriever jina   --feedback-model openai-emb --eval-start 148 --n-eval-queries 500 --no-cloud-inference

# Baseline + rerank only (skip RF methods):
uv run python benchmark.py --dataset fiqa --retriever mxbread --feedback-model openai-emb --eval-start 148 --n-eval-queries 500 --methods rerank

# Rerank-only sweep across all dataset × retriever × feedback model combinations
bash rerank_search.sh

# Full grid search (rerank + RF hyperparameters)
# Second argument is eval-start (= --n-train-queries used in step 3)
bash grid_search.sh fiqa 148
```

### benchmark.py flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dataset` | | `scidocs` | BEIR dataset name — must be registered in `beir_data_path.json` (step 1) |
| `--retriever` | ✓ | | `jina` or `mxbread` |
| `--feedback-model` | ✓ | | `openai-emb`, `logprobs`, `logprobs-ordinal`, `logprobs-reasoning` |
| `--eval-start` | ✓ | | First query index to evaluate. **Must equal `--n-train-queries` used in step 3** to avoid evaluating on training queries |
| `--n-eval-queries` | ✓ | | How many queries to evaluate (from `--eval-start` onward) |
| `--methods` | | all | Which methods to run alongside baseline: `rerank`, `rf`, `pure-rf` (space-separated) |
| `--baseline-limit` | | `10` | Docs retrieved by ANN for baseline. Also the evaluation cutoff — Recall@N and nDCG@10 are computed over this top-N |
| `--rerank-limit` | | `25` | Pool size for reranking: ANN fetches this many docs, all rescored by the feedback model, top-`--baseline-limit` kept |
| `--rf-context-limit` | | `5` | Docs scored by the feedback model to form the RF signal (positive/negative examples passed to Qdrant RF API) |
| `--rf-limit` | | `20` | Extra docs fetched by the RF query. For `rf` (RF+Rerank): these are rescored and merged with the context docs. For `pure-rf`: these are returned directly by Qdrant's RF-modified HNSW traversal |
| `--cloud-inference` / `--no-cloud-inference` | | cloud on | Whether to embed queries via Qdrant cloud inference (`mxbread`) or locally via fastembed (`jina`) |

### Adding a new BEIR dataset

1. Run steps 1–4 above with `--dataset <name>`.
2. **`openai-emb`** works out of the box for any dataset.
3. **Logprobs models** (`logprobs`, `logprobs-ordinal`, `logprobs-reasoning`) need dataset-specific prompts — add an entry to `feedback_prompts.json` under each model key. The script will raise a clear error if the entry is missing.
4. **Cost estimates** in the benchmark output table are best-effort. Add an entry to `cost_per_call.json` for accurate numbers; if missing, cost is shown as `—`.

## Testing

```bash
# Sanity-check all feedback models against known query/document pairs
uv run python test_feedback_models.py
```
