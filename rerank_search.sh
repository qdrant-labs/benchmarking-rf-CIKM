#!/usr/bin/env bash
# Rerank-only sweep across all dataset × retriever × feedback model combinations.
# Usage:
#   bash rerank_search.sh                     # defaults: both datasets, rerank_limit=25
#   bash rerank_search.sh --rerank-limit 50
#
# Run overnight without sleep / failures:
#   nohup caffeinate -i bash rerank_search.sh > rerank_search.log 2>&1 &

set -uo pipefail

RERANK_LIMIT=25
START_FROM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rerank-limit) RERANK_LIMIT="$2"; shift 2 ;;
        --start-from)   START_FROM="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_FILE="rerank_search_limit${RERANK_LIMIT}.txt"
if (( START_FROM == 1 )); then
    > "$RESULTS_FILE"
fi

DATASETS=(scidocs fiqa)
RETRIEVERS=(jina mxbread)
FEEDBACK_MODELS=(openai-emb logprobs-reasoning)

total=$(( ${#DATASETS[@]} * ${#RETRIEVERS[@]} * ${#FEEDBACK_MODELS[@]} ))
i=0

for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        scidocs) eval_start=200 ;;
        fiqa)    eval_start=148 ;;
    esac
    for retriever in "${RETRIEVERS[@]}"; do
        case "$retriever" in
            jina)    cloud_flag="--no-cloud-inference" ;;
            mxbread) cloud_flag="--cloud-inference" ;;
        esac
        for feedback_model in "${FEEDBACK_MODELS[@]}"; do
            ((i++))

            if (( i < START_FROM )); then
                echo "Skipping run $i (--start-from $START_FROM)"
                continue
            fi

            echo ""
            echo "=== Run $i/$total: $dataset × $retriever × $feedback_model (rerank_limit=$RERANK_LIMIT) ==="

            {
                echo ""
                echo "=== Run $i/$total: $dataset × $retriever × $feedback_model (rerank_limit=$RERANK_LIMIT) ==="
                uv run python benchmark.py \
                    --dataset "$dataset" \
                    --retriever "$retriever" \
                    --feedback-model "$feedback_model" \
                    $cloud_flag \
                    --methods rerank \
                    --eval-start "$eval_start" \
                    --n-eval-queries 200 \
                    --baseline-limit 10 \
                    --rerank-limit "$RERANK_LIMIT" \
                    && echo "[OK] Run $i completed" \
                    || echo "[FAILED] Run $i exited non-zero — continuing"
            } >> "$RESULTS_FILE" 2>&1

        done
    done
done

echo ""
echo "Done. Results in $RESULTS_FILE"
