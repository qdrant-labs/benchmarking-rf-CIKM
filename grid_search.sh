#!/usr/bin/env bash
# Grid search over retrieval hyperparameters (rerank + RF context/limit combinations).
# NOTE: Currently configured for the last-run experiment set (logprobs-ordinal, rf_pairs="5 20").
# To rerun a different combination, adjust FEEDBACK_MODELS, RF_PAIRS, and RERANK_LIMITS.
# Output is written to grid_search_${DATASET}.txt — rename after the run to match the
# existing naming convention (e.g. grid_search_openai_emb_fiqa.txt).
set -euo pipefail

DATASET="${1:-scidocs}"
EVAL_START="${2:-25}"
RESULTS_FILE="grid_search_${DATASET}.txt"
> "$RESULTS_FILE"

RETRIEVERS=(mxbread jina)
FEEDBACK_MODELS=(logprobs-ordinal)
RERANK_LIMITS=(25 50)
RF_PAIRS=("5 20")

total=0
for _ in "${RETRIEVERS[@]}"; do
    for _ in "${FEEDBACK_MODELS[@]}"; do
        for _ in "${RERANK_LIMITS[@]}"; do
            for _ in "${RF_PAIRS[@]}"; do ((total++)); done
        done
    done
done

i=0
for retriever in "${RETRIEVERS[@]}"; do
    case "$retriever" in
        jina)    cloud_flag="--no-cloud-inference" ;;
        mxbread) cloud_flag="--cloud-inference" ;;
    esac
    for feedback_model in "${FEEDBACK_MODELS[@]}"; do
        for rerank_limit in "${RERANK_LIMITS[@]}"; do
            for rf_pair in "${RF_PAIRS[@]}"; do
                rf_context_limit=$(echo "$rf_pair" | cut -d' ' -f1)
                rf_limit=$(echo "$rf_pair" | cut -d' ' -f2)
                ((i++))

                echo ""
                echo "=== Experiment $i/$total: retriever=$retriever, feedback=$feedback_model, rerank=$rerank_limit, rf_context=$rf_context_limit, rf_limit=$rf_limit ==="

                {
                    echo ""
                    echo "=== Experiment $i/$total: retriever=$retriever, feedback=$feedback_model, rerank=$rerank_limit, rf_context=$rf_context_limit, rf_limit=$rf_limit ==="
                    uv run python benchmark.py \
                        --dataset "$DATASET" \
                        --retriever "$retriever" \
                        --feedback-model "$feedback_model" \
                        $cloud_flag \
                        --eval-start "$EVAL_START" \
                        --n-eval-queries 500 \
                        --baseline-limit 10 \
                        --rerank-limit "$rerank_limit" \
                        --rf-context-limit "$rf_context_limit" \
                        --rf-limit "$rf_limit"
                } >> "$RESULTS_FILE" 2>&1

            done
        done
    done
done

echo ""
echo "Done. Results saved to $RESULTS_FILE"
