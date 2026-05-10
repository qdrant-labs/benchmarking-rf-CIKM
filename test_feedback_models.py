"""
Sanity checks for feedback models against known query/qrel pairs.

Hardcoded pairs (fast, no dataset loading):
  SciDocs: paper citation prediction pair
  FiQA:    financial Q&A pair

Real qrel test (loads actual BEIR data):
  Picks query at index REAL_QREL_QUERY_INDEX, finds the highest-scored relevant
  doc from qrels and a random non-relevant doc; all models must score relevant > non-relevant.

Checks per model:
  - score() returns a list of the right length
  - all scores are finite floats
  - LogprobsFeedback / LogprobsReasoningFeedback: scores are in [0, 1]
  - relevant doc scores strictly higher than non-relevant doc
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

from beir.datasets.data_loader import GenericDataLoader

from openai_feedback import OpenAIFeedback
from logprobs_feedback import LogprobsFeedback
from logprobs_ordinal_feedback import LogprobsOrdinalFeedback
from logprobs_reasoning_feedback import LogprobsReasoningFeedback

REAL_QREL_QUERY_INDEX = 201  # outside training range for both datasets (200 / 148)

FEEDBACK_PROMPTS_FILE = "feedback_prompts.json"

# --- SciDocs ---

SCIDOCS_QUERY = "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect"

SCIDOCS_RELEVANT_DOC = (
    "Dynamic economic dispatch (DED) is one of the main functions of power generation "
    "operation and control. It determines the optimal settings of generator units with "
    "predicted load demand over a certain period of time. The objective is to operate an "
    "electric power system most economically while the system is operating within its "
    "security limits. This paper proposes a new hybrid methodology for solving DED. The "
    "proposed method is developed in such a way that a simple evolutionary programming (EP) "
    "is applied as a based level search, which can give a good direction to the optimal "
    "global region, and a local search sequential quadratic programming (SQP) is used as a "
    "fine tuning to determine the optimal solution at the final. Ten units test system with "
    "nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed "
    "method compared with those obtained from EP and SQP alone."
)

SCIDOCS_NON_RELEVANT_DOC = (
    "This report summarizes the objectives and evaluation of the SemEval 2015 task on the "
    "sentiment analysis of figurative language on Twitter (Task 11). This is the first "
    "sentiment analysis task wholly dedicated to analyzing figurative language on Twitter. "
    "Specifically, three broad classes of figurative language are considered: irony, sarcasm "
    "and metaphor. Gold standard sets of 8000 training tweets and 4000 test tweets were "
    "annotated using workers on the crowdsourcing platform CrowdFlower. Participating systems "
    "were required to provide a fine-grained sentiment score on an 11-point scale (-5 to +5, "
    "including 0 for neutral intent) for each tweet, and systems were evaluated against the "
    "gold standard using both a Cosinesimilarity and a Mean-Squared-Error measure."
)

# --- FiQA ---

FIQA_QUERY = "What does it mean to short a stock?"

FIQA_RELEVANT_DOC = (
    "Shorting a stock means borrowing shares from a broker and selling them immediately on "
    "the open market, hoping to buy them back later at a lower price. The short seller profits "
    "from the difference if the stock price falls. However, because a stock can theoretically "
    "rise without limit, the potential loss on a short position is unlimited, while the maximum "
    "gain is capped at 100% if the stock goes to zero. Brokers typically require a margin "
    "account and charge interest on the borrowed shares."
)

FIQA_NON_RELEVANT_DOC = (
    "A dividend reinvestment plan (DRIP) allows shareholders to automatically reinvest cash "
    "dividends into additional shares of the same stock, often at a slight discount to market "
    "price and without paying brokerage commissions. DRIPs are a popular strategy for "
    "long-term investors who want to compound returns over time without actively managing "
    "their portfolio. Many large companies offer DRIPs directly through their transfer agents."
)


def check(model_name: str, scores: list[float], is_probability: bool) -> None:
    rel_score, nrel_score = scores[0], scores[1]

    print(f"  relevant score:     {rel_score:.6f}")
    print(f"  non-relevant score: {nrel_score:.6f}")

    assert len(scores) == 2, f"expected 2 scores, got {len(scores)}"

    for i, s in enumerate(scores):
        assert isinstance(s, float), f"score[{i}] is not float: {type(s)}"
        assert s == s, f"score[{i}] is NaN"
        assert s not in (float("inf"), float("-inf")), f"score[{i}] is infinite"

    if is_probability:
        for i, s in enumerate(scores):
            assert 0.0 <= s <= 1.0, f"score[{i}]={s:.6f} outside [0, 1]"

    assert rel_score > nrel_score, (
        f"{model_name}: relevant score ({rel_score:.6f}) should be "
        f"> non-relevant score ({nrel_score:.6f})"
    )

    print(f"  PASSED")


def test_openai_feedback(dataset: str = "scidocs") -> None:
    print(f"OpenAIFeedback [{dataset}]")
    api_key = os.environ["OPENAI_API_KEY"]
    model = OpenAIFeedback(api_key=api_key)
    query, relevant, non_relevant = _get_pair(dataset)
    scores = model.score(query, [relevant, non_relevant])
    check("OpenAIFeedback", scores, is_probability=False)


def test_logprobs_feedback(dataset: str = "scidocs") -> None:
    print(f"LogprobsFeedback [{dataset}]")
    api_key = os.environ["OPENAI_API_KEY"]
    with open(FEEDBACK_PROMPTS_FILE) as f:
        prompts = json.load(f)["logprobs"][dataset]
    model = LogprobsFeedback(api_key=api_key, **prompts)
    query, relevant, non_relevant = _get_pair(dataset)
    scores = model.score(query, [relevant, non_relevant])
    check("LogprobsFeedback", scores, is_probability=True)


def test_logprobs_ordinal_feedback(dataset: str = "scidocs") -> None:
    print(f"LogprobsOrdinalFeedback [{dataset}]")
    api_key = os.environ["OPENAI_API_KEY"]
    with open(FEEDBACK_PROMPTS_FILE) as f:
        prompts = json.load(f)["logprobs-ordinal"][dataset]
    model = LogprobsOrdinalFeedback(api_key=api_key, **prompts)
    query, relevant, non_relevant = _get_pair(dataset)
    scores = model.score(query, [relevant, non_relevant])
    check("LogprobsOrdinalFeedback", scores, is_probability=False)


def test_logprobs_reasoning_feedback(dataset: str = "scidocs") -> None:
    print(f"LogprobsReasoningFeedback [{dataset}]")
    api_key = os.environ["OPENAI_API_KEY"]
    with open(FEEDBACK_PROMPTS_FILE) as f:
        prompts = json.load(f)["logprobs-reasoning"][dataset]
    model = LogprobsReasoningFeedback(api_key=api_key, **prompts)
    query, relevant, non_relevant = _get_pair(dataset)
    scores = model.score(query, [relevant, non_relevant])
    check("LogprobsReasoningFeedback", scores, is_probability=True)


def _get_pair(dataset: str) -> tuple[str, str, str]:
    if dataset == "scidocs":
        return SCIDOCS_QUERY, SCIDOCS_RELEVANT_DOC, SCIDOCS_NON_RELEVANT_DOC
    elif dataset == "fiqa":
        return FIQA_QUERY, FIQA_RELEVANT_DOC, FIQA_NON_RELEVANT_DOC
    else:
        raise ValueError(f"No test pair defined for dataset '{dataset}'")


def test_real_qrel(dataset: str = "fiqa") -> None:
    """Load a real query+relevant doc from qrels and check all models score it above a non-relevant doc."""
    print(f"\n--- Real qrel test [{dataset}] ---")

    with open("beir_data_path.json") as f:
        data_path = json.load(f)[dataset]

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    query_ids = list(queries.keys())

    # Pick first query at/after REAL_QREL_QUERY_INDEX that has qrels
    query_id = next(qid for qid in query_ids[REAL_QREL_QUERY_INDEX:] if qid in qrels and qrels[qid])
    query_text = queries[query_id]

    # Highest-scored relevant doc
    relevant_doc_id = max(qrels[query_id], key=lambda did: qrels[query_id][did])
    relevant_doc = corpus[relevant_doc_id]
    relevant_text = (relevant_doc.get("title") or "") + "\n" + (relevant_doc.get("text") or "")

    # First corpus doc not in qrels for this query
    non_relevant_doc_id = next(did for did in corpus if did not in qrels[query_id])
    non_relevant_doc = corpus[non_relevant_doc_id]
    non_relevant_text = (non_relevant_doc.get("title") or "") + "\n" + (non_relevant_doc.get("text") or "")

    print(f"  query:        {query_text[:120]}")
    print(f"  relevant:     {relevant_text[:120]}")
    print(f"  non-relevant: {non_relevant_text[:120]}")

    api_key = os.environ["OPENAI_API_KEY"]
    with open(FEEDBACK_PROMPTS_FILE) as f:
        all_prompts = json.load(f)

    print()
    print("OpenAIFeedback")
    scores = OpenAIFeedback(api_key=api_key).score(query_text, [relevant_text, non_relevant_text])
    check("OpenAIFeedback", scores, is_probability=False)

    print()
    print("LogprobsOrdinalFeedback")
    scores = LogprobsOrdinalFeedback(api_key=api_key, **all_prompts["logprobs-ordinal"][dataset]).score(
        query_text, [relevant_text, non_relevant_text]
    )
    check("LogprobsOrdinalFeedback", scores, is_probability=False)

    print()
    print("LogprobsFeedback")
    scores = LogprobsFeedback(api_key=api_key, **all_prompts["logprobs"][dataset]).score(
        query_text, [relevant_text, non_relevant_text]
    )
    check("LogprobsFeedback", scores, is_probability=True)

    print()
    print("LogprobsReasoningFeedback")
    scores = LogprobsReasoningFeedback(api_key=api_key, **all_prompts["logprobs-reasoning"][dataset]).score(
        query_text, [relevant_text, non_relevant_text]
    )
    check("LogprobsReasoningFeedback", scores, is_probability=True)


if __name__ == "__main__":
    for dataset in ("scidocs", "fiqa"):
        print(f"\n=== {dataset} ===")
        test_openai_feedback(dataset)
        print()
        test_logprobs_ordinal_feedback(dataset)
        print()
        test_logprobs_feedback(dataset)
        print()
        test_logprobs_reasoning_feedback(dataset)

    print(f"\n{'=' * 50}")
    for dataset in ("scidocs", "fiqa"):
        test_real_qrel(dataset)
