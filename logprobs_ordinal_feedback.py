"""
Relevance feedback using an ordinal 1–5 schema with expected-value extraction.

Instead of a boolean field (which collapses to near-0/near-1 under temperature=0
constrains), the model outputs a score in {1, 2, 3, 4, 5}.

Score extraction:
    At the token position where a digit 1–5 is emitted, traverse top_logprobs
    (up to 10). For each entry, try to parse it as an integer 1–5. Digits not found in
    top_logprobs are assigned P=0.

    Final score = Σ i · P(i)  for i ∈ {1..5} (well, it's not exactly P(i) or E[i] since the constrained logprobs may easily not sum to 1.0 due to numerical noise, but it's close enough for our purposes).
"""

import math
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel
from qdrant_relevance_feedback.feedback import Feedback


class LogprobsOrdinalFeedback(Feedback):
    class ModelOutputError(Exception):
        """Raised when the model returns a valid response but with unusable output."""

    class _RelevanceAssessment(BaseModel):
        score: Literal[1, 2, 3, 4, 5]

    _MODEL = "gpt-4o-mini"

    _DEFAULT_SYSTEM_PROMPT = (
        "Given a question and a candidate answer, rate the relevance of the candidate "
        "answer to the question on a scale from 1 to 5:\n"
        "1 = irrelevant\n"
        "2 = not sure if relevant\n"
        "3 = marginally relevant\n"
        "4 = relevant\n"
        "5 = perfect match"
    )
    _DEFAULT_USER_TEMPLATE = "Question: {query}\n\nCandidate answer: {document}\n\n"

    def __init__(
        self,
        api_key: str,
        model_name: str = _MODEL,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        user_template: str = _DEFAULT_USER_TEMPLATE,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._user_template = user_template
        self._client = OpenAI(api_key=api_key, timeout=60.0, **kwargs)

    def score(self, query: str, responses: list[str]) -> list[float]:
        """Score each document independently; one API call per document."""
        return [self._score_one(query, doc) for doc in responses]

    def _score_one(self, query: str, document: str) -> float:
        resp = self._client.chat.completions.parse(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": self._user_template.format(query=query, document=document),
                },
            ],
            response_format=self._RelevanceAssessment,
            logprobs=True,
            top_logprobs=10,
            max_completion_tokens=20,
            temperature=0,
        )

        msg = resp.choices[0].message
        if msg.refusal:
            raise LogprobsOrdinalFeedback.ModelOutputError(
                f"Model refused: {msg.refusal}"
            )
        if msg.parsed is None:
            raise LogprobsOrdinalFeedback.ModelOutputError(
                "Model returned no parsed output"
            )

        token_logprobs = resp.choices[0].logprobs
        if token_logprobs is None:
            raise LogprobsOrdinalFeedback.ModelOutputError(
                "No logprobs in response"
            )

        for tok in token_logprobs.content:
            if tok.token in ["1", "2", "3", "4", "5"]:
                # Found the digit token position — collect distribution over 1–5.
                # Only exact single-digit tokens are kept; variants like " 3" or "3\n"
                # are filtered out. Missing digits are treated as P=0.
                score_probs = {
                    int(lp.token): math.exp(lp.logprob)
                    for lp in tok.top_logprobs
                    if lp.token in ["1", "2", "3", "4", "5"]
                }

                if not score_probs:
                    raise LogprobsOrdinalFeedback.ModelOutputError(
                        f"Score token '{tok.token}' found but no exact digit in top_logprobs: "
                        f"{[(lp.token, lp.logprob) for lp in tok.top_logprobs]}"
                    )

                return sum(v * p for v, p in score_probs.items())

        raise LogprobsOrdinalFeedback.ModelOutputError(
            f"No score token (1–5) found in logprobs. "
            f"Tokens seen: {[t.token for t in token_logprobs.content]}"
        )
