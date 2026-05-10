"""
Relevance feedback using structured-output logprobs on gpt-4o-mini.
# https://www.colehoffer.ai/articles/structured-logprobs-for-classification/
# https://developers.openai.com/api/docs/guides/structured-outputs

--- Purpose ---

Scores each (query, document) pair with a relevance score.
The score should strive to be:
  - Continuous, not a hard label.
  - Absolute (independent of other documents in the batch).
  - Calibrated: 0.9 on one query means the same thing as 0.9 on another.
  - Higher score means higher relevance (relevance according to SCIDOCS retrieval task).

--- Structured output ---

The usual implementation asks the model to freely output "Yes"/"No" and
scans top_logprobs for "Yes" variants. Problems:
  - "Yes" could fall outside the top-5, silently returning 0.0.
  - Casing variants ("yes"/"Yes"/"YES") require a lookup set.

With response_format=json_schema and strict=True, at each token, the model's
logits are masked to only allow tokens that could produce valid JSON conforming
to the schema. For a boolean field, only "true" and "false" are ever valid,
so P(true) + P(false) ≈ 1.0 in the constrained space.

--- Score extraction ---

The API parameter top_logprobs=N returns the top N most probable tokens at every
token position in the output (max allowed value: 20). We set top_logprobs=2.

At the boolean token position (where "true" or "false" is emitted), we look for
both tokens in the returned top_logprobs. When both are present, the score is a
softmax over the two:

    score = exp(logprob_true) / (exp(logprob_true) + exp(logprob_false))

This is more robust than exp(logprob_true) alone because the constrained logprobs
may not sum to exactly 1.0 in exp-space due to numerical noise from the masking.

However, top_logprobs=2 does NOT guarantee both "true" and "false" appear. When
the model is near-certain, the 2nd slot can be taken by a tokenizer variant (e.g.
" false" with a leading space) rather than the opposite boolean. In that case only
one of the two is present and the complement is inferred: if only "false" is found,
score = 1.0 - exp(logprob_false); if only "true" is found, score = exp(logprob_true).
"""

import math
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from qdrant_relevance_feedback.feedback import Feedback


class LogprobsFeedback(Feedback):
    class ModelOutputError(Exception):
        """Raised when the model returns a valid response but with unusable output."""

    class _RelevanceAssessment(BaseModel):
        relevant: bool

    _MODEL = "gpt-4o-mini"

    _DEFAULT_SYSTEM_PROMPT = (
        "Given a question, decide whether the candidate answer is relevant."
    )
    _DEFAULT_USER_TEMPLATE = (
        "Question: {query}\n\nCandidate answer: {document}\n\n"
    )

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
            top_logprobs=2, #per token
            max_completion_tokens=20, #should be enough for json schema
            temperature=0,
        )

        msg = resp.choices[0].message
        if msg.refusal:
            raise LogprobsFeedback.ModelOutputError(f"Model refused to assess relevance: {msg.refusal}")
        if msg.parsed is None:
            raise LogprobsFeedback.ModelOutputError("Model returned no parsed output (msg.parsed is None)")

        token_logprobs = resp.choices[0].logprobs
        if token_logprobs is None:
            raise LogprobsFeedback.ModelOutputError("No logprobs in response (logprobs field is None)")

        for tok in token_logprobs.content:
            if tok.token not in ("true", "false"):
                continue

            true_lp = next(
                (lp.logprob for lp in tok.top_logprobs if lp.token == "true"), None
            )
            false_lp = next(
                (lp.logprob for lp in tok.top_logprobs if lp.token == "false"), None
            )

            if true_lp is not None and false_lp is not None:
                p_true = math.exp(true_lp)
                p_false = math.exp(false_lp)
                return p_true / (p_true + p_false)
            elif true_lp is not None:
                # Model is near-certain true; false has negligible probability.
                return math.exp(true_lp)
            elif false_lp is not None:
                # Model is near-certain false; true has negligible probability.
                return 1.0 - math.exp(false_lp)
            else:
                raise LogprobsFeedback.ModelOutputError(
                    f"Neither 'true' nor 'false' in top_logprobs at boolean position: "
                    f"{[(lp.token, lp.logprob) for lp in tok.top_logprobs]}"
                )

        raise LogprobsFeedback.ModelOutputError(
            f"No boolean token ('true'/'false') found in logprobs. "
            f"Tokens seen: {[t.token for t in token_logprobs.content]}"
        )
