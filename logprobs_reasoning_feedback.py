"""
Relevance feedback using structured-output logprobs on gpt-4o-mini,
with a short reasoning field before the boolean verdict.

Identical to LogprobsFeedback except:
  - Schema: {"reasoning": str, "relevant": bool}
  - The model reasons briefly (1-2 sentences) before committing to true/false,
    which produces more calibrated logprob distributions on borderline cases.
  - Boolean extraction scans in reverse to find the LAST true/false token —
    robust against "relevant" or true/false appearing inside the reasoning text.
  - max_completion_tokens=150 to accommodate reasoning + JSON overhead.
"""

import math
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from qdrant_relevance_feedback.feedback import Feedback


class LogprobsReasoningFeedback(Feedback):
    class ModelOutputError(Exception):
        """Raised when the model returns a valid response but with unusable output."""

    class _RelevanceAssessment(BaseModel):
        reasoning: str
        relevant: bool

    _MODEL = "gpt-4o-mini"

    _DEFAULT_SYSTEM_PROMPT = (
        "Given a question and a candidate answer, briefly reason in 1-2 sentences "
        "about whether the candidate answer is relevant to the question, then give your verdict."
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
            top_logprobs=2,
            max_completion_tokens=150,  # reasoning (~60 tokens) + JSON overhead + boolean
            temperature=0,
        )

        msg = resp.choices[0].message
        if msg.refusal:
            raise LogprobsReasoningFeedback.ModelOutputError(
                f"Model refused to assess relevance: {msg.refusal}"
            )
        if msg.parsed is None:
            raise LogprobsReasoningFeedback.ModelOutputError(
                "Model returned no parsed output (msg.parsed is None)"
            )

        token_logprobs = resp.choices[0].logprobs
        if token_logprobs is None:
            raise LogprobsReasoningFeedback.ModelOutputError(
                "No logprobs in response (logprobs field is None)"
            )

        # Scan in reverse — `relevant` is the last field in the schema, so the last
        # true/false token is always the verdict, even if reasoning contains those words.
        bool_tok = None
        for tok in reversed(token_logprobs.content):
            if tok.token in ("true", "false"):
                bool_tok = tok
                break

        if bool_tok is None:
            raise LogprobsReasoningFeedback.ModelOutputError(
                f"No boolean token ('true'/'false') found in logprobs. "
                f"Tokens seen: {[t.token for t in token_logprobs.content]}"
            )

        true_lp = next(
            (lp.logprob for lp in bool_tok.top_logprobs if lp.token == "true"), None
        )
        false_lp = next(
            (lp.logprob for lp in bool_tok.top_logprobs if lp.token == "false"), None
        )

        if true_lp is not None and false_lp is not None:
            p_true = math.exp(true_lp)
            p_false = math.exp(false_lp)
            return p_true / (p_true + p_false)
        elif true_lp is not None:
            return math.exp(true_lp)
        elif false_lp is not None:
            return 1.0 - math.exp(false_lp)
        else:
            raise LogprobsReasoningFeedback.ModelOutputError(
                f"Neither 'true' nor 'false' in top_logprobs at boolean position: "
                f"{[(lp.token, lp.logprob) for lp in bool_tok.top_logprobs]}"
            )
