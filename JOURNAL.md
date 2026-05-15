# Relevance Feedback Benchmarking — Scientific Journal

Running log of hypotheses, findings, decisions, and open questions.

## Purpose

Testing Qdrant Relevance Feedback API as a **semantic-similarity** retrieval instrument.

**Core narrative:** People typically reach for reranking to improve retrieval quality. This project asks: can one-step RF API usage, in one of the two proposed configurations, be a better alternative than one-step reranking? “Better alternative” can mean, for example, cheaper, comparable in latency, and better on retrieval-quality metrics.
**Note:** For Agentic Search, reranking transformed itself into agentic reasoning, so a side goal would be to figure out if using RF API for agents (without reranking as a separate mechanism) is more beneficial than vanilla retrieval Query API, as RF API allows agent to propagate relevance feedback **right into vector search/index traversal**.

There are different search audiences who might be interested in different RF API configurations as a retrieval method:
- Traditional search users needing good nDCG@10 and low latency/cost at scale
- Agentic (RAG) users focused mainly on high Recall@1/3 to save on tokens, and less focused on latency (genAI is slow anyway)
- High-precision / (deep) research-oriented users who care about full discoverability (high Recall@10, high disc@10) of relevant documents, and are less focused on latency (and sometimes even less focused on cost)

The goal is to test different RF API usage patterns for different audiences, and understand which configuration (“Pure RF”, or "RF+Rerank") makes sense + under which constraints it does so.

### Experiment Goals

0. Find a strong feedback model that is a **good reranker** — one that meaningfully improves over a baseline (vanilla dense retrieval). “Meaningfully improves” means the reranker leads to better results on classical IR metrics like Recall and nDCG.
Why? 
(1) To prove that RF could be a meaningful replacement. If no reranker adds value, there's nothing to replace.
(2) Moreover, only a good feedback model, which brings value, should make sense in light of RF usage, as feedback model is the one adjusting the vector search with its signals.

**Note:**
RF was not designed specifically to replace reranking, but as an additional IR tool to improve relevance of results.  

However, as I've often been asked how it compares to reranking — and moreover, the RF API's main idea, which led to its development, is that it is a vector-index-native relevance feedback instrument, changing HNSW traversal based on feedback, whereas all other relevance feedback methods (that change the scoring function between query and document) treat search engines as black boxes and implement relevance-feedback-driven reranking on a limited, retrieved subset of results.  
Hence, this experiment is (roughly) a comparison between that entire family of methods (reranking-driven) and the Relevance Feedback API — to my knowledge, the first of its kind (across vector search engines).

Choices to try:
- LLM as a feedback model (binary, ordinal, binary with reasoning). A side-side goal is to discover which gpt-40 model configuration makes the best reranker, for a separate thought leadership piece (spoiler: None).
- OpenAI text-embeddings-large-3 embeddings (3072 dim)

**Note:**
These choices are based on the fact that “weaker” options — OpenAI small text embeddings and ColBERT as rerankers on BEIR SciDocs — failed to improve over the baseline (vanilla retrieval with mxbread) on standard BEIR qrel-based metrics.  
That said, it doesn’t mean they can’t help discoverability (disc@N metric) of new relevant documents via RF retrieval, they can. However, for search audiences it’s still easier to understand/compare with traditional Recall/nDCG when talking about search quality.

This failure may be connected (with a high probability) to the fact that SciDocs is not a semantic-similarity dataset per se but a citation-prediction dataset. That said, all retrievers out there are still benchmarked on BEIR SciDocs and better retrievers generally score higher on it (cc pyserini), so there is some semantic similarity signal, and a strong enough reranker should still help, quality metrics-wise.

Why SciDocs then? The answer is embarassing, it was already indexed and easy to access, making it the easiest initial choice for evaluating “is it a good reranker?”.

However, experiments in this repo also cover FiQA (much closer to a typical semantic-similarity / RAG task, hence a better dataset for the main goal - making a recommendation of RF API config for different search users), and, in general, code is generalized to support benchmarking any BEIR dataset.

2. Check that at least one of the two configurations — "RF+Rerank" or "Pure RF" — makes sense as a retrieval method compared to reranking, and identify under which generalizable conditions.

Specifically, that:
- **RF+Rerank** adds value compared to the reranker: same or cheaper cost (e.g., rerank 50 vs RF+rerank on 25) with higher quality (R@10/NSCG@10)/higher discoverability (disc@N). This way of RF usage is latency-heavy, so a quality improvement should be meaningful (not a noise of < 1% gain) to justify it.
- **Pure RF** gets close to reranker-based gains at significantly lower cost and latency.

### Hypotheses

**Prerequisite (P) for H1 and H2.** At least one feedback model produces a useful reranker — i.e., meaningfully improves over baseline retrieval.

**H1.** RF+Rerank > Rerank on all metrics, because RF discovers documents a reranker can’t reach (a reranker is limited to its initial fetch window).

It is worse latency-wise (retrieve → rerank → RF → rerank → merge by score, vs. retrieve → rerank one big pool) but can be equal or better cost-wise (if reranking a bigger pool still doesn’t surface relevant documents discovered by RF+Rerank).

**H2.** Pure RF is a viable cheap alternative to reranking, at least for one search audience (for example, agentic RAG search, where better R@1/3 matters, or some industries, where much lower cost and latency matters, but they still need reranking).

#### Side Hypotheses

**H3.** An LLM (gpt-4o-mini) can be a good relevance classifier → reranker if properly configured (testing logprobs + binary, ordinal, and binary with reasoning).

**H4.** Even if not (H3 fails), Pure RF or RF+Rerank may still improve over baseline retrieval with an LLM as a feedback model, making it a sensible production option (given the cost/latency/quality trade-off). For that, gains over baseline need to be truly meaningful.

**H5.** There is a good default context window size for RF methods, around 4–5, that works across different datasets.

**H6.** Retriever strength affects the usability of RF methods in a predictable way.

## Setup

**Datasets**
- **SciDocs** — citation prediction, 25k corpus. 200 train queries, eval-start=200, 500 eval queries. It was picked initially, but it is a weaker example for semantic similarity and RAG purposes, as it is a citation prediction dataset.
- **FiQA** — financial QA, 57k corpus. 148 train queries, eval-start=148, 500 eval queries. A good example of domain-specific RAG/semantic similarity.

The code in this repo allows benchmarking any BEIR dataset.

**Retrievers**
- `jina` — jina-embeddings-v2-base-en, 768d, local fastembed (CPU)
- `mxbread` — mxbai-embed-large-v1, 1024d, Qdrant cloud inference

**Feedback models**
- `openai-emb` — cosine similarity via text-embedding-3-large.
- `logprobs` — gpt-4o-mini structured output P(relevant=true) extracted from boolean token logprobs.
- `logprobs-reasoning` — same as logprobs but generates a short reasoning field (1–2 sentences) before the boolean verdict.
- `logprobs-ordinal` — same as logprobs but ordinal 1–5 score, Σ i·P(i) extracted from top_logprobs at the digit token position.

**Metrics**
Recall@[1, 3, 5, 10] measured by `ranx` + nDCG@10.

**Why Recall@1/3/5:** fewer retrieved chunks = more agent/LLM-friendly method.

**Why nDCG@10:** enables comparison with the pyserini BEIR leaderboard of retrievers, as a sanity check. SciDocs nDCG@10 is around 0.217 and FiQA nDCG@10 around 0.421.

**Note on tables:** in all results tables, entries formatted as `+X% → value` (or `−X% → value`) are percentage changes **relative to the Baseline column** in the same row.

Additionally, the **disc@N** metric can show how many relevant documents (according to qrels) were discovered and ranked highly enough to enter the @N window by RF-based methods — documents that were unreachable for the reranker (which can only reshuffle its initial retrieved pool).  

**Summed** across training set of queries (500, disc@N (raw)) and **normalized:** **disc@N (norm)** = disc@N (raw) / Σ_q min(N, |relevant docs outside rerank pool for query q|). The denominator is the maximum number of new relevant docs RF *could* have surfaced at cutoff N across all queries. A value of 0.07 means RF placed 7% of those theoretically discoverable documents across the full test set into the top-N window.
This metric tracks the needs of search audiences who are interested in research-oriented tasks - tasks where discovering as much relevant data as possible is critical.

### Methods

| Method | Description |
|--------|-------------|
| **Baseline** | ANN retrieval by embedding similarity |
| **Rerank** | Retrieve N docs, rescore all with feedback model |
| **RF+Rerank** | Retrieve context docs (3–5) → score with feedback model → RF query (filtering out context docs) → score RF results with feedback model → merge by score |
| **Pure RF** | Retrieve context docs (3–5) → score with feedback model → RF query (no filtering of the context docs) |

**RF formula:** `a × cosine(candidate, query) + sum_{ctx_pair}[confidence^b × delta × c]`

- `confidence` = feedback_score(positive_ctx) − feedback_score(negative_ctx)
- `delta` = cosine(candidate, positive_ctx) − cosine(candidate, negative_ctx)
- Parameters `a, b, c` trained per `{dataset}-{retriever}-{feedback_model}` on the training split.
- **Training** uses top-1 confidence context pair per query to learn a, b, c.
- **Inference** passes all context docs to Qdrant RF API, which sums contributions from all possible pairs.

## Findings

### Disagreement rates, training

Fraction of training queries where the feedback model strongly disagrees with retriever ranking (i.e., the feedback model scores a lower-ranked doc as more relevant than a higher-ranked one).

| Dataset | Retriever | Feedback model | Disagreement |
|---------|-----------|---------------|--------------|
| FiQA | mxbread | openai-emb | 21.6% |
| FiQA | jina | openai-emb | 21.6% |
| FiQA | mxbread | logprobs | 47.3% |
| FiQA | jina | logprobs | 48.65% |
| FiQA | jina | logprobs-ordinal | 45.95% |
| FiQA | mxbread | logprobs-ordinal | 56.08% |
| SciDocs | jina | logprobs-ordinal | 35.81% |
| SciDocs | mxbread | logprobs-ordinal | 42.57% |

A feedback model disagreeing is the key signal: if the model is right in its relevance judgments — i.e., if it is a good reranker — there is a better chance that RF will improve results. If it’s wrong, it’s noise.

### Good Reranker (P) and LLM (openai-40-mini) as a Good Reranker (H3)

#### P (Good Reranker)
openai-emb is a meaningful reranker.  
It adds meaningful value on FiQA (both retrievers, +15–20% R@1) and SciDocs × jina (+15% R@1).
However, it adds almost nothing on SciDocs × mxbread — the mxbread baseline is already strong for similarity-based retrieval, and openai-emb is a better similarity measurer, not a better citation predictor (the task of SciDocs).
FiQA is more semantic-similarity-driven, so a stronger embedding model genuinely reranks better → FiQa dataset should be the main one for drawing conclusions.

#### H3 (openai-40-mini as a Good Reranker)

**Results** (FiQA × mxbread, 200 eval queries, rerank_limit=25):

| Method | R@1 | R@3 | R@5 | R@10 | nDCG@10 | Est. cost/200q |
|--------|-----|-----|-----|------|---------|----------------|
| Baseline | 0.2344 | 0.3925 | 0.4439 | 0.5193 | 0.4767 | — |
| logprobs-reasoning rerank | -25% → 0.1747 | -11% → 0.3507 | -2% → 0.4349 | ~0% → 0.5178 | -9% → 0.4326 | $0.37 |
| logprobs rerank | -19% → 0.1898 | -1% → 0.3880 | +10% → 0.4865 | +10% → 0.5723 | -2% → 0.4685 | $0.16 |
| logprobs-ordinal rerank | -13% → 0.2032 | **+4% → 0.4093** | +7% → 0.4765 | +7% → 0.5576 | ~0% → 0.4747 | $0.22 |
| openai-emb rerank | **+15% → 0.2697** | **+19% → 0.4664** | **+19% → 0.5279** | **+17% → 0.6074** | **+17% → 0.5564** | $0.10 |

**Note:**
**LLM-as-a-reranker non-determinism:** It seems impossible to enforce determinism (temperature=0 is greedy, but not bit-for-bit reproducible across calls). Practical implication: differences of less than ~3% between runs are noise, not signal.

**All of them are worse than openai-emb** across all metrics. Why?
- **Failure modes for logprobs/logprobs-reasoning:** *Polarization* — many docs score 1.0 (both irrelevant and relevant ones), so ordering within the group is arbitrary, destroying the ANN signal that would have placed more relevant docs at rank 1. This explains the R@1/R@3 drops.
- **Failure modes for all logprobs models:** *Mislabeling* — relevant docs are scored low and pushed below rank 10. The current prompts, however, are reasonably aligned with BEIR task definitions, and further prompt engineering is out of scope of these experiments.

If we still have to choose between LLM-as-a-reranker config:
- **Reasoning isn’t worth the cost; it even makes reranking worse.** Likely because the generated reasoning string commits the model to a direction before the boolean verdict, eliminating uncertainty, so there's no fine grained reranking possible.
- **Ordinal vs binary logprobs — split result:** ordinal wins at R@1 and R@3. Binary logprobs win at R@5/R@10, though that gap is borderline given ~3% LLM non-determinism-related noise. For RF context selection (top 3–5 docs, very small cutoffs, closer to R@3), **ordinal** may still be the better feedback model => it is **the final choice** for experiments.

#### H5 (Context Window Size)

**Recommendation: ctx=5.** Optimal or tied-best across FiQA and SciDocs with openai-emb and both retrievers. The exception is SciDocs × mxbread, where ctx=4 wins at R@1 and ctx=3 at nDCG for both RF methods — but RF adds almost nothing on that dataset anyway, so it’s not a meaningful counter-signal.

- FiQA: ctx=5 wins on all 4 combinations by nDCG and R@3–R@10. R@1 is marginal: ctx=4 occasionally edges ctx=5 but differences are within noise.
- SciDocs × jina: ctx=5 is best for both Pure RF and RF+Rerank across all metrics.
- SciDocs × mxbread: ctx=4 is best at R@1 for Pure RF (0.0576) and RF+Rerank (0.0548); ctx=3 is best at nDCG. But RF adds almost nothing over baseline on this dataset and retriever regardless.

### H1 — RF+Rerank > Rerank

Tested with a good reranker (openai-emb, which provides meaningful improvements).

**FiQA results (ctx=5, 500q):** (grid_search_openai_emb_fiqa)

| | Baseline | Rerank(25) | RF+Rerank(25) | Rerank(50) |
|--|---------|-----------|--------------|-----------|
| mxbread R@1 | 0.2080 | +17.9% → 0.2452 | +18.0% → 0.2455 | **+19.6% → 0.2488** |
| mxbread R@3 | 0.3579 | +19.8% → 0.4289 | **+21.5% → 0.4348** | +20.9% → 0.4326 |
| mxbread R@5 | 0.4166 | +19.3% → 0.4970 | **+21.0% → 0.5042** | +20.4% → 0.5014 |
| mxbread R@10 | 0.5027 | +14.5% → 0.5754 | +16.4% → 0.5852 | **+17.7% → 0.5918** |
| mxbread nDCG@10 | 0.4362 | +16.3% → 0.5075 | +17.8% → 0.5138 | **+18.8% → 0.5181** |
| jina R@1 | 0.1924 | +24.6% → 0.2397 | +23.5% → 0.2376 | **+25.2% → 0.2409** |
| jina R@3 | 0.3173 | +29.9% → 0.4122 | +30.1% → 0.4128 | **+32.5% → 0.4205** |
| jina R@5 | 0.3705 | +26.7% → 0.4695 | +29.0% → 0.4780 | **+32.7% → 0.4916** |
| jina R@10 | 0.4600 | +13.9% → 0.5239 | +18.1% → 0.5431 | **+24.1% → 0.5710** |
| jina nDCG@10 | 0.3932 | +21.4% → 0.4773 | +23.1% → 0.4841 | **+27.1% → 0.4997** |

| cost/500q | — | $0.2437 | $0.2437 | $0.4875 |

Gains are small (+1–4%) but present for nDCG@10/R@10 at comparable pool sizes (25). If we increase the reranking pool to 50 (making RF+Rerank roughly 2× cheaper), RF+Rerank(25) approaches Rerank(50) at half the cost (mxbread −0.8%, jina −3.1%). However, it still has **worse latency**: the openai-emb API is called twice (context scoring + RF doc scoring), making RF+Rerank slower than even Rerank(50) (~1260ms vs ~900ms for mxbread; ~1000ms vs ~840ms for jina).

| Retriever | Method | disc@1 | disc@3 | disc@5 | disc@10 | disc@25 |
|-----------|--------|--------|--------|--------|---------|---------|
| mxbread | RF+Rerank(25) | 0.010 | 0.030 | 0.038 | 0.060 | 0.085 |
| jina | RF+Rerank(25) | 0.012 | 0.033 | 0.049 | 0.079 | 0.123 |

RF+Rerank has higher disc@10 than Pure RF (6.0% vs 2.1% for mxbread; 7.9% vs 5.0% for jina) despite exploring similar HNSW neighborhoods, as the second reranking step pulls newly discovered relevant docs from the outer pool (25) up into the top-10 window.

disc@10 at 6-7% means 6-7% of all relevant documents undiscoverable by reranker become visible in top-10 results.

**SciDocs results (ctx=5, 500q):** (grid_search_openai_emb_scidocs)

Note: openai-emb is a weaker reranker on SciDocs × mxbread, it adds almost nothing over baseline (see P).

| | Baseline | Rerank(25) | RF+Rerank(25) | Rerank(50) |
|--|---------|-----------|--------------|-----------|
| mxbread R@1 | 0.0531 | **+3.2% → 0.0548** | +0.2% → 0.0532 | +0.9% → 0.0536 |
| mxbread R@3 | 0.1246 | +2.7% → 0.1280 | **+5.8% → 0.1318** | +3.5% → 0.1289 |
| mxbread R@5 | 0.1794 | +2.4% → 0.1837 | **+3.2% → 0.1852** | +0.9% → 0.1811 |
| mxbread R@10 | 0.2611 | **+3.6% → 0.2706** | +1.6% → 0.2653 | +0.8% → 0.2632 |
| mxbread nDCG | 0.2419 | **+3.3% → 0.2500** | +1.9% → 0.2465 | +1.1% → 0.2446 |
| jina R@1 | 0.0451 | +18.0% → 0.0532 | **+21.3% → 0.0547** | +16.0% → 0.0523 |
| jina R@3 | 0.1065 | +19.0% → 0.1267 | **+25.1% → 0.1332** | +20.9% → 0.1288 |
| jina R@5 | 0.1556 | +12.6% → 0.1752 | **+18.0% → 0.1836** | +14.3% → 0.1779 |
| jina R@10 | 0.2208 | +12.7% → 0.2488 | **+18.2% → 0.2611** | +15.8% → 0.2556 |
| jina nDCG | 0.2055 | +14.6% → 0.2355 | **+19.6% → 0.2457** | +16.4% → 0.2391 |

| cost/500q | — | $0.3250 | $0.3250 | $0.6500 |

| Retriever | Method | disc@1 | disc@3 | disc@5 | disc@10 | disc@25 |
|-----------|--------|--------|--------|--------|---------|---------|
| mxbread | RF+Rerank(25) | 0.004 | 0.011 | 0.011 | 0.012 | 0.011 |
| jina | RF+Rerank(25) | 0.004 | 0.009 | 0.008 | 0.010 | 0.009 |

SciDocs raw disc counts are large (52–60 at disc@10) but the denominator is enormous. Citation-prediction datasets have huge relevant sets per query, driving normalized disc close to zero regardless of method.

**Since openai-emb barely improves over baseline, RF+Rerank introduces noise.**

#### Conclusion

For the conclusion, we also need to consider the training costs of RF (one training per dataset × retriever × feedback model triple), as reranking requires no training at all.

| Training | Cost |
|-----------|------|
| Per retriever — openai-emb (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.07 · SciDocs ~$0.13 |
| Per retriever — logprobs-ordinal (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.16 · SciDocs ~$0.27 |

RF+Rerank(25) is slower latency-wise: it requires two feedback-model API calls (context scoring + RF doc scoring) and two search calls, versus one and one for reranking (25/50). 

RF+Rerank(25) consistently beats Rerank(25) on FiQA at the same cost of reranking 25 documents (+1–4% R@3–R@10, +1.2–1.4% nDCG@10). Against Rerank(50) which has 2× cost, RF+Rerank(25) falls behind — mxbread: −1.1% R@10, −0.8% nDCG@10; jina: −5.1% R@10, −3.2% nDCG@10. It shows a need in increasing RF+Rerank limit, for example, to 35 (still saves cost compared to reranking of 50). However, it is a matter of experimentation per dataset, retriever, feedback model and budget => the optimal rf-limit relative to the reranking pool size should be defined based on the cost–latency–quality balance.

The clear advantage here is discovery: RF+Rerank surfaces documents from entirely outside the initial ANN pool. On FiQA, it placed 6.0% (mxbread) and 7.9% (jina) of all theoretically discoverable relevant documents across the test set of queries — those unreachable by reranker (Rerank(25)) — into the top-10 window (disc@10).

**Target audience:** latency and cost-tolerant, discovery-oriented use cases — research, legal, medical, any domain where breadth of recall at a wider window matters more than speed.

**Condition:** only applies when the feedback model has real signal. Breaks on SciDocs where openai-emb is too weak — RF+Rerank degrades to noise.

### H2 — Pure RF as cheap alternative to reranking

Tested with a good reranker (openai-emb, which provides meaningful improvements).

- Pure RF with ctx=5 and RF=20 is 5× cheaper than Rerank(25).  
- Latency here is comparable to or lower than reranking, depending on the feedback model.

**FiQA results (ctx=5, 500q):**

| | Baseline | Rerank(25) | Pure RF (ctx=5) |
|--|---------|-----------|----------------|
| mxbread R@1 | 0.2080 | +18% → 0.2452 | +18% → 0.2445 |
| mxbread R@3 | 0.3579 | +20% → 0.4289 | +13% → 0.4041 |
| mxbread R@5 | 0.4166 | +19% → 0.4970 | +9% → 0.4526 |
| mxbread R@10 | 0.5027 | +14% → 0.5754 | +8% → 0.5402 |
| mxbread nDCG | 0.4362 | +16% → 0.5075 | +11% → 0.4834 |
| jina R@1 | 0.1924 | +25% → 0.2397 | +15% → 0.2211 |
| jina R@3 | 0.3173 | +30% → 0.4122 | +13% → 0.3587 |
| jina R@5 | 0.3705 | +27% → 0.4695 | +10% → 0.4079 |
| jina R@10 | 0.4600 | +14% → 0.5234 | +5% → 0.4837 |
| jina nDCG | 0.3932 | +21% → 0.4770 | +10% → 0.4340 |

| cost/500q | — | $0.2437 | $0.0488 |

For mxbread, Pure RF is essentially tied with Rerank(25) at R@1 (0.2445 vs 0.2452, −0.3% — within noise) at 5× less cost and comparable latency (947ms vs 918ms). At deeper cutoffs it falls behind: −6% R@3, −9% R@5, −6% R@10.  
For jina, the gap is larger: −8% R@1, −13% R@3. Both retrievers beat baseline substantially (+17% R@1 mxbread, +15% R@1 jina).

| Retriever | Method | disc@1 | disc@3 | disc@5 | disc@10 | disc@25 |
|-----------|--------|--------|--------|--------|---------|---------|
| mxbread | Pure RF(ctx=5) | 0.000 | 0.000 | 0.005 | 0.021 | 0.090 |
| jina | Pure RF(ctx=5) | 0.000 | 0.005 | 0.018 | 0.050 | 0.138 |

**SciDocs results (ctx=5, 500q):**

Note: as with H1, openai-emb adds little on SciDocs × mxbread as a reranker, so the comparison there is mainly Pure RF vs baseline.

| | Baseline | Rerank(25) | Pure RF (ctx=5) |
|--|---------|-----------|----------------|
| mxbread R@1 | 0.0531 | +3% → 0.0548 | +7% → 0.0568 |
| mxbread R@3 | 0.1246 | +3% → 0.1280 | +5% → 0.1314 |
| mxbread R@5 | 0.1794 | +2% → 0.1837 | −4% → 0.1732 |
| mxbread R@10 | 0.2611 | +4% → 0.2706 | −11% → 0.2318 |
| mxbread nDCG | 0.2419 | +3% → 0.2500 | −5% → 0.2305 |
| jina R@1 | 0.0451 | +18% → 0.0532 | +23% → 0.0553 |
| jina R@3 | 0.1065 | +19% → 0.1267 | +17% → 0.1245 |
| jina R@5 | 0.1556 | +13% → 0.1752 | +9% → 0.1699 |
| jina R@10 | 0.2208 | +13% → 0.2488 | +8% → 0.2376 |
| jina nDCG | 0.2055 | +15% → 0.2355 | +11% → 0.2289 |

| cost/500q | — | $0.3250 | $0.0650 |

Pure RF wins at R@1 over both Rerank and baseline for both retrievers (+3.7% mxbread, +3.9% jina over Rerank(25)).  
Notably, Pure RF beats Rerank at R@1 on SciDocs × mxbread despite openai-emb being a weak reranker there. At R@3 it still beats Rerank on mxbread (+2.7%) but falls just below for jina (−1.7%). R@5 onward it drops behind — sharply for mxbread (−14% R@10).

| Retriever | Method | disc@1 | disc@3 | disc@5 | disc@10 | disc@25 |
|-----------|--------|--------|--------|--------|---------|---------|
| mxbread | Pure RF(ctx=5) | 0.000 | 0.005 | 0.009 | 0.011 | 0.012 |
| jina | Pure RF(ctx=5) | 0.000 | 0.002 | 0.005 | 0.009 | 0.010 |

#### Conclusion

For the conclusion, we also need to consider the training costs of RF (one training per dataset × retriever × feedback model triple), as reranking requires no training at all.

| Training | Cost |
|-----------|------|
| Per retriever — openai-emb (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.07 · SciDocs ~$0.13 |
| Per retriever — logprobs-ordinal (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.16 · SciDocs ~$0.27 |

Pure RF (ctx=5, rf=20) is 5× cheaper than Rerank(25) ($0.0488 vs $0.2437/500q) at comparable latency (947ms vs 918ms for mxbread).  
On FiQA, mxbread matches Rerank(25) at R@1 (0.2445 vs 0.2452, −0.3% — within noise); jina falls behind (−8% R@1) but still much better than baseline.  

At deeper cutoffs both retrievers trail Rerank(25).

disc@10 is not high (2.1% for mxbread, 5.0% for jina on FiQA) and is 0 at small cutoffs → **Pure RF’s value is top-cutoff ranking at low cost, not discovery.**

**Target audience:** RAG / agentic search where the right top-1/3 match is the primary signal or audiences where per-query cost/latency matters but they still need reranking for high R@1/R@3. Strongest case here with a strong retriever (mxbread) at R@1.

**Condition:** only applies where the feedback model has real signal (semantic similarity tasks, openai-emb on FiQA).

### Other findings

#### H4 (There's Still Some Use in LLM (openai-4o) -as-a-feedback model for RF API)
grid_search_logprobs_ordinal declines the hypothesis, no gains for RF, methods perform worse than the baseline.

#### H6 (Retriever Strength Clearly Affects RF Results)

**Retriever strength and ranking ability (NDCG@N) of RF API:** On FiQA, the stronger retriever (mxbread) produces better absolute Pure RF results, but marginal gains from RF over baseline are nearly identical across retrievers (+11% vs +10% nDCG).  
On SciDocs the picture is mixed and confounded by openai-emb’s weak signal on that dataset.  
No clear pattern — too little data to conclude whether retriever strength amplifies or dampens Pure RF’s marginal contribution.

**Retriever strength and discovery (disc@N) ability of RF API:** On FiQA, the weaker retriever (jina) has higher disc@10 (0.079 vs 0.060) — consistent with the intuition that a stronger retriever’s initial pool already covers the relevant set more completely, leaving less room for RF traversal to surface new documents. SciDocs disc is near zero for both retrievers, making it uninformative for this hypothesis.

More datasets would be needed to draw a reliable conclusion.

## Conclusion

These experiments compare RF API configurations against reranking. Query rewriting is out of scope and can be considered complementary, not competing.

**LLM as a feedback model** (logprobs, logprobs-ordinal, logprobs-reasoning) is neither a good reranker nor a good RF feedback model on these datasets. No configuration produces meaningful gains over baseline.

**Recommendations by search audience:**

| Audience | Primary signal | Method | Notes |
|---|---|---|---|
| RAG / agentic or "needs reranker but cheap" | R@1/R@3 | Pure RF | Matches Rerank(25) at R@1 (mxbread, −0.3%), 5× cheaper, same or lower latency. One-time training $0.07/retriever. |
| Discovery-oriented (research, legal, medical) | R@10, disc@10, latency-tolerant | RF+Rerank | 6–8% disc@10 on FiQA, beats Rerank(25) on R@10/nDCG@10 at the same cost. Worse latency due to two API call batches. |

**Conditions:** both methods require a feedback model with real signal. ctx=5 is a robust default across datasets (H5). Retriever strength affects discovery headroom but not ranking gain — more datasets are needed to conclude (H6).

Would be nice to try: an agent (Claude) using Pure RF as a primary search tool (a=1, b=0, c=0 for vanilla search; use a skill to let the agent train weights for itself; feedback doesn’t even have to come from the dataset; could use Cloud Inference), for example on Qdrant skills. A result should probably include a comparison with query rewriting, in at least one call.

## Appendix

### Experiment Output Files

| File | Contents |
|------|----------|
| `rerank_search_limit25.txt` | Phase 1 rerank-only sweep. 8 runs: 2 datasets × 2 retrievers × {openai-emb, logprobs-reasoning}, rerank_limit=25, 200 eval queries each. Used to establish the best reranker & check how LLM with reasoning works as a reranker. |
| `grid_search_openai_emb_fiqa.txt` | Full grid search: FiQA × {mxbread, jina} × openai-emb. 12 experiments: 2 retrievers × rerank_limits={25,50} × rf_pairs={(3,22),(4,21),(5,20)}, 500 eval queries. |
| `grid_search_openai_emb_scidocs.txt` | Full grid search: SciDocs × {mxbread, jina} × openai-emb. 12 experiments: 2 retrievers × rerank_limits={25,50} × rf_pairs={(3,22),(4,21),(5,20)}, 500 eval queries. |
| `grid_search_logprobs_ordinal_fiqa.txt` | FiQA × {mxbread, jina} × logprobs-ordinal. 4 experiments: rerank={25,50} × rf_pairs={(5,20)}, 500 eval queries. |
| `discovery_experiment.txt` | D1 discovery experiment: FiQA and SciDocs × {mxbread, jina} × openai-emb. 8 runs: pure-rf and rf+rerank, ctx=5, rf-limit=20, rerank pool=25, 500 eval queries each. Measures disc@N alongside Recall/nDCG. |
