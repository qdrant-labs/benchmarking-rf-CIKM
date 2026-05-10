# Relevance Feedback Benchmarking — Scientific Journal

Running log of hypotheses, findings, decisions, and open questions.

## Purpose

Testing Qdrant Relevance Feedback as a **semantic similarity** retrieval instrument.

**Core narrative:** People typically reach for reranking to improve retrieval quality. This project asks: can 1-step RF in one of the two proposed configurations be a better alternative than reranking? Better alternative = cheaper, comparable latency-wise, and providing better retrieval quality on metrics.

There are two audiences considering RF as a retrieval method: traditional search users needing good nDCG scores and low latency/cost at scale, and agentic search/RAG users needing mainly high Recall@1/3 for saving on tokens and caring about cost more than latency.

The goal is to check RF applicability for both audiences and under which configurations and conditions it is applicable.

### Experiment Goals

0. Find a strong feedback model that is a **good reranker** — one that meaningfully improves over a baseline (vanilla dense retrieval). "Meaningfully improves" means using the reranker leads to better results on classical IR metrics like Recall and nDCG.
Why? To prove that RF could be a meaningful replacement. If no reranker adds value, there's nothing to replace.

**Note:**
RF may still improve results over a baseline using a feedback model that proved to be a "bad" reranker. This is also a side hypothesis to check.

**Note:**
RF itself was not designed to replace reranking, but as an additional IR tool to improve relevance of results.  
However, as I've often been asked how it compares to reranking — and moreover, the RF API's main value is that it is a vector-index-native relevance feedback instrument, reordering the whole dataset based on feedback (changing HNSW traversal based on feedback), whereas all other relevance feedback methods that change not the query but the scoring function between query and document treat search engines as black boxes and implement relevance-feedback-driven reranking.  
Hence, this experiment effectively makes a comparison between that entire family of methods (reranking-driven) and the Relevance Feedback API, the first of its kind for a vector search engine.

Choices to try:
- LLM as a feedback model (binary, ordinal, binary with reasoning). A side goal is to discover which model configuration makes the best reranker, for a separate thought leadership piece.
- OpenAI text-large embeddings

**Note:**
These choices are based on the fact that "weaker" options — OpenAI small embeddings and ColBERT used as rerankers on BEIR SciDocs — demonstrated an inability to improve over the baseline (vanilla retrieval metrics).

This may be connected to the fact that SciDocs is not a semantic similarity dataset per se but a citation prediction dataset. That said, all retrievers are tested on SciDocs and better retrievers generally score higher on it (cc pyserini), so the hypothesis is that a capable enough reranker will improve the metrics.

Why SCIDOCS then? It was already indexed and easy to access, making it the initial choice for evaluating "is it a good reranker?"  
However, experiments in this repo also cover FiQA (much closer to a typical semantic similarity / RAG task), and the code infrastructure allows benchmarking on any BEIR dataset.

2. Check that at least one of two configurations — RF+Rerank or Pure RF — adds value compared to the reranker and identify under which generalizable conditions (via a grid search over reranking pool size, context window size for RF methods, and the number of points RF-based retrieval returns).

Specifically:
- **RF+Rerank** adds value compared to the reranker: same or cheaper cost (e.g., rerank 50 vs RF+rerank on 25) with higher quality. This way of RF usage is latency-heavy, so a quality improvement should be truly meaningful to justify it.
- **Pure RF** gets close to reranker-based quality improvements at significantly lower cost and latency.

Additionally, provide metrics to demonstrate that RF-based methods surface relevant documents from the dataset that the reranker cannot reach (disc@N).

### Hypotheses

**Prerequisite (P) for H1 and H2.** At least one feedback model produces a useful reranker — i.e. meaningfully improves over baseline retrieval.

**H1.** RF+Rerank > Rerank on all metrics, because RF discovers documents a reranker can't reach (a reranker is limited to its initial fetch window).  
It is worse latency-wise (retrieve → rerank → RF → rerank → merge by score, vs. retrieve → rerank one big pool) but can be better cost-wise if reranking a bigger pool still doesn't surface relevant documents.

**H2.** Pure RF is a viable cheap alternative to reranking, at least for one of the two main search use cases (e.g., RAG/agentic search, where better R@1/3 at far smaller cost and comparable latency matters).

#### Side Hypotheses

**H3.** An LLM (gpt-4o-mini) can be a good relevance classifier → reranker if properly configured (testing logprobs + binary, ordinal, binary with reasoning).

**H4.** Even if not (H3 fails), Pure RF or RF+Rerank may still give gains over baseline retrieval with an LLM as a feedback model, making it a sensible option for production retrieval (cost-latency-quality tradeoff-wise). For that, gains over baseline need to be truly meaningful.

**H5.** There is a good default context window size for RF methods, around 4–5, that works across different datasets.

**H6.** Retriever strength may affect the usability of RF methods.

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
Recall@[1, 3, 5, 10] measured by `ranx` + NDCG@10.
**Why Recall@1/3/5:** fewer retrieved chunks = more agent/LLM-positive method.
**Why nDCG@10:** enables comparison with the pyserini BEIR leaderboard of retrievers, just to check if results make sense at all, SciDocs nDCG@10 around 0.217 and FiQA nDCG@10 around 0.421.

Additionally, the **disc@N** metric can show how many relevant documents (according to qrels) were discovered and ranked highly enough to enter the @N window by RF-based methods — documents that were unreachable for the reranker (which can only reshuffle its initial retrieved pool).  
However, it wasn't considered in main experiments, as it's not a traditional IR metric but rather a custom one.

### Methods

| Method | Description |
|--------|-------------|
| **Baseline** | ANN retrieval, top-10 by embedding similarity |
| **Rerank** | Retrieve N docs, rescore all with feedback model, take top-10 |
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

Fraction of training queries where feedback model strongly disagrees with retriever ranking (feedback model scores a lower-ranked doc as more relevant than a higher-ranked one).

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

A feedback model disagreeing roughly half the time is the key signal: if the model is right in its relevance judgements — i.e., if it is a good reranker — there is a better chance that RF will improve results. If wrong, it is noise.

### Good Reranker (P) and LLM as a Reranker (H3)

#### P
openai-emb is a meaningful reranker.
It adds meaningful value on FiQA (both retrievers, +15–20% R@1) and SciDocs × jina (+15% R@1).
However, it adds almost nothing on SciDocs × mxbread — the mxbread baseline is already strong for similarity-based retrieval, and openai-emb is a better similarity measurer, not a better citation predictor (the task of SciDocs).
FiQA is more semantic-similarity-driven, so a stronger embedding model genuinely reranks better.

#### H3

**Results** (FiQA × mxbread, 200 eval queries, rerank_limit=25):

| Method | R@1 | R@3 | R@5 | R@10 | nDCG@10 | Est. cost/200q |
|--------|-----|-----|-----|------|---------|----------------|
| Baseline | 0.2344 | 0.3925 | 0.4439 | 0.5193 | 0.4767 | — |
| logprobs-reasoning rerank | -25% → 0.1747 | -11% → 0.3507 | -2% → 0.4349 | ~0% → 0.5178 | -9% → 0.4326 | $0.37 |
| logprobs rerank | -19% → 0.1898 | -1% → 0.3880 | +10% → 0.4865 | +10% → 0.5723 | -2% → 0.4685 | $0.16 |
| logprobs-ordinal rerank | -13% → 0.2032 | **+4% → 0.4093** | +7% → 0.4765 | +7% → 0.5576 | ~0% → 0.4747 | $0.22 |
| openai-emb rerank | **+15% → 0.2697** | **+19% → 0.4664** | **+19% → 0.5279** | **+17% → 0.6074** | **+17% → 0.5564** | $0.10 |

**Reasoning is not worth the cost, it even makes reranking worse**. Likely because the generated reasoning string commits the model to a direction before the boolean verdict, eliminating uncertainty → more polarized scores, not less.
**Ordinal vs plain logprobs — split result:** ordinal wins at R@1 (-13% vs -19%) and R@3 (+4% vs -1%). Plain logprobs wins at R@5/R@10 (+10% vs +7%) — though that gap is borderline given ~3% LLM non-determinism noise. For RF context selection (top 3–5 docs), ordinal may still be the better feedback model — it is the final choice for experiments.

**All of them are still worse than openai-emb** across all metrics. LLM-based reranking remains uncompetitive with embedding similarity for this task. Why?
- **Failure modes for logprobs/logprobs-reasoning:** *Polarization* — many docs score 1.0 (both irrelevant and relevant ones), so ordering within the group is arbitrary, destroying the ANN signal that would have placed more relevant docs at rank 1. Explains the R@1/R@3 drops.
- **Failure modes for all logprobs models:** *Mislabeling* — relevant docs are scored low and pushed below rank 10. The prompts are reasonably aligned with BEIR task definitions, so prompt engineering alone is unlikely to fix this.

**Note:**
**LLM as a reranker non-determinism:** It seems impossible to enforce determinism (temperature=0 is greedy but not bit-for-bit reproducible across calls). Practical implication: differences of less than ~3% between runs are noise, not signal.

### H1 — RF+Rerank > Rerank

Tested on a good reranker (openai-emb, which brings meaningful value improvement).

**FiQA results (ctx=5, 500q):** (grid_search_openai_emb_fiqa)

| | Rerank(25) | RF+Rerank(25) | Rerank(50) |
|--|-----------|--------------|-----------|
| mxbread R@1 | 0.2452 | 0.2455 | **0.2488** |
| mxbread R@3 | 0.4289 | **0.4348** | 0.4326 |
| mxbread R@5 | 0.4970 | **0.5042** | 0.5014 |
| mxbread R@10 | 0.5754 | 0.5852 | **0.5918** |
| mxbread nDCG | 0.5075 | 0.5138 | **0.5181** |
| jina R@1 | 0.2397 | 0.2376 | **0.2409** |
| jina R@3 | 0.4122 | 0.4128 | **0.4205** |
| jina R@5 | 0.4695 | 0.4780 | **0.4916** |
| jina R@10 | 0.5239 | 0.5431 | **0.5710** |
| jina nDCG | 0.4773 | 0.4841 | **0.4997** |

| cost/500q | $0.2437 | $0.2437 | $0.4875 |

Gains are small (+1–4%). R@1 effect is negligible or negative (jina).

RF+Rerank(25) approaches Rerank(50) at half the cost (mxbread −0.8%, jina −3.1%), but with **worse latency**: the openai-emb API is called twice (context scoring + RF doc scoring), making RF+Rerank slower than even Rerank(50) (~1260ms vs ~900ms for mxbread, ~1000ms vs ~840ms for jina).

**SciDocs results (ctx=5, 500q):** (grid_search_openai_emb_scidocs)

Note: openai-emb is a weaker reranker on SciDocs × mxbread, it adds almost nothing over baseline (see P).

| | Rerank(25) | RF+Rerank(25) | Rerank(50) |
|--|-----------|--------------|-----------|
| mxbread R@1 | **0.0548** | 0.0532 | 0.0536 |
| mxbread R@3 | 0.1280 | **0.1318** | 0.1289 |
| mxbread R@5 | 0.1837 | **0.1852** | 0.1811 |
| mxbread R@10 | **0.2706** | 0.2653 | 0.2632 |
| mxbread nDCG | **0.2500** | 0.2465 | 0.2446 |
| jina R@1 | 0.0532 | **0.0547** | 0.0523 |
| jina R@3 | 0.1267 | **0.1332** | 0.1288 |
| jina R@5 | 0.1752 | **0.1836** | 0.1779 |
| jina R@10 | 0.2488 | **0.2611** | 0.2556 |
| jina nDCG | 0.2355 | **0.2457** | 0.2391 |

| cost/500q | $0.3250 | $0.3250 | $0.6500 |

For jina, RF+Rerank(25) beats Rerank(25) by +3–5% at R@3–R@10 and nDCG, and edges Rerank(50) by +2–5% at half the cost (~920ms vs ~805ms — a latency penalty, but smaller than for mxbread). For mxbread, where openai-emb barely improves over baseline, RF+Rerank introduces noise: R@1, R@10, and nDCG all drop below Rerank(25).

#### Conclusion

The H1 pattern holds for jina across both datasets — RF+Rerank(25) ≈ Rerank(50) at half the cost, at a latency penalty. But gains are small (+2–5%). 
For mxbread, H1 is not meaningful on SciDocs (openai-emb adds too little to rerank from) and weak on FiQA (gains exist but tiny).
**This method, at least for these datasets, might make sense as a substitute for reranking only when even a small gain (+2–5%) brings a lot of value & latency is not a top priority.**

### H2 — Pure RF as cheap alternative to reranking

Tested on a good reranker (openai-emb, which brings meaningful value improvement).

Pure RF is 5× cheaper than Rerank(25) (scoring 5 context docs vs. 25/50 for reranking).  
Latency is comparable to or lower than reranking, depending on the feedback model.

**FiQA results (ctx=5, 500q):**

| | Baseline | Rerank(25) | Pure RF (ctx=5) |
|--|---------|-----------|----------------|
| mxbread R@1 | 0.2080 | **0.2452** | 0.2445 |
| mxbread R@3 | 0.3579 | **0.4289** | 0.4041 |
| mxbread R@5 | 0.4166 | **0.4970** | 0.4526 |
| mxbread R@10 | 0.5027 | **0.5754** | 0.5402 |
| mxbread nDCG | 0.4362 | **0.5075** | 0.4834 |
| jina R@1 | 0.1924 | **0.2397** | 0.2211 |
| jina R@3 | 0.3173 | **0.4122** | 0.3587 |
| jina R@5 | 0.3705 | **0.4695** | 0.4079 |
| jina R@10 | 0.4600 | **0.5234** | 0.4837 |
| jina nDCG | 0.3932 | **0.4770** | 0.4340 |

| cost/500q | — | $0.2437 | **$0.0488** |

For mxbread, Pure RF is essentially tied with Rerank(25) at R@1 (0.2445 vs 0.2452, −0.3% — within noise) at 5× less cost and comparable latency (947ms vs 918ms). At deeper cutoffs it falls behind: −6% R@3, −9% R@5, −6% R@10. For jina, the gap is larger: −8% R@1, −13% R@3. Both retrievers beat baseline substantially (+17% R@1 mxbread, +15% R@1 jina).

**SciDocs results (ctx=5, 500q):**

Note: as with H1, openai-emb adds little on SciDocs × mxbread as a reranker, so the comparison there is mainly Pure RF vs baseline.

| | Baseline | Rerank(25) | Pure RF (ctx=5) |
|--|---------|-----------|----------------|
| mxbread R@1 | 0.0531 | 0.0548 | **0.0568** |
| mxbread R@3 | 0.1246 | 0.1280 | **0.1314** |
| mxbread R@5 | 0.1794 | **0.1837** | 0.1732 |
| mxbread R@10 | 0.2611 | **0.2706** | 0.2318 |
| mxbread nDCG | 0.2419 | **0.2500** | 0.2305 |
| jina R@1 | 0.0451 | 0.0532 | **0.0553** |
| jina R@3 | 0.1065 | **0.1267** | 0.1245 |
| jina R@5 | 0.1556 | **0.1752** | 0.1699 |
| jina R@10 | 0.2208 | **0.2488** | 0.2376 |
| jina nDCG | 0.2055 | **0.2355** | 0.2289 |

| cost/500q | — | $0.3250 | **$0.0650** |

Pure RF wins at R@1 over both Rerank and baseline for both retrievers (+3.7% mxbread, +3.9% jina over Rerank(25)). Notably, Pure RF beats Rerank at R@1 on SciDocs × mxbread despite openai-emb being a weak reranker there — suggesting RF's HNSW traversal surfaces a different top-1 than reranking can reach. At R@3 it still beats Rerank on mxbread (+2.7%) but falls just below for jina (−1.7%). R@5 onward it drops behind — sharply for mxbread (−14% R@10).

#### Conclusion
Pure RF reorders the top well but trades off coverage at deeper cutoffs. It is a viable and cheap alternative to reranking for RAG/agentic use cases where R@1/R@3 is what matters — especially with a strong retriever (mxbread on FiQA). For traditional IR use cases requiring high R@5/R@10/nDCG, it is not a substitute for reranking.

For the conclusion, we also need to consider the training costs of RF (one training per dataset × retriever × feedback model triple), as reranking requires no training at all.

| Training | Cost |
|-----------|------|
| Per retriever — openai-emb (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.07 · SciDocs ~$0.13 |
| Per retriever — logprobs-ordinal (148q FiQA / 200q SciDocs × 25 docs) | FiQA ~$0.16 · SciDocs ~$0.27 |

But it's extremely cheap.

### Other findings

#### H4
**A bad reranker is not necessarily a bad feedback model.**

Tested on FiQA × {mxbread, jina} × logprobs-ordinal (ctx=5, rerank=25). logprobs-ordinal drops R@1 vs baseline as a reranker on both retrievers (−5% mxbread, −5% jina).

For mxbread, Pure RF with logprobs-ordinal still beats baseline at R@1 (+4.6%, 0.2176 vs 0.2080) and ties at R@3 — H4 supported for the RAG use case. For jina, Pure RF falls below baseline on all metrics — H4 not supported. The split mirrors H6: a stronger retriever tolerates a noisier feedback signal.

As a practical note: Pure RF with logprobs-ordinal is 5× cheaper ($0.1075 vs $0.5375) and ~4× faster (~4s vs ~17–22s) than reranking with it — making it the clearly preferable way to use this model if at all.

#### H5

**Recommendation: ctx=5.** Optimal or tied-best across FiQA and SciDocs with openai-emb and both retrievers. The exception is SciDocs × mxbread, where ctx=4 wins at R@1 and ctx=3 at nDCG for both RF methods — but RF adds almost nothing on that dataset regardless, so it is not a meaningful counter-signal.

- FiQA: ctx=5 wins on all 4 combinations by nDCG and R@3–R@10. R@1 is marginal: ctx=4 occasionally edges ctx=5 but differences are within noise.
- SciDocs × jina: ctx=5 is best for both Pure RF and RF+Rerank across all metrics.
- SciDocs × mxbread: ctx=4 is best at R@1 for Pure RF (0.0576) and RF+Rerank (0.0548); ctx=3 is best at nDCG. But RF adds almost nothing over baseline on this dataset and retriever regardless.

#### H6

**Retriever strength and RF:** A stronger retriever allows RF methods to close the gap with Rerank more effectively.

FiQA × openai-emb supports this: mxbread Pure RF is essentially tied with Rerank(25) at R@1 (−0.3%, nDCG gap −4.7%); jina trails by −7.8% R@1 and −9.0% nDCG. The logprobs-ordinal data adds the same signal: Pure RF beats baseline with mxbread (+4.6% R@1) but falls below it with jina.

SciDocs does not support it: at R@1 both retrievers beat Rerank by similar margins, and jina's nDCG gap to Rerank is smaller than mxbread's — the opposite of what H6 predicts.

More datasets would be needed to draw a reliable conclusion.

## Conclusion

**For now just raw thoughts**

These experiments form a comparison against rerankers (automated reranking; user-in-the-loop is out of scope) — essentially a second family of RF methods -- "changing scoring function". Query rewriting is not in scope & can be considered complementary, not competing.

**H1** is weakly supported: RF+Rerank matches Rerank on a 2× larger pool at half the cost, but gains are small (+2–5%) and latency is worse, making it worthwhile only when even marginal quality gains are highly valuable. **H2** is well supported: Pure RF is a viable, 5× cheaper alternative to reranking for RAG/agentic use cases where R@1/R@3 is what matters, especially with a strong retriever.

It seems like an LLM as a feedback model is not very beneficial, at least for these datasets: it is (1) a bad reranker and (2) not a good feedback model, as there are no meaningful benefits on RF-based method metrics either. A more interesting direction later is to add more semantic similarity BEIR datasets to the experiment pool using openai-emb as a reranker.

Would be nice to try: agent (Claude) using pure RF as a search tool, for example on qdrant skills.
Sketch: explain search skill → call this file ("is it answering your question?") → call this file ("out of the docs you saw, which looks more relevant and which less?" → judgements to hard scores) → call this file (RF) → "does it answer your question?". A result should be a comparison with query rewriting, in at least one call.

## Appendix

### Experiment Output Files

| File | Contents |
|------|----------|
| `rerank_search_limit25.txt` | Phase 1 rerank-only sweep. 8 runs: 2 datasets × 2 retrievers × {openai-emb, logprobs-reasoning}, rerank_limit=25, 200 eval queries each. Used to establish the best reranker & check how LLM with reasoning works as a reranker. |
| `grid_search_openai_emb_fiqa.txt` | Full grid search: FiQA × {mxbread, jina} × openai-emb. 12 experiments: 2 retrievers × rerank_limits={25,50} × rf_pairs={(3,22),(4,21),(5,20)}, 500 eval queries. |
| `grid_search_openai_emb_scidocs.txt` | Full grid search: SciDocs × {mxbread, jina} × openai-emb. 12 experiments: 2 retrievers × rerank_limits={25,50} × rf_pairs={(3,22),(4,21),(5,20)}, 500 eval queries. |
| `grid_search_logprobs_ordinal_fiqa.txt` | FiQA × {mxbread, jina} × logprobs-ordinal. 4 experiments: rerank={25,50} × rf_pairs={(5,20)}, 500 eval queries. |
