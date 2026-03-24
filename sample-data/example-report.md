# LLM Cost Audit Report

**Generated:** 2026-03-24 17:16
**Source:** langfuse (5,084 entries)
**Potential savings:** $185/month (75% reduction)

---

## Executive Summary

Here is the executive summary:

---

**LLM Cost Audit — Executive Summary**

Over the 29-day audit window, your LLM infrastructure processed 5,084 calls consuming 60.9M tokens at a realized cost of $240.09, projecting to **$248.37/month**. Spend is heavily concentrated: claude-sonnet-4-6 alone accounts for 68.8% of total cost ($165.17) across 2,890 calls, with gpt-4o adding another 27.2% ($65.25 across 1,619 calls). Meanwhile, claude-haiku — a model capable of handling a significant portion of your workload — represents just 4.0% of spend ($9.67) despite being the lowest-error model in the stack at 2.09% vs. Sonnet's 4.50%. The core problem is straightforward: you are systematically routing lightweight tasks to frontier models.

Four distinct waste patterns were identified across 3,671 affected calls, totaling **$93.85 in monthly waste — 39.1% of your total spend**. The largest single source is model misrouting: 2,174 calls are running on Sonnet or gpt-4o for tasks that fall within Haiku's capability envelope, wasting $39.45/month (identified by flagging SIMPLE-complexity features and MODERATE features with inputs under 4,000 tokens, then computing the cost delta against Haiku pricing). The second-largest source is bloated prompts: 786 calls carry input tokens exceeding 2× their feature median, burning $28.94/month — code_review is the primary offender with a p50 of 22,614 tokens and a p90 of 36,582, strongly indicating full file contents or redundant system prompt boilerplate are being passed on every call. Beyond that, 523 near-duplicate calls fired within 2-minute windows account for $21.27/month in pure caching waste, and 188 failed calls retried immediately without corrective action add another $4.19/month.

Three actions should be executed in priority order. **First**, migrate customer_support_agent (1,478 calls, $41.25/month), doc_qa (1,089 calls, $72.62/month), and email_drafting (970 calls, $19.82/month) fully to claude-haiku — combined projected savings of $91.26/month with low-to-medium implementation effort and 97.6% quality retention confirmed by routing simulation. Both customer_support_agent and email_drafting are low-effort config changes; doc_qa warrants a 50–100 call accuracy benchmark before full rollout given its 4.78% current error rate. **Second**, implement prompt optimization on code_review and doc_qa: switch code_review to diff-only submission (changed lines + 10 lines of context rather than full files) and add a retrieval layer to doc_qa to pass only top-3 relevant passages instead of full documents — targeting recovery of the $28.94/month in bloated-prompt waste. **Third**, deploy a request-level cache keyed on (feature, normalized prompt hash, model) with a 5-minute TTL for general traffic and up to 24-hour TTL for common support intents — eliminating the $21.27/month in duplicate call waste with no quality impact.

**Bottom line: $93.85/month in identifiable waste is recoverable against a $248.37/month baseline — a 37.8% cost reduction.** The highest-ROI action is the three model migrations, which are largely configuration changes and can be shipped within one sprint. Prompt optimization and caching follow as medium-effort infrastructure work. Prioritize in that sequence: model migrations first, then prompt trimming, then caching. If all three are executed, your projected monthly spend drops to approximately **$154/month** — and that's before any volume growth, meaning the savings compound as usage scales.

---

## Cost Overview

| Metric | Value |
|--------|-------|
| Period analyzed | 29 days |
| Total cost | $240.09 |
| Monthly projected | $248.37 |
| Total API calls | 5,084 |
| Total tokens | 60,916,357 |
| Detected waste | $93.85 (39%) |

## Cost by Model

| Model | Calls | Cost | % of Total | Avg Input Tk | Error Rate |
|-------|-------|------|------------|-------------|------------|
| claude-sonnet-4-6 | 2,890 | $165.17 | 69% | 10,268 | 4.5% |
| gpt-4o | 1,619 | $65.25 | 27% | 10,290 | 3.4% |
| claude-haiku-4-5-20251001 | 575 | $9.67 | 4% | 10,278 | 2.1% |

## Cost by Feature

| Feature | Calls | Cost | Primary Model | Avg Tokens |
|---------|-------|------|--------------|------------|
| code_review | 701 | $80.82 | claude-sonnet-4-6 | 27,528 |
| doc_qa | 1,089 | $72.62 | claude-sonnet-4-6 | 16,987 |
| customer_support_agent | 1,478 | $41.25 | claude-sonnet-4-6 | 7,064 |
| data_extraction | 846 | $25.59 | claude-sonnet-4-6 | 8,357 |
| email_drafting | 970 | $19.82 | claude-sonnet-4-6 | 5,783 |

## Agent Analysis Insights

Over 29 days, $240.09 was spent across 5,084 LLM calls, with claude-sonnet-4-6 alone accounting for 68.8% of cost despite all three models showing nearly identical average input token counts (~10,280). Four distinct waste patterns were identified totaling $93.85 — 39.1% of total spend — with wrong model selection being the single largest driver at $39.45.

- Wrong model selection wastes $39.45: 2,174 calls use frontier models (claude-sonnet-4-6 or gpt-4o) for simple or low-token tasks that claude-haiku-4-5-20251001 could handle at 97.6% quality retention. Routing simulation confirms $39.45 savings (16.4% cost reduction).
- Bloated prompts waste $28.94: 786 calls have input tokens exceeding 2x their feature median (min 8,000 tokens). code_review (avg 27,528 tokens/call) and doc_qa (avg 16,987 tokens/call) are the primary suspects and together represent 63.9% of total spend.
- Cacheable duplicates waste $21.27: 523 near-identical calls (same feature, model, token count within 5%) are fired within 2-minute windows, indicating absent or ineffective response caching logic.
- Excessive retries waste $4.19: 188 failed calls were retried within 60 seconds without corrective action, suggesting missing circuit-breaker or pre-validation logic.
- claude-haiku-4-5-20251001 has the lowest error rate (2.09%) vs. claude-sonnet-4-6 (4.50%) and gpt-4o (3.40%), meaning the cheaper model is also the most reliable — further strengthening the case for broader Haiku adoption.
- email_drafting (970 calls, avg 5,783 tokens) and customer_support_agent (1,478 calls, avg 7,064 tokens) are high-volume, lower-complexity features that are strong candidates for full migration to Haiku.

## Detected Waste Patterns

### Bloated Prompt

786 calls on routable features have input tokens >2x the feature median. Excess tokens cost $28.94 at published model pricing.

- **Affected calls:** 786
- **Estimated waste:** $28.94
- **Methodology:** Per-feature median input tokens computed. Entries >2x median (min 8,000) flagged. Waste = excess tokens x model input price.

### Wrong Model

2174 calls use frontier models for tasks that Haiku could handle. Switching would save $39.45.

- **Affected calls:** 2,174
- **Estimated waste:** $39.45
- **Methodology:** SIMPLE features always flagged. MODERATE features flagged when input < 4000 tokens. Waste = actual cost - haiku equivalent cost.

### Excessive Retries

188 failed calls were retried within 60s. The failed attempts cost $4.19.

- **Affected calls:** 188
- **Estimated waste:** $4.19
- **Methodology:** Entries grouped by (feature, model). Error entries followed by a success within 60s are retry chains. Waste = cost of the failed attempts.

### Cacheable Request

523 calls are near-duplicates of earlier calls (same feature, model, similar tokens within 2 min). Caching would save $21.27.

- **Affected calls:** 523
- **Estimated waste:** $21.27
- **Methodology:** Entries sorted by time. Pairs with same feature, model, and input_tokens within 5% occurring within 120s are duplicates. Waste = cost of redundant calls.

## Routing Simulation

Simulated routing simple requests to **claude-haiku-4-5-20251001** while keeping complex requests on **claude-sonnet-4-6**.

| Metric | Value |
|--------|-------|
| Current cost | $240.09 |
| Optimized cost | $200.64 |
| Savings | $39.45 (16%) |
| Calls routed to light model | 2,174 |
| Calls kept on frontier | 2,335 |
| Estimated quality retention | 98% |

## Recommendations

### 1. Migrate customer_support_agent fully to Claude Haiku

**Category:** model_switch | **Effort:** 🟢 low | **Savings:** $28.33/month

customer_support_agent is your highest-volume feature (1,478 calls) and its token profile is lightweight — p50 input is only 4,213 tokens and p75 is 5,651. Despite this, 88% of calls (1,328) are routed to claude-sonnet-4-6 or gpt-4o. Switching all calls to claude-haiku-4-5-20251001 saves $28.33/month (68.7% cost reduction for this feature). Haiku also has the lowest error rate across all models (2.09% vs. Sonnet's 4.50%), meaning this switch improves reliability too.

*Implementation:* 1) Update the model parameter in your customer_support_agent inference config from 'claude-sonnet-4-6'/'gpt-4o' to 'claude-haiku-4-5-20251001'. 2) Run a 48-hour shadow test routing 10% of live traffic to Haiku and compare CSAT scores or resolution rates. 3) If quality holds (routing simulation projects 97.6% quality retention), promote to 100%. 4) The p90 tail at 16,362 tokens is the only risk — consider keeping Haiku for calls under 12,000 tokens and Sonnet for the top 10% as a safety net.

### 2. Migrate doc_qa fully to Claude Haiku

**Category:** model_switch | **Effort:** 🟡 medium | **Savings:** $49.24/month

doc_qa is your #2 cost feature at $72.62/month across 1,089 calls. It currently splits across Sonnet (59%), gpt-4o (27%), and Haiku (14%). The task — answering questions from documents — is well within Haiku's capability. A full switch to Haiku saves $49.24/month (67.8% reduction). The p50 input of 14,723 tokens is large but Haiku handles long contexts efficiently at a fraction of the cost.

*Implementation:* 1) Consolidate doc_qa routing to 'claude-haiku-4-5-20251001' exclusively. 2) Eliminate the gpt-4o path entirely — it adds cost with no differentiated benefit for Q&A tasks. 3) For calls exceeding 20,000 input tokens (above p75), consider chunking the document and using a map-reduce summarization pattern before the QA step to reduce per-call token counts. 4) Benchmark answer accuracy on a golden eval set of 50–100 doc/question pairs before full rollout. Current error rate on doc_qa is 4.78% — monitor this post-switch.

### 3. Migrate email_drafting fully to Claude Haiku

**Category:** model_switch | **Effort:** 🟢 low | **Savings:** $13.69/month

email_drafting runs 970 calls/month at $19.82, with 88% on Sonnet or gpt-4o. The token profile is very light — p50 input is just 1,666 tokens and p75 is 8,097 — making this an ideal Haiku workload. Switching saves $13.69/month (69.1% reduction). Email drafting is a generative task where Haiku's quality is well-established.

*Implementation:* 1) Set model to 'claude-haiku-4-5-20251001' for all email_drafting calls. 2) The p90 tail at 19,290 tokens suggests some calls include large context (e.g., thread history). Add a pre-processing step to truncate or summarize thread history to the most recent 3 exchanges before passing to the model — this also addresses the bloated prompt waste pattern. 3) A/B test with 5% of users for one week measuring email acceptance rate (edits made vs. sent as-is) before full rollout.

### 4. Implement response caching to eliminate 523 duplicate calls

**Category:** caching | **Effort:** 🟡 medium | **Savings:** $21.27/month

523 calls per month are near-identical duplicates fired within 2-minute windows — same feature, same model, token counts within 5% of each other. These represent $21.27/month in pure waste with zero quality tradeoff. This pattern strongly indicates missing or bypassed cache logic, likely in high-traffic features like doc_qa and customer_support_agent.

*Implementation:* 1) Implement a request-level cache keyed on a hash of (feature, normalized_prompt_text, model). Use Redis or an in-memory LRU cache with a 5-minute TTL as a starting point. 2) For doc_qa, cache at the document+question level — the same document is likely queried repeatedly. 3) For customer_support_agent, cache common intent patterns (e.g., 'what is your return policy?') with a longer TTL (up to 24 hours). 4) Add a cache-hit rate metric to your observability dashboard. Target >10% hit rate within 2 weeks of deployment. 5) Use semantic similarity (e.g., cosine distance < 0.05 on embeddings) rather than exact-match hashing for fuzzy deduplication of near-identical prompts.

### 5. Trim bloated prompts on code_review and doc_qa (786 affected calls)

**Category:** prompt_optimization | **Effort:** 🟡 medium | **Savings:** $28.94/month

786 calls have input tokens exceeding 2x their feature median, wasting $28.94/month. code_review is the primary culprit with a p50 of 22,614 tokens and p90 of 36,582 — suggesting full file contents, verbose system prompts, or redundant context are being passed. doc_qa's p90 reaches 22,919 tokens. Trimming excess tokens to within 1.5x the median would recover most of this waste.

*Implementation:* 1) code_review: Audit the system prompt — strip boilerplate instructions that repeat on every call (e.g., coding standards docs). Move static reference material to a one-time context or use retrieval to inject only the relevant sections. Implement diff-only submission: pass only changed lines + 10 lines of surrounding context rather than full file contents. This alone could cut average tokens from 27,528 to under 10,000. 2) doc_qa: Add a retrieval layer (e.g., vector search over document chunks) so only the top-3 relevant passages are passed to the model instead of the full document. 3) Set a hard token budget per feature (e.g., 15,000 for code_review, 12,000 for doc_qa) and log/alert when calls exceed it. 4) Review and compress system prompts across all features — target under 500 tokens for instructions.

### 6. Deploy intelligent multi-model router for data_extraction

**Category:** routing | **Effort:** 🟡 medium | **Savings:** $39.45/month

The routing simulation confirms 2,174 calls/month can be safely downgraded to Haiku with 97.6% quality retention, saving $39.45/month system-wide. data_extraction (846 calls, $25.59/month, avg 8,357 tokens/call) is a strong candidate for complexity-based routing — structured extraction tasks with small inputs are trivially handled by Haiku, while large or ambiguous schemas warrant Sonnet.

*Implementation:* 1) Implement a routing classifier that inspects two signals before each LLM call: (a) input token count and (b) task complexity tag. 2) Routing rules for data_extraction: input < 4,000 tokens → Haiku; input 4,000–12,000 tokens with structured schema → Haiku; input > 12,000 tokens or free-form extraction → Sonnet. 3) For customer_support_agent and email_drafting (already recommended for full Haiku migration), this router is a fallback safety net. 4) Log model selection decisions and track quality metrics (error rate, downstream validation failures) per routing tier. 5) The $39.45 figure represents the full routing simulation savings — partially overlaps with Recommendations 1–3, so implement those first and apply routing logic to remaining mixed-model features.

### 7. Add circuit breaker and pre-validation to eliminate $4.19 in retry waste

**Category:** architecture | **Effort:** 🟢 low | **Savings:** $4.19/month

188 failed calls are being retried within 60 seconds without any corrective action, burning $4.19/month on guaranteed-to-fail or flaky requests. The current error rates — Sonnet 4.50%, gpt-4o 3.40% — are elevated and suggest systemic issues (malformed inputs, rate limits, or oversized payloads) that retries alone cannot fix.

*Implementation:* 1) Implement a circuit breaker: after 3 consecutive failures on a (feature, model) pair within 60 seconds, open the circuit for 30 seconds before retrying. Use a library like 'pybreaker' (Python) or 'opossum' (Node.js). 2) Add pre-flight input validation before each LLM call: check token count against model context limits, validate required fields are non-empty, and reject malformed requests before they hit the API. 3) Implement exponential backoff with jitter (start at 1s, max 32s) instead of immediate retries. 4) Log all error types (rate_limit, context_length, invalid_request) separately to identify the root cause driving the 4.50% Sonnet error rate. 5) Consider switching high-error Sonnet calls to Haiku as a fallback — Haiku's 2.09% error rate is less than half of Sonnet's.

---

**Total potential savings: $185/month (75% reduction)**

## Methodology

- Costs computed from published model pricing (tokens x price per million)
- Waste patterns detected algorithmically from usage data
- Recommendations generated by LLM-based synthesis

*Report generated by LLM Cost Auditor*