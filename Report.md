# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Name:** Madaka Narasinga Rao
**Roll No.:** 2025201011
**GitHub:** https://github.com/narsing64/RagAssignment

## Introduction

In a RAG (retrieval-augmented generation) system, we first search a document for the chunks that best match a question, and then give those chunks to an LLM, which writes the answer using only them. If the search picks the wrong chunks, the LLM cannot give the right answer, so retrieval quality matters a lot. In this assignment I compare two ways of choosing those chunks: a fast bi-encoder alone (baseline), and the same bi-encoder followed by a cross-encoder that re-ranks its results.

I used one long PDF: the **IIIT Hyderabad Students' Parliament Annual Report 2025–2026** (95 pages). It is full of rules, proposals, penalties and exceptions, so it is a good document for testing retrieval.

I removed the cover and contents pages (PDF pages 1–5), the header and page number repeated on every page, and the soft hyphens that split words across lines. Then I split the text into **266 chunks of 150 words**, with **30 words of overlap** between neighbouring chunks. The longest chunk has 253 tokens, so it fits inside the bi-encoder's 256-token limit and no text is cut off.

| Part | What I used |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-number vectors) |
| Vector store | FAISS `IndexFlatIP` (exact search; vectors are normalised, so inner product = cosine similarity) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini 2.5 Flash, temperature 0 |
| Retrieval | Top 10 chunks from the bi-encoder; 3 chunks given to the LLM |
| Prompt | Same for both pipelines: answer only from the given passages, cite them, otherwise reply "The provided context does not contain this information." |
| Hardware | Kaggle notebook with an NVIDIA Tesla T4 GPU |

Both pipelines use the same chunks, the same LLM and the same prompt. The only difference is **how the 3 chunks for the LLM are chosen**:

- **Pipeline 1 (baseline):** take the first 3 of the bi-encoder's top 10.
- **Pipeline 2 (re-ranking):** score the same top 10 again with the cross-encoder and take its first 3.

## 1. Architecture Comparison

**Bi-encoder.** The bi-encoder reads the question and each chunk *separately*. It turns each one into a single vector of 384 numbers, and relevance is the cosine similarity between the two vectors. A chunk's vector does not depend on the question, so all chunk vectors can be computed once, before any question is asked, and stored in FAISS. When a question comes in, only the question is encoded and the stored vectors are searched, which is very fast. The weakness is that a whole chunk is squeezed into one vector, and the question and the chunk never "see" each other inside the model. Small but important details, such as "not", "except", or whether a rule is only proposed, can get lost.

**Cross-encoder.** The cross-encoder reads the question and the chunk *together*, as one input: `[CLS] question [SEP] chunk [SEP]`. Inside the transformer, self-attention connects every word of the question with every word of the chunk, in every layer. The output is one relevance score for the pair, not a vector. This lets the model check whether the chunk really answers the question. The cost is that nothing can be computed in advance: the model has to run again for every (question, chunk) pair.

**Example from my results.** For Query 1 ("maximum working days of leave per semester"), the bi-encoder ranked three chunks about a *proposed* new leave rule highest, because they use the same words ("leave", "working days", "limit"). The chunk with the rule that is actually in force (chunk 40: "up to 18 working days of leave in a semester") was only 7th. The cross-encoder moved it to 1st. The scores tell the same story. For an easy test question, the bi-encoder gave the right chunk 0.54 and the next chunk 0.38, a small gap. The cross-encoder gave the right chunk +4.46 and every other chunk between −9 and −11, a very clear gap.

Both models use the same small 6-layer MiniLM network, so the difference comes from *how* they read the text, not from model size.

## 2. Qualitative & Rank Shift Analysis

I wrote six test questions, one or more for each failure type in the assignment. For each question, the **gold chunk** is the chunk that contains the true answer. I found it by searching the chunks for the exact sentence from the PDF.

### 2.1 Rank shifts

| # | Failure type | Question | Gold chunk (PDF page) | Rank: bi-encoder → cross-encoder | Gold in top 3? P1 / P2 |
|---|---|---|---|---|---|
| 1 | Keyword trap | Maximum working days of leave per semester | 40 (p. 18–19) | #7 → **#1** | No / Yes |
| 2 | Exception (penalty) | Backing out of an approved event without a valid reason | 22 (p. 13) | #8 → **#1** | No / Yes |
| 3 | Exception ("except" clause) | Leave for representing the Institute at external events | 21 (p. 12–13) | #5 → **#1** | No / Yes |
| 4 | Negation | Has any water softener been installed? | 118 (p. 44–45) | #8 → **#2** | No / Yes |
| 5 | Buried chunk (numbers) | Current monthly stipend for a Quarter TA | 47 (p. 21) | #7 → **#2** | No / Yes |
| 6 | Limit of re-ranking | Was weekly room cleaning introduced? | 109 (p. 41–42) | #16 → not re-ranked | No / No |

### 2.2 Answer 1 vs. Answer 2

"Not in context" means the LLM read the 3 chunks it was given and replied "The provided context does not contain this information." Our prompt tells the LLM to give exactly this reply when it cannot find the answer in those chunks, instead of guessing.

| # | Answer 1 (baseline) | Answer 2 (re-ranked) | Verdict |
|---|---|---|---|
| 1 | Not in context | "Up to 18 working days of leave in a semester" | A1 omission · A2 correct |
| 2 | Not in context | "Reduced support for the next event they enter" | A1 omission · A2 correct |
| 3 | Not in context | "Up to 7 working days … with exceptions for events of high importance", plus two more correct details | A1 omission · A2 correct, but does not say it is a draft |
| 4 | Not in context | "No softener has been installed" | A1 omission · A2 correct |
| 5 | Not in context | Not in context | Both omissions, even though P2 had the gold chunk |
| 6 | "Weekly room cleaning would be possible with more staff" | Same as Answer 1 | Both incomplete: neither says "no" |

**Why the baseline missed the gold chunk:**

- **Query 1 (keyword trap):** the top 3 were about a *proposed* 6-day leave limit (Section 3.18) that is on hold. They share the question's words but describe a proposal, not the rule in force.
- **Query 2 (penalty):** the top 3 were about malpractice penalties (Section 2.10), delays in processing leave (Section 3.8) and Dean's List rules (Section 3.1). None of them is about events. The cross-encoder gave the real penalty sentence +7.6 and every other candidate −1.9 or less, while the bi-encoder's scores for the same ten chunks were all between 0.39 and 0.46.
- **Query 3 (except clause):** the top 3 had two chunks about the proposed leave limit and chunk 20, which starts the correct policy but is cut off at "A student can receive up to 7" by the chunk boundary. The number and the exception were in the next chunk (chunk 21), which was 5th.
- **Query 4 (negation):** the top 3 described the softener *plans* ("would be installed … for a 15-day trial", "put the plans on hold"). The bi-encoder could not tell "planned" apart from "not installed". The chunk saying "no softener has been installed" was 8th.
- **Query 5 (numbers):** the top 3 had the old stipend (Rs. 6,000) and the Parliament's demands (Rs. 8,000). The announced amount, Rs. 7,500, was 7th.

**Where re-ranking was not enough:**

- **Query 5:** re-ranking moved the gold chunk to 2nd, so the chunk saying "Rs. 7,500 for a Quarter" was one of the 3 chunks given to the LLM. Even so, the LLM replied "not in context". The same 3 chunks also contained the Parliament's demand of Rs. 8,000 "from Monsoon 2026", and the word "current" does not appear in the text. With two different amounts for the same period, the LLM most likely could not tell which one is the current stipend, so it gave the "not in context" reply instead of picking one. So good retrieval is needed, but it does not always guarantee a good answer.
- **Query 3:** Answer 2 is correct, but it does not mention that the policy is still a draft. That fact is in chunk 22, which was not in the top 3.
- **Query 6:** the gold chunk was 16th, so it never reached the cross-encoder. Both answers repeat a related statement but never say "no".

**Did the baseline hallucinate, leave out a clause, or misstate a rule?** In my runs, the baseline never hallucinated and never stated a wrong rule. Every baseline failure was an **omission**: five times it replied "not in context", and once it gave an incomplete answer. I think this is because of the strict prompt, which tells the LLM to give the "not in context" reply instead of guessing when the answer is not in the 3 chunks. Re-ranking brought the gold chunk into the top 3 for **5 of the 6** questions and turned **4 "not in context" replies into correct answers** (Queries 1–4).

## 3. Latency Profiling

I timed the retrieval part of both pipelines on the T4 GPU for 8 questions. Each measurement was run once as a warm-up and then 20 more times, and the table shows the average. I did not repeat the LLM call in this test, because it is the same for both pipelines and the free API allows only a few calls per day. The LLM time is the average of the 12 LLM calls from the failure tests.

| Stage | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Encode question + FAISS search | 6.4 ms | 6.4 ms |
| Cross-encoder on 10 pairs | — | 26.1 ms |
| **Retrieval total** | **6.4 ms** | **32.5 ms** (5.1×) |
| LLM answer (average) | 0.61 s | 0.61 s |
| **Total time per question (approx.)** | **0.62 s** | **0.64 s** |

Re-ranking makes retrieval about **5 times slower** (32.5 ms instead of 6.4 ms). But that is only **26 ms extra**, while the LLM call takes about 610 ms, so the full answer is only about **4% slower**. In return, re-ranking fixed 4 of the 6 failure questions. Building the index (embedding all 266 chunks) took 0.89 s, but that is done only once, not for every question.

## 4. Scaling Question

**Why not run the cross-encoder on every chunk from the start?** Because the cross-encoder cannot prepare anything in advance. Its score depends on the question and the chunk together, so for every new question it has to run the full model once for every chunk. Its total cost grows with the number of chunks, O(N), and each run also gets slower for longer inputs, because self-attention costs O((L_q + L_d)²) for a question of length L_q and a chunk of length L_d. The bi-encoder is different: it embeds all chunks once, and then each question needs only one encoding and a quick index search.

I measured how the cross-encoder's time grows with the number of chunks it scores (k):

| Chunks scored (k) | Time | Time per chunk |
|---|---|---|
| 10 | 24.5 ms | 2.45 ms |
| 50 | 122.4 ms | 2.45 ms |
| 100 | 306.7 ms | 3.07 ms |
| 266 (all chunks) | 857.8 ms | 3.22 ms |

Even on a GPU, each chunk costs about 2.5–3.2 ms, so the time grows almost in a straight line with k. Scoring all 266 chunks took 857.8 ms: about **35 times** the cost of re-ranking 10 chunks and **134 times** the bi-encoder search. At about 3.2 ms per chunk, a collection of 100,000 chunks would need about **5.4 minutes per question**, and 1,000,000 chunks about **54 minutes per question**. That is far too slow for a real system.

That is why we use two stages. The fast bi-encoder picks a small set of likely chunks from the whole collection, and the slower but more accurate cross-encoder only re-orders that small set. The size of the set, k, is a trade-off between finding the answer and speed: k = 10 took 24.5 ms, k = 50 took 122.4 ms and k = 100 took 306.7 ms.

The two-stage design has one weakness: **the cross-encoder can only re-order what the bi-encoder found.** In Query 6 the gold chunk was 16th, so it was never re-ranked and both pipelines failed. When I let the cross-encoder score all 266 chunks, it put that chunk 2nd, but at about 35 times the cost. So k must be large enough that the right chunk is usually inside the top k.

## Conclusion

Cross-encoder re-ranking clearly improved the answers to hard questions (keyword traps, exceptions, negation and buried numbers) for only about 26 ms extra per question. It cannot recover chunks that the first stage misses, and better retrieval does not always mean the LLM gives a better answer.
