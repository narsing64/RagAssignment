# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Name:** Madaka Narasinga Rao
**Roll No.:** 2025201011
**GitHub:** https://github.com/narsing64/RagAssignment

## Introduction

In a RAG (retrieval-augmented generation) system, we first search a document for the chunks that best match a question, and then an LLM writes the answer using only those chunks. If the search picks the wrong chunks, the answer will be wrong. This assignment compares two ways of choosing the chunks: a bi-encoder alone (baseline), and the same bi-encoder followed by a cross-encoder that re-ranks its results.

I used the **IIIT Hyderabad Students' Parliament Annual Report 2025–2026** (95 pages), which is full of rules, proposals, penalties and exceptions. After removing the cover and contents pages (PDF pages 1–5), the repeated page headers and the soft hyphens, I split the text into **266 chunks of 150 words** with a **30-word overlap**. The longest chunk has 253 tokens, within the bi-encoder's 256-token limit.

| Part | What I used |
|---|---|
| Bi-encoder + vector store | `all-MiniLM-L6-v2` (384-number vectors) + FAISS `IndexFlatIP` (cosine similarity on normalised vectors) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM and prompt | Gemini 2.5 Flash, temperature 0. Same prompt for both pipelines: answer only from the given chunks and cite them, otherwise reply "The provided context does not contain this information." |
| Pipeline 1 (baseline) | Top 10 chunks by cosine similarity → first 3 given to the LLM → **Answer 1** |
| Pipeline 2 (re-ranking) | Same top 10 → re-scored by the cross-encoder → new top 3 given to the LLM → **Answer 2** |
| Hardware | Kaggle notebook, NVIDIA Tesla T4 GPU |

## 1. Architecture Comparison

**Bi-encoder.** The bi-encoder reads the question and each chunk *separately*. It turns each one into a single vector of 384 numbers, and relevance is the cosine similarity between the two vectors. A chunk's vector does not depend on the question, so all chunk vectors can be computed once, before any question is asked, and stored in FAISS. When a question comes in, only the question is encoded and the stored vectors are searched, which is very fast. The weakness is that a whole chunk is squeezed into one vector, and the question and the chunk never "see" each other inside the model. Small but important details, such as "not", "except", or whether a rule is only proposed, can get lost.

**Cross-encoder.** The cross-encoder reads the question and the chunk *together*, as one input: `[CLS] question [SEP] chunk [SEP]`. Inside the transformer, self-attention connects every word of the question with every word of the chunk, in every layer. The output is one relevance score for the pair, not a vector. This lets the model check whether the chunk really answers the question. The cost is that nothing can be computed in advance: the model has to run again for every (question, chunk) pair.

**Example from my results.** For Query 1 ("maximum working days of leave per semester"), the bi-encoder ranked three chunks about a *proposed* new leave rule highest, because they use the same words ("leave", "working days", "limit"). The chunk with the rule that is actually in force (chunk 40: "up to 18 working days of leave in a semester") was only 7th. The cross-encoder moved it to 1st. The scores tell the same story. For an easy test question, the bi-encoder gave the right chunk 0.54 and the next chunk 0.38, a small gap. The cross-encoder gave the right chunk +4.46 and every other chunk between −9 and −11, a very clear gap.

Both models use the same small 6-layer MiniLM network, so the difference comes from *how* they read the text, not from model size.

## 2. Qualitative & Rank Shift Analysis

I wrote six test questions covering the failure types in the assignment. The **gold chunk** is the chunk that contains the true answer; I found it by searching for the exact sentence from the PDF. **"Not in context"** means the LLM read the 3 chunks it was given and replied "The provided context does not contain this information." Our prompt asks for this reply when the answer is not in the chunks, and it counts as an **omission**.

### 2.1 Rank shifts

| # | Failure type | Gold chunk (PDF page) | Rank: bi-encoder → cross-encoder | Answer 1 (baseline) | Answer 2 (re-ranked) |
|---|---|---|---|---|---|
| 1 | Keyword trap | 40 (p. 18–19) | #7 → **#1** | Omission | Correct |
| 2 | Exception (penalty) | 22 (p. 13) | #8 → **#1** | Omission | Correct |
| 3 | Exception ("except" clause) | 21 (p. 12–13) | #5 → **#1** | Omission | Correct |
| 4 | Negation | 118 (p. 44–45) | #8 → **#2** | Omission | Correct |
| 5 | Buried chunk (numbers) | 47 (p. 21) | #7 → **#2** | Omission | Omission |
| 6 | Limit of re-ranking | 109 (p. 41–42) | #16 → not re-ranked\* | Incomplete | Incomplete |

\* Outside the bi-encoder's top 10, so the cross-encoder never scored it. Scoring all 266 chunks ranks it 2nd (see Section 4).

### 2.2 Answer 1 vs. Answer 2

For Queries 1–5, Answer 1 was "not in context", because the gold chunk was not in the baseline's top 3.

**Query 1 (keyword trap):** *"What is the maximum number of working days of leave a student can take in a semester?"* True answer: 18 working days (Section 3.6, p. 19). The baseline gave the LLM chunks 60, 62 and 63, which describe a *proposed* 6-day leave limit that is on hold (Section 3.18). They share the question's words, so the bi-encoder ranked them first. After re-ranking (top 3: **40**, 21, 61), Answer 2 was "A student can take up to 18 working days of leave in a semester. [1]" → **correct**.

**Query 2 (exception, penalty):** *"What happens to a student who backs out of an approved event without a valid reason?"* True answer: reduced support for the next event (Section 2.8, p. 13). The baseline's chunks (25, 41, 32) are about malpractice penalties, delays in processing leave and Dean's List rules, none of them about events. All ten bi-encoder scores were close (0.39–0.46), and the penalty sentence (chunk 22) was 8th. The cross-encoder gave it +7.6 and every other chunk −1.9 or less (top 3: **22**, 41, 33). Answer 2: "…receives reduced support for the next event they enter. [1]" → **correct**.

**Query 3 (exception, "except" clause):** *"How many days of leave can a student get for representing the Institute at external events?"* True answer: up to 7 working days per semester, with exceptions for events of high importance (a draft policy, Section 2.8). The baseline got two proposal chunks (60, 62) and chunk 20, which starts the right policy but is cut off by the chunk boundary at "A student can receive up to 7". After re-ranking (top 3: **21**, 20, 62), Answer 2 gave the 7 days *and* the exception clause → **correct**, although it does not say the policy is still a draft (that fact is in chunk 22).

**Query 4 (negation):** *"Has any water softener been installed in the hostels?"* True answer: no (Section 6.4, p. 45). The baseline's chunks (116, 117, 115) describe the *plans* ("would be installed … for a 15-day trial", "put the plans on hold"). The bi-encoder could not separate "planned" from "not installed", so the chunk with the outcome (118) was 8th. After re-ranking (top 3: 116, **118**, 117), Answer 2 was "No softener has been installed. [2]" → **correct**.

**Query 5 (buried chunk, numbers):** *"What is the current monthly stipend for a Quarter Teaching Assistant?"* True answer: Rs. 7,500 from Monsoon 2026 (Section 3.13, p. 21). The baseline's chunks (49, 51, 44) hold the old stipend (Rs. 6,000) and the Parliament's demands (Rs. 8,000); the announced amount (chunk 47) was 7th. Re-ranking moved chunk 47 into the top 3 (49, **47**, 48), but Answer 2 was still "not in context". The same chunks also contain the Rs. 8,000 demand "from Monsoon 2026", and the word "current" does not appear in the text, so the LLM most likely could not decide between the two amounts. Retrieval was fixed, but the answer was not.

**Query 6 (limit of re-ranking):** *"Was weekly room cleaning introduced in the hostels?"* True answer: no (Section 6.1, p. 41). The gold chunk (109) was 16th, outside the top 10, so the cross-encoder never saw it. Both answers only said that "weekly room cleaning would be possible with more staff" and never answered "no" → **incomplete**.

**What this shows.** The baseline never hallucinated or stated a wrong rule: all its failures were omissions (five "not in context" replies and one incomplete answer), most likely because the strict prompt tells the LLM not to guess. Re-ranking brought the gold chunk into the top 3 for 5 of the 6 questions and gave correct answers for 4 of them (Queries 1–4). It has two limits: good retrieval does not always give a good answer (Query 5), and it cannot fix a chunk that the bi-encoder did not retrieve (Query 6).

## 3. Latency Profiling

I timed retrieval on the T4 GPU for 8 questions (1 warm-up run, then the average of 20 runs). The LLM call is the same for both pipelines, so I did not repeat it (the free API allows only a few calls per day); its time is the average of the 12 LLM calls in the failure tests.

| Stage | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Encode question + FAISS search | 6.4 ms | 6.4 ms |
| Cross-encoder on 10 pairs | — | 26.1 ms |
| **Retrieval total** | **6.4 ms** | **32.5 ms** (5.1×) |
| LLM answer (average) | 0.61 s | 0.61 s |
| **Total per question (approx.)** | **0.62 s** | **0.64 s** |

Re-ranking makes retrieval about **5 times slower**, but that is only **26 ms extra** next to a 610 ms LLM call, so the full answer is only about **4% slower**, and it fixed 4 of the 6 failure questions. Building the index (embedding all 266 chunks) took 0.89 s, but this is done only once.

## 4. Scaling Question

The cross-encoder cannot prepare anything in advance, because its score depends on the question and the chunk together. So every new question needs one full model run per chunk: the cost grows linearly with the number of chunks, O(N), and each run costs O((L_q + L_d)²) because of self-attention. The bi-encoder embeds all chunks once, and each question then needs only one encoding and a fast index search.

| Chunks scored (k) | 10 | 50 | 100 | 266 (all) |
|---|---|---|---|---|
| Cross-encoder time | 24.5 ms | 122.4 ms | 306.7 ms | 857.8 ms |
| Time per chunk | 2.45 ms | 2.45 ms | 3.07 ms | 3.22 ms |

Each chunk costs about 2.5–3.2 ms even on a GPU, so the time grows almost linearly with k. Scoring all 266 chunks took 857.8 ms, about **35 times** the 10-chunk re-ranking and **134 times** the bi-encoder search. At about 3.2 ms per chunk, 100,000 chunks would take about **5.4 minutes per question** and 1,000,000 chunks about **54 minutes**, far too slow for real use.

That is why the two stages are combined: the fast bi-encoder narrows the whole collection down to k candidates, and the accurate cross-encoder only re-orders those. A larger k finds more answers but costs more. The weakness is that the cross-encoder can only re-order what the bi-encoder found: in Query 6 the gold chunk was 16th and was missed, although scoring all 266 chunks would have ranked it 2nd, at about 35 times the cost. So k must be large enough to keep the right chunk in the candidate set.

## Limitations

- Only 6 failure questions, written by me to show each failure type: clear examples, not a full evaluation.
- In my 8 notebook questions, re-ranking never pushed a correct chunk out of the top 3, but a cross-encoder can also be misled, for example by a word used with a different meaning. I tried only one cross-encoder and one LLM.
- Latency was measured on one GPU (Tesla T4); the times will differ on other hardware.

## Conclusion

Cross-encoder re-ranking clearly improved the answers to hard questions (keyword traps, exceptions, negation and buried numbers) for only about 26 ms extra per question. It cannot recover chunks that the first stage misses, and better retrieval does not always mean the LLM gives a better answer.
