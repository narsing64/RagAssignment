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

**Bi-encoder.** The bi-encoder reads the question and each chunk *separately*. It turns each one into a single vector of 384 numbers, and relevance is the cosine similarity between the question vector **q** and the chunk vector **d**:

$$\cos(q, d) = \frac{q \cdot d}{\lVert q \rVert \, \lVert d \rVert}$$

Because all vectors are normalised to length 1, this is simply the dot product q · d, which is what FAISS `IndexFlatIP` computes. A chunk's vector does not depend on the question, so all chunk vectors can be computed once, before any question is asked, and stored in FAISS. When a question comes in, only the question is encoded and the stored vectors are searched, which is very fast. The weakness is that a whole chunk is squeezed into one vector, and the question and the chunk never "see" each other inside the model. Small but important details, such as "not", "except", or whether a rule is only proposed, can get lost.

**Cross-encoder.** The cross-encoder reads the question and the chunk *together*, as one input: `[CLS] question [SEP] chunk [SEP]`. Inside the transformer, self-attention connects every word of the question with every word of the chunk, in every layer. The output is one relevance score for the pair, not a vector. This lets the model check whether the chunk really answers the question. The cost is that nothing can be computed in advance: the model has to run again for every (question, chunk) pair.

**Example from my results.** For Query 1 ("maximum working days of leave per semester"), the bi-encoder ranked three chunks about a *proposed* new leave rule highest, because they use the same words ("leave", "working days", "limit"). The chunk with the rule that is actually in force (chunk 40: "up to 18 working days of leave in a semester") was only 7th. The cross-encoder moved it to 1st. The scores tell the same story. For an easy test question, the bi-encoder gave the right chunk 0.54 and the next chunk 0.38, a small gap. The cross-encoder gave the right chunk +4.46 and every other chunk between −9 and −11, a very clear gap.

Both models use the same small 6-layer MiniLM network, so the difference comes from *how* they read the text, not from model size.

## 2. Qualitative & Rank Shift Analysis

I wrote six test questions that cover the failure types listed in the assignment. For each question I found the **gold chunk**, the chunk that contains the true answer, by searching the chunks for the exact sentence from the PDF. Then I checked where each pipeline ranked it and what the LLM answered.

Two terms used below:

- **"Not in context"**: the LLM read the 3 chunks it was given and replied "The provided context does not contain this information." Our prompt tells it to give this reply, instead of guessing, when the answer is not in those 3 chunks.
- **Omission**: the answer leaves out the information that was asked for. A "not in context" reply is an omission.

### 2.1 Summary of rank shifts

| # | Failure type | Gold chunk (PDF page) | Rank: bi-encoder → cross-encoder | Answer 1 (baseline) | Answer 2 (re-ranked) |
|---|---|---|---|---|---|
| 1 | Keyword trap | 40 (p. 18–19) | #7 → **#1** | Omission | Correct |
| 2 | Exception (penalty) | 22 (p. 13) | #8 → **#1** | Omission | Correct |
| 3 | Exception ("except" clause) | 21 (p. 12–13) | #5 → **#1** | Omission | Correct (does not say "draft") |
| 4 | Negation | 118 (p. 44–45) | #8 → **#2** | Omission | Correct |
| 5 | Buried chunk (numbers) | 47 (p. 21) | #7 → **#2** | Omission | Omission |
| 6 | Limit of re-ranking | 109 (p. 41–42) | #16 → not re-ranked\* | Incomplete | Incomplete |

\* Only the bi-encoder's top 10 are re-ranked. This chunk was 16th, so the cross-encoder never scored it. When the cross-encoder scored all 266 chunks (Section 4), it ranked this chunk 2nd.

In Pipeline 1 the gold chunk was never among the 3 chunks given to the LLM. After re-ranking, it was among them for Queries 1–5.

### 2.2 Query-by-query comparison

#### Query 1: Keyword trap

- **Question:** "What is the maximum number of working days of leave a student can take in a semester?"
- **True answer** (Section 3.6, PDF p. 19): up to 18 working days per semester.
- **Chunks given to the LLM:** Pipeline 1: 60, 62, 63 · Pipeline 2: **40**, 21, 61
- **Why the baseline missed it:** chunks 60, 62 and 63 describe a *proposed* new leave rule (a combined 6-day limit, Section 3.18) that is on hold. They use the same words as the question ("leave", "working days", "limit"), so the bi-encoder ranked them first, even though they describe a proposal and not the rule in force.
- **Answer 1:** "The provided context does not contain this information." → **Omission**
- **Answer 2:** "A student can take up to 18 working days of leave in a semester. [1]" → **Correct**

#### Query 2: Exception (penalty)

- **Question:** "What happens to a student who backs out of an approved event without a valid reason?"
- **True answer** (Section 2.8, PDF p. 13): the student receives reduced support for the next event they enter.
- **Chunks given to the LLM:** Pipeline 1: 25, 41, 32 · Pipeline 2: **22**, 41, 33
- **Why the baseline missed it:** its chunks are about malpractice penalties (Section 2.10), delays in processing leave (Section 3.8) and Dean's List rules (Section 3.1). None of them is about events. The bi-encoder's scores for all ten candidates were close together (0.39 to 0.46), so the penalty sentence (chunk 22) ended up 8th. The cross-encoder gave chunk 22 a score of +7.6 and every other candidate −1.9 or less.
- **Answer 1:** "The provided context does not contain this information." → **Omission**
- **Answer 2:** "A student who backs out of an approved event without a valid reason receives reduced support for the next event they enter. [1]" → **Correct**

#### Query 3: Exception ("except" clause)

- **Question:** "How many days of leave can a student get for representing the Institute at external events?"
- **True answer** (Section 2.8, PDF p. 12–13): up to 7 working days per semester, with exceptions for events of high importance. This is a draft policy.
- **Chunks given to the LLM:** Pipeline 1: 60, 62, 20 · Pipeline 2: **21**, 20, 62
- **Why the baseline missed it:** two of its chunks are about the proposed leave limit again. Chunk 20 is the start of the correct policy, but the chunk boundary cuts it off at "A student can receive up to 7", so the number and the exception clause are in the next chunk (21), which was 5th.
- **Answer 1:** "The provided context does not contain this information." → **Omission**
- **Answer 2:** "A student can receive up to 7 working days of leave per semester for representing the Institute at national events, with exceptions for events of high importance. The leave provisions also apply to international events [1, 2]. However, the Parliament rejected a proposed 6-day limit for leave, stating that it should be raised to at least 12 working days, as a single event can take a week [3]." → **Correct**, including the exception clause. It does not say that the policy is still a draft, because that fact is in chunk 22, which was not given to the LLM.

#### Query 4: Negation

- **Question:** "Has any water softener been installed in the hostels?"
- **True answer** (Section 6.4, PDF p. 45): no, no softener has been installed.
- **Chunks given to the LLM:** Pipeline 1: 116, 117, 115 · Pipeline 2: 116, **118**, 117
- **Why the baseline missed it:** its chunks describe the softener *plans*: a softener "would be installed … for a 15-day trial", and later the committee "put the plans on hold". The bi-encoder could not tell "planned" apart from "not installed", so the chunk with the actual outcome (118) was 8th.
- **Answer 1:** "The provided context does not contain this information." → **Omission**
- **Answer 2:** "No softener has been installed. [2]" → **Correct**

#### Query 5: Buried chunk (numbers)

- **Question:** "What is the current monthly stipend for a Quarter Teaching Assistant?"
- **True answer** (Section 3.13, PDF p. 21): Rs. 7,500 per month, from Monsoon 2026.
- **Chunks given to the LLM:** Pipeline 1: 49, 51, 44 · Pipeline 2: 49, **47**, 48
- **Why the baseline missed it:** its chunks contain the old stipend (Rs. 6,000) and the Parliament's demands ("Rs. 8,000 was a reasonable figure"). The chunk with the announced amount (47) was 7th.
- **Answer 1:** "The provided context does not contain this information." → **Omission**
- **Answer 2:** "The provided context does not contain this information." → **Omission**, even though chunk 47 was given to the LLM. The same 3 chunks also contained the Parliament's demand of Rs. 8,000 "from Monsoon 2026", and the word "current" does not appear in the text. With two amounts for the same period, the LLM most likely could not tell which one is current, so it gave the "not in context" reply. Re-ranking fixed the retrieval here, but not the answer.

#### Query 6: Limit of re-ranking

- **Question:** "Was weekly room cleaning introduced in the hostels?"
- **True answer** (Section 6.1, PDF p. 41): no, weekly room cleaning was not introduced during the term.
- **Chunks given to the LLM:** Pipeline 1: 107, 108, 172 · Pipeline 2: 107, 79, 108
- **Why both pipelines missed it:** the gold chunk (109) was 16th, so it was not in the top 10 and the cross-encoder never saw it.
- **Answer 1:** "The Head, CFS stated that weekly room cleaning would be possible with more staff, but did not give a timeline. [2]" → **Incomplete**: it never says "no".
- **Answer 2:** the same answer, citing [3] → **Incomplete**

### 2.3 What the results show

- **The baseline never hallucinated and never stated a wrong rule.** All of its failures were omissions: five "not in context" replies and one incomplete answer. This is most likely because of the strict prompt, which tells the LLM not to guess.
- **Re-ranking fixed most failures.** It brought the gold chunk into the top 3 for 5 of the 6 questions and turned 4 omissions into correct answers (Queries 1–4).
- **Re-ranking has two limits.** Good retrieval does not always give a good answer (Query 5), and the cross-encoder cannot fix a chunk that the bi-encoder did not retrieve (Query 6).

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

## Limitations

- I tested only 6 failure questions, and I wrote them myself after reading the document, to show each failure type. They give clear examples, not a full evaluation.
- In the 8 questions in my notebook (6 failure questions and 2 sanity checks), re-ranking never pushed a correct chunk out of the top 3. But a cross-encoder can also make mistakes, for example when a word in the question matches a passage that uses the same word with a different meaning. I tried only one cross-encoder model.
- All answers come from one LLM (Gemini 2.5 Flash) with a strict prompt. A different LLM or a less strict prompt might answer differently, for example in Query 5.
- Latency was measured on one GPU (Tesla T4), so the exact times will be different on other hardware.

## Conclusion

Cross-encoder re-ranking clearly improved the answers to hard questions (keyword traps, exceptions, negation and buried numbers) for only about 26 ms extra per question. It cannot recover chunks that the first stage misses, and better retrieval does not always mean the LLM gives a better answer.
