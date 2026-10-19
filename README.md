# RAG: Baseline Bi-Encoder Retrieval vs Cross-Encoder Re-Ranking

LMA assignment: two retrieval pipelines over the **same chunked corpus** of one PDF, compared on stress-test queries, with latency profiling.

Everything (code, outputs, comments) is in one notebook: **[`RAG_Baseline_vs_Reranking.ipynb`](RAG_Baseline_vs_Reranking.ipynb)**.

## Repository contents

| File | What it is |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | The full notebook, executed, with all outputs |
| `requirements.txt` | Library versions used in the run |
| `Report.md` | The report |
| `parliament-activities-report-annual-2025-2026.pdf` | The input document (95 pages) |
| `README.md` | This file |

## Document

*IIIT Hyderabad Students' Parliament — Parliament Activities Report, Annual Report 2025–2026* (95-page PDF, text-based).
PDF pages 1–5 (cover and contents) are skipped; the running header, page numbers and soft hyphens are removed. The remaining 90 pages (31,868 words) are split into **266 chunks** of 150 words with a 30-word overlap.

## Pipelines

**Pipeline 1 — Baseline**
1. Bi-encoder embeds every chunk once; vectors stored in a FAISS `IndexFlatIP` (normalised vectors, so inner product = cosine similarity).
2. For a query, retrieve the **top 10** chunks.
3. Send the **top 3** to the LLM → **Answer 1**.

**Pipeline 2 — Re-ranking**
1. Take the **same top 10** candidates.
2. Score all 10 (query, chunk) pairs with a cross-encoder and re-sort.
3. Send the new **top 3** to the LLM with the **identical prompt** → **Answer 2**.

## Models and settings

| Component | Choice |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256-token limit) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (512-token limit for query + chunk) |
| Vector store | FAISS `IndexFlatIP` (exact search) |
| LLM | Gemini `gemini-2.5-flash`, temperature 0, thinking disabled |
| Prompt | One fixed template for both pipelines: answer only from the given passages, otherwise reply that the context does not contain the information; cite passage numbers |
| Hardware | Kaggle notebook, NVIDIA Tesla T4 GPU |

## Results (from the submitted run)

### Failure-mode queries

The **gold chunk** is the chunk containing the true answer, located by its exact wording in the PDF.

| # | Category | Query | Gold chunk (PDF page) | Bi-encoder rank | Cross-encoder rank | Answer 1 (baseline) | Answer 2 (re-ranked) |
|---|---|---|---|---|---|---|---|
| 1 | Keyword trap | Maximum working days of leave per semester | 40 (p. 18–19) | #7 | **#1** | No answer | Correct: 18 working days |
| 2 | Exception (penalty) | Backing out of an approved event without a valid reason | 22 (p. 13) | #8 | **#1** | No answer | Correct: reduced support for the next event |
| 3 | Exception (except-clause) | Leave for representing the Institute at external events | 21 (p. 12–13) | #5 | **#1** | No answer | Correct: up to 7 working days, with exceptions for events of high importance |
| 4 | Negation | Has any water softener been installed? | 118 (p. 44–45) | #8 | **#2** | No answer | Correct: no softener has been installed |
| 5 | Buried chunk (numbers) | Current monthly stipend for a Quarter TA | 47 (p. 21) | #7 | **#2** | No answer | No answer |
| 6 | Limit: gold outside top 10 | Was weekly room cleaning introduced? | 109 (p. 41–42) | #16 | not re-ranked | Did not state the outcome | Did not state the outcome |

"No answer" means the LLM replied *"The provided context does not contain this information."*

### Latency (T4 GPU, mean of 20 timed runs over 8 queries, LLM excluded)

| Stage | Time |
|---|---|
| Pipeline 1: query encoding + FAISS search | 6.4 ms |
| Cross-encoder re-ranking of 10 pairs | 26.1 ms |
| Pipeline 2: retrieval + re-ranking | 32.5 ms (5.1× Pipeline 1) |
| LLM call (same for both pipelines; Step 6 runs) | 0.61 s mean (0.43–0.89 s) |
| One-time indexing of all 266 chunks (bi-encoder) | 0.89 s |

### Scaling: cross-encoder on more candidates

| Pairs scored (k) | Cross-encoder time |
|---|---|
| 10 | 24.5 ms |
| 50 | 122.4 ms |
| 100 | 306.7 ms |
| 266 (all chunks) | 857.8 ms |

Scoring all 266 chunks ranks the gold chunk of query 6 at #2.

## How to run (Kaggle)

1. Create a Kaggle notebook and import `RAG_Baseline_vs_Reranking.ipynb` (**File → Import Notebook**).
2. Upload the PDF (`parliament-activities-report-annual-2025-2026.pdf`, included in this repository) as a Kaggle dataset (**Add Input → Upload**) and set `PDF_PATH` in Step 1 to its path (Cell 2 prints the path).
3. Session options: **Internet On**, **Accelerator: GPU T4**.
4. Add a Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)) under **Add-ons → Secrets** with the name `GEMINI_API_KEY`, and attach it to the notebook.
5. **Run All** (about 8–10 minutes).

### Notes

- A full run makes **16 Gemini calls**. The free tier of `gemini-2.5-flash` allows 20 requests per day and only a few per minute, so the notebook waits `PAUSE = 15` seconds between calls (outside every timed section).
- Each Step 6 result is also saved to `/kaggle/working/failure_log.json`, so Step 7 can rebuild its summary after a session restart without calling Gemini again.
- With temperature 0, LLM wording can still vary slightly between runs; the retrieval and re-ranking results are deterministic.
