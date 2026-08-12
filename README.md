# embed-ai

Two small, runnable exercises building up from embeddings to a working
RAG pipeline — no framework, no hidden magic, just the API calls and the
math.

- **Part 1 (this folder)** — turn sentences into embedding vectors, look
  at the raw numbers, measure how "close" two vectors are, and plot them
  in 2D so semantically similar sentences visibly cluster together.
- **Part 2 ([`rag-search-exercise/`](rag-search-exercise/))** — use those
  same vectors to retrieve relevant documents for a question, so Claude
  answers grounded in real text instead of guessing.

Do Part 1 first — Part 2 assumes you're already comfortable with
embeddings and cosine similarity.

---

## Part 1: Vectorization & Embeddings

Turn sentences into embedding vectors, look at the raw numbers, measure
how "close" two vectors are, and plot them in 2D so semantically similar
sentences visibly cluster together.

![Example output: three clusters of sentence embeddings](embeddings_plot.png)

*Real output from a run of this script — 12 sentences across 3 topics
(animals, technology, food), embedded with Voyage AI and reduced to 2D with
PCA. Same-topic sentences land near each other automatically.*

## Why Voyage AI, not Claude?

The Claude API (including Haiku) is a text-generation model — it does not
produce embeddings. **Voyage AI** is Anthropic's recommended embeddings
provider and is what actually turns text into vectors here. This project
uses:

- **Voyage AI** (`voyage-3.5`) — generates the embedding vectors
- **Claude Haiku** (`claude-haiku-4-5`) — optional bonus step that explains
  in plain English why the clustering happened

## Setup

```bash
git clone https://github.com/arjunuvlad2/embed-ai.git
cd embed-ai
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `VOYAGE_API_KEY` — **required**. Get one at https://dashboard.voyageai.com/
- `ANTHROPIC_API_KEY` — optional, only needed for the closing Claude Haiku
  explanation step. If it's missing (or invalid), the script still runs
  and produces the plot — it just skips that last step.

## Run

```bash
python embed_and_visualize.py
```

## What it does

1. Takes 12 short sentences across 3 topics (animals, technology, food)
2. Embeds all of them in one Voyage AI API call
3. Prints the raw vector shape and the first few numbers of one vector
4. Computes cosine similarity to show same-topic sentences score higher
   than different-topic sentences
5. Reduces the high-dimensional vectors to 2D with PCA and saves a scatter
   plot (`embeddings_plot.png`) — you should see three visible clusters,
   like the one above
6. If `ANTHROPIC_API_KEY` is set and valid, asks Claude Haiku to explain
   the result in plain English

## Example output

```
Embedding 12 sentences with Voyage AI (voyage-3.5)...

Vector shape: (12, 1024)  -> each sentence became a 1024-dimensional vector
First 8 numbers of sentence 0's vector:
  [-0.015   0.0451 -0.0128  0.0335  0.0279 -0.0134 -0.0222 -0.0541]

Cosine similarity from sentence 0:
  to another 'animals' sentence -> 0.7773
  to a 'technology' sentence -> 0.6314

Reducing 1024-ish dimensions down to 2D with PCA so we can actually plot it...
Saved plot to embeddings_plot.png
```

## Things to try next

- Swap in your own sentences/categories in `SENTENCES_BY_CATEGORY`
- Try a different Voyage model (`voyage-3-large` for higher quality,
  `voyage-3.5-lite` for cheaper/faster) and compare the plot
- Swap PCA for `sklearn.manifold.TSNE` and compare the layout
- Print the full similarity matrix instead of just one row

---

## Part 2: Semantic Search & RAG

Once Part 1 makes sense, head into
[`rag-search-exercise/`](rag-search-exercise/) — same embedding model,
same cosine similarity math, now used to retrieve relevant documents for
a question so Claude answers grounded in real text instead of guessing.
Full setup and run instructions are in that folder's own README.

## License

MIT — see [LICENSE](LICENSE).
