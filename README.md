# FAQ Retrieval Assistant (Embeddings + Cosine Similarity)

I built this project as a simple AI agent for customer support that retrieves the most relevant answers from a small knowledge base using sentence embeddings and cosine similarity.

## Approach
- Loading the FAQ dataset i manually created containing question–answer pairs.
- Creating embeddings for the questions using a multilingual SentenceTransformer model.
- Normalizing all embeddings, so cosine similarity can be computed efficiently using a dot product.
- For a user query:
  - Embed the query,
  - Compute cosine similarity against all FAQ embeddings,
  - Retrieve the top-3 most similar questions,
  - Return the best-matching answer.
- Compute a simple confidence level (HIGH / MEDIUM / LOW) based on the top similarity score and the score margin between the best matches.

## Tools Used
- Python
- Sentence-transformers
- NumPy

## How to Run (Local)
1) (Optional) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Run the CLI demo
```bash
python -m src.app
```

Example:
Ask a question, or press ENTER to exit: How to reset my password?
Confidence: HIGH (top1=0.996, margin=0.682)
Login -> Forgot your password? -> Here you enter your email address, then follow the link we send.

Ask a question, or press ENTER to exit: Каков тип на плаќање поддржува вашата компанија?
Confidence: HIGH (top1=0.678, margin=0.153)
We accept major credit cards, debit cards, and PayPal as payment methods for online orders.

How can this project scale?

- Using an ANN index for faster top-k retrieval on large knowledge bases
- Adding a reranking stage for better precision
- Monitoring retrieval quality and adding evaluation sets
