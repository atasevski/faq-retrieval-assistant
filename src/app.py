import json
from pathlib import Path
from .embedder import Embedder
from .retriever import Retriever

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    faqs_path = base_dir / "data" / "faqs.json"

    with faqs_path.open("r", encoding="utf-8") as f:
        faqs = json.load(f)

    embedder = Embedder()
    faq_embs = embedder.encode([f["question"] for f in faqs])

    retriever = Retriever(faqs, faq_embs)

    while True:
        query = input("Ask a question, or press ENTER to exit: ")
        if not query.strip():
            break
        results = retriever.search(query, embedder, top_k=3)

        if not results:
            print("No results found!")
            continue
        best_answer = results[0]

        top1 = results[0].score
        top2 = results[1].score if len(results) > 1 else None
        margin = (top1 - top2) if top2 is not None else None

        if top1 >= 0.5 and (margin is None or margin >= 0.05):
            confidence = "HIGH"
        elif top1 >= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if margin is None:
            print(f"Confidence: {confidence} (top1={top1:.3f})")
        else:
            print(f"Confidence: {confidence} (top1={top1:.3f}, margin={margin:.3f})")
        if confidence == "LOW":
            print("Low confidence — try rephrasing your question.")
            continue

        print(best_answer.answer)
if __name__ == '__main__':
    main()
