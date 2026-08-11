"""
Vectorization 101: turn sentences into embedding vectors, look at the raw
numbers, measure how "close" two vectors are, and plot them in 2D so
semantically similar sentences visibly cluster together.

Embeddings come from Voyage AI (Anthropic's recommended embedding provider —
the Claude API itself does not produce embeddings). Claude Haiku is used at
the end as a bonus step to narrate why the clustering happened.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your keys

Run:
    python embed_and_visualize.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import voyageai
from dotenv import load_dotenv
from sklearn.decomposition import PCA

load_dotenv()

VOYAGE_MODEL = "voyage-3.5"
HAIKU_MODEL = "claude-haiku-4-5"

# Three topics, four sentences each. Embeddings should place same-topic
# sentences closer together than different-topic ones.
SENTENCES_BY_CATEGORY = {
    "animals": [
        "The dog chased the ball across the yard.",
        "A cat curled up and fell asleep on the sofa.",
        "Wolves hunt in coordinated packs at night.",
        "The parrot mimicked its owner's voice perfectly.",
    ],
    "technology": [
        "The new smartphone has a faster processor.",
        "She debugged the code late into the night.",
        "Cloud servers handle millions of requests per second.",
        "The startup shipped its first AI feature this week.",
    ],
    "food": [
        "The chef simmered the tomato sauce for hours.",
        "Fresh basil makes the pasta taste incredible.",
        "They grilled vegetables over an open flame.",
        "The bakery sells warm croissants every morning.",
    ],
}

CATEGORY_COLORS = {"animals": "tab:orange", "technology": "tab:blue", "food": "tab:green"}


def get_embeddings(texts):
    client = voyageai.Client()  # reads VOYAGE_API_KEY from env
    result = client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def cosine_similarity_matrix(vectors):
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return normed @ normed.T


def explain_with_haiku(sentences, categories):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    import anthropic

    client = anthropic.Anthropic()
    listing = "\n".join(f"- [{c}] {s}" for s, c in zip(sentences, categories))
    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "These sentences were converted into embedding vectors and "
                        "plotted in 2D. In 2-3 sentences, explain in plain terms why "
                        "sentences about the same topic end up near each other in "
                        "vector space.\n\n" + listing
                    ),
                }
            ],
        )
    except anthropic.APIError as e:
        print(f"(Claude Haiku call failed, skipping commentary: {e})")
        return None
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    sentences, categories = [], []
    for category, texts in SENTENCES_BY_CATEGORY.items():
        sentences.extend(texts)
        categories.extend([category] * len(texts))

    print(f"Embedding {len(sentences)} sentences with Voyage AI ({VOYAGE_MODEL})...\n")
    vectors = get_embeddings(sentences)

    print(f"Vector shape: {vectors.shape}  -> each sentence became a {vectors.shape[1]}-dimensional vector")
    print(f"First 8 numbers of sentence 0's vector:\n  {np.round(vectors[0][:8], 4)}\n")

    sims = cosine_similarity_matrix(vectors)
    same_topic_idx = [i for i, c in enumerate(categories) if c == categories[0]][1]
    other_topic_idx = next(i for i, c in enumerate(categories) if c != categories[0])
    print("Cosine similarity from sentence 0:")
    print(f"  to another '{categories[0]}' sentence -> {sims[0, same_topic_idx]:.4f}")
    print(f"  to a '{categories[other_topic_idx]}' sentence -> {sims[0, other_topic_idx]:.4f}\n")

    print("Reducing 1024-ish dimensions down to 2D with PCA so we can actually plot it...")
    coords = PCA(n_components=2).fit_transform(vectors)

    plt.figure(figsize=(9, 7))
    for category in SENTENCES_BY_CATEGORY:
        idx = [i for i, c in enumerate(categories) if c == category]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=category, color=CATEGORY_COLORS[category], s=80)
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence[:24] + "...", (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.7)
    plt.title(f"Sentence embeddings ({VOYAGE_MODEL}) reduced to 2D with PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("embeddings_plot.png", dpi=150)
    print("Saved plot to embeddings_plot.png")

    commentary = explain_with_haiku(sentences, categories)
    if commentary:
        print(f"\nClaude Haiku ({HAIKU_MODEL}) explains the clustering:\n{commentary}")
    else:
        print("\n(Set ANTHROPIC_API_KEY to also get a short Claude Haiku explanation of the clustering.)")


if __name__ == "__main__":
    main()
