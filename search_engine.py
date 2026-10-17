import os
import re
import html
import requests
import numpy as np
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

SERPER_API_KEY = "4b965251390f8ab75032eaf21304abaec9f66f2f"
SEARCH_URL = "https://google.serper.dev/search"

_TAG_RE = re.compile(r"<[^>]+>")


def _text(page, key):
    return (page.get(key) or "").strip()


def _clean(value):
    """Strip any HTML tags/entities and collapse whitespace."""
    value = html.unescape(value or "")
    value = _TAG_RE.sub("", value)
    return " ".join(value.split())


def search_web(query, count=20):
    """Retrieve real webpages using the Serper.dev (Google) API."""
    if not SERPER_API_KEY:
        raise RuntimeError(
            "Missing API key. Set SERPER_API_KEY in your terminal "
            "before running the program."
        )

    try:
        count = max(1, min(int(count), 20))
    except (TypeError, ValueError):
        count = 20

    response = requests.post(
        SEARCH_URL,
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "q": query,
            "num": count,
            "gl": "gb",   # country
            "hl": "en",   # language
        },
        timeout=20,
    )

    if not response.ok:
        raise RuntimeError(
            f"Serper API error {response.status_code}: {response.text}"
        )

    data = response.json()

    pages = []
    for item in data.get("organic") or []:
        url = item.get("link") or ""
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        pages.append({
            "title": _clean(item.get("title")),
            "snippet": _clean(item.get("snippet")),
            "url": url,
            "domain": domain,
            # Serper does not provide publisher/author metadata.
            "publisher": "",
            "published": item.get("date") or "",
            # Placeholder until you implement real exposure data.
            "underexposure": 0.0,
        })

    return pages


def calculate_quality(page):
    """
    Metadata-completeness heuristic (0-1), not a credibility check.
    Uses integer points to avoid floating-point threshold issues.
    """
    points = 0

    if len(_text(page, "title")) >= 10:
        points += 1
    if len(_text(page, "snippet")) >= 80:
        points += 2
    if _text(page, "domain"):
        points += 1
    if page.get("author") or page.get("publisher") or page.get("published"):
        points += 1

    return points / 5  # max 5 points -> score in [0, 1]


def rank_results(
    query,
    pages,
    limit=10,
    relevance_weight=0.60,
    diversity_weight=0.25,
    exposure_weight=0.15,
    min_relevance=0.30,
    min_quality=0.60,
):
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10

    if not 0 <= min_relevance <= 1:
        raise ValueError("min_relevance must be between 0 and 1")
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between 0 and 1")
    if not pages:
        return []

    texts = [
        f"{_text(p, 'title')}. {_text(p, 'snippet')}"
        for p in pages
    ]

    embeddings = model.encode(
        [query] + texts,
        normalize_embeddings=True,
    )
    query_emb, page_embs = embeddings[0], embeddings[1:]

    # Semantic relevance using raw cosine similarity
    relevance = np.clip(page_embs @ query_emb, 0.0, 1.0)

    qualities = [calculate_quality(p) for p in pages]

    keep = [
        i for i in range(len(pages))
        if relevance[i] >= min_relevance and qualities[i] >= min_quality
    ]

    if not keep:
        return []

    E = page_embs[keep]
    rel = relevance[keep]

    under = np.array([
        min(max(float(pages[i].get("underexposure") or 0.0), 0.0), 1.0)
        for i in keep
    ])

    # Pairwise similarity for diversity reranking
    sim = E @ E.T
    max_sim = np.zeros(len(keep))
    chosen = []
    taken = np.zeros(len(keep), dtype=bool)

    while len(chosen) < min(limit, len(keep)):
        scores = (
            relevance_weight * rel
            - diversity_weight * max_sim
            + exposure_weight * under
        )

        scores[taken] = -np.inf
        best = int(np.argmax(scores))

        chosen.append((best, float(scores[best])))
        taken[best] = True
        max_sim = np.maximum(
            max_sim,
            np.clip(sim[best], 0.0, None),
        )

    results = []

    for local_idx, score in chosen:
        orig_idx = keep[local_idx]
        page = pages[orig_idx].copy()
        page["relevance"] = float(rel[local_idx])
        page["quality"] = qualities[orig_idx]
        page["underexposure"] = float(under[local_idx])
        page["ranking_score"] = score
        results.append(page)

    return results


def main():
    query = input("What would you like to search for? ").strip()

    if not query:
        print("Please enter a search query.")
        return

    try:
        pages = search_web(query, count=20)

        print(f"\nRetrieved {len(pages)} webpages.")
        results = rank_results(query, pages, limit=10)

        if not results:
            print(
                "No pages passed the filters. Try lowering "
                "min_relevance or min_quality."
            )
            return

        print("\n=== Recommended results ===")

        for i, page in enumerate(results, start=1):
            print(f"\n{i}. {page['title']}")
            print(f"   URL: {page['url']}")
            print(f"   Relevance: {page['relevance']:.3f}")
            print(f"   Quality: {page['quality']:.2f}")
            print(f"   Ranking score: {page['ranking_score']:.3f}")

    except requests.RequestException as error:
        print(f"Search API request failed: {error}")
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()