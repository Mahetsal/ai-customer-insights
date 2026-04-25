"""
Topic Categorization Pipeline.
Classifies text into pre-defined business themes using
zero-shot classification models.
"""

from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Lazy-load the model
_classifier = None

CATEGORIES = [
    "Product Quality",
    "Shipping & Delivery",
    "Customer Support",
    "Pricing & Value",
    "User Interface",
    "Features & Functionality",
]


def get_classifier():
    """Get or initialize the zero-shot classifier."""
    global _classifier
    if _classifier is None:
        logger.info("Loading categorization model...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    return _classifier


def categorize(text: str) -> dict:
    """
    Categorize text into predefined topics.
    """
    if not text or not text.strip():
        return {"primary": "Uncategorized", "categories": []}
    
    classifier = get_classifier()
    result = classifier(text[:512], candidate_labels=CATEGORIES, multi_label=True)
    
    # Get categories with score > 0.3
    top_categories = [
        label for label, score in zip(result["labels"], result["scores"])
        if score > 0.3
    ]
    
    return {
        "primary": result["labels"][0],
        "categories": top_categories,
        "scores": {l: round(s, 4) for l, s in zip(result["labels"], result["scores"])},
    }


def categorize_batch(texts: list[str]) -> list[dict]:
    """Process a batch of texts for categorization."""
    return [categorize(t) for t in texts]


def get_category_distribution(results: list[dict]) -> dict:
    """Generate distribution stats for categories."""
    total = len(results)
    if total == 0:
        return {"total": 0}
    
    dist = {}
    for r in results:
        primary = r["primary"]
        if primary not in dist:
            dist[primary] = {"count": 0, "percentage": 0}
        dist[primary]["count"] += 1
    
    for cat in dist:
        dist[cat]["percentage"] = round(dist[cat]["count"] / total * 100, 1)
    
    # Sort by count
    sorted_dist = dict(sorted(dist.items(), key=lambda item: item[1]["count"], reverse=True))
    
    return {
        "total": total,
        "distribution": sorted_dist
    }
