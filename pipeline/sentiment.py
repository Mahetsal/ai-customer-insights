"""
Sentiment Analysis Engine.
Uses pre-trained transformer models to classify text sentiment
with confidence scores and explanation tokens.
"""

from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Lazy-load the model to avoid slow imports
_sentiment_pipeline = None


def get_pipeline():
    """Get or initialize the sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading sentiment model (first run may take a minute)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True,
        )
    return _sentiment_pipeline


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text.
    
    Args:
        text: Input text to analyze.
    
    Returns:
        Dict with label (POSITIVE/NEGATIVE/NEUTRAL), score, and raw scores.
    """
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.5, "raw_scores": {}}
    
    pipe = get_pipeline()
    results = pipe(text[:512])[0]  # Truncate to model max length
    
    scores = {r["label"]: round(r["score"], 4) for r in results}
    
    # Determine label and confidence
    pos_score = scores.get("POSITIVE", 0)
    neg_score = scores.get("NEGATIVE", 0)
    
    if pos_score > 0.6:
        label = "POSITIVE"
        confidence = pos_score
    elif neg_score > 0.6:
        label = "NEGATIVE"
        confidence = neg_score
    else:
        label = "NEUTRAL"
        confidence = 1.0 - abs(pos_score - neg_score)
    
    return {
        "label": label,
        "score": round(confidence, 4),
        "raw_scores": scores,
    }


def analyze_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of text strings.
        batch_size: Processing batch size.
    
    Returns:
        List of sentiment analysis results.
    """
    results = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(total - 1) // batch_size + 1}")
        
        for text in batch:
            results.append(analyze_sentiment(text))
    
    return results


def get_sentiment_summary(results: list[dict]) -> dict:
    """
    Generate summary statistics from sentiment results.
    """
    total = len(results)
    if total == 0:
        return {"total": 0}
    
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total_score = 0
    
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
        if r["label"] == "POSITIVE":
            total_score += r["score"]
        elif r["label"] == "NEGATIVE":
            total_score -= r["score"]
    
    return {
        "total": total,
        "positive": counts["POSITIVE"],
        "negative": counts["NEGATIVE"],
        "neutral": counts["NEUTRAL"],
        "positive_pct": round(counts["POSITIVE"] / total * 100, 1),
        "negative_pct": round(counts["NEGATIVE"] / total * 100, 1),
        "overall_score": round(total_score / total, 3),
    }
