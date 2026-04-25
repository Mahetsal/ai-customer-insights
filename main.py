"""
Customer Insights CLI — Main entry point.
Processes customer feedback CSV files through sentiment analysis
and topic categorization pipelines.
"""

import argparse
import csv
import json
import os
import logging
from datetime import datetime
from pipeline.sentiment import analyze_sentiment, analyze_batch, get_sentiment_summary
from pipeline.categorizer import categorize, categorize_batch, get_category_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_csv(input_path: str, output_dir: str):
    """Process a CSV file of customer reviews."""
    logger.info(f"Reading {input_path}...")
    
    reviews = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("review") or row.get("text") or row.get("feedback", "")
            if text.strip():
                reviews.append({"text": text, "date": row.get("date", ""), "id": row.get("id", "")})
    
    logger.info(f"Loaded {len(reviews)} reviews")
    
    # Run sentiment analysis
    texts = [r["text"] for r in reviews]
    sentiments = analyze_batch(texts)
    categories = categorize_batch(texts)
    
    # Combine results
    results = []
    for review, sentiment, category in zip(reviews, sentiments, categories):
        results.append({
            **review,
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "category": category["primary"],
            "all_categories": category["categories"],
        })
    
    # Generate summary
    summary = {
        "processed_at": datetime.utcnow().isoformat(),
        "total_reviews": len(results),
        "sentiment": get_sentiment_summary(sentiments),
        "categories": get_category_distribution(categories),
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "analyzed_reviews.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Reviews: {summary['total_reviews']}")
    print(f"\nSentiment Breakdown:")
    s = summary["sentiment"]
    print(f"  ✅ Positive: {s['positive']} ({s['positive_pct']}%)")
    print(f"  ❌ Negative: {s['negative']} ({s['negative_pct']}%)")
    print(f"  ➖ Neutral:  {s.get('neutral', 0)}")
    print(f"\nTop Categories:")
    for cat, info in list(summary["categories"]["distribution"].items())[:5]:
        print(f"  📁 {cat}: {info['count']} ({info['percentage']}%)")
    print("=" * 50)


def analyze_single(text: str):
    """Analyze a single text input."""
    sentiment = analyze_sentiment(text)
    category = categorize(text)
    
    print(f"\n📝 Input: {text}")
    print(f"😊 Sentiment: {sentiment['label']} (confidence: {sentiment['score']})")
    print(f"📁 Category: {category['primary']}")
    print(f"🏷️  All topics: {', '.join(category['categories'])}")


def main():
    parser = argparse.ArgumentParser(description="AI Customer Insights Analyzer")
    parser.add_argument("--input", "-i", help="Path to CSV file with reviews")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--text", "-t", help="Analyze a single text")
    
    args = parser.parse_args()
    
    if args.text:
        analyze_single(args.text)
    elif args.input:
        process_csv(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
