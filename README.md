# 📊 AI Customer Insights

A Python NLP pipeline that analyzes customer feedback using sentiment analysis, topic categorization, and trend detection — turning raw reviews into actionable business intelligence.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Sentiment Analysis** — Classify feedback as positive, negative, or neutral using pre-trained transformer models
- **Topic Categorization** — Automatically categorize feedback into themes (shipping, quality, support, pricing)
- **Trend Detection** — Track sentiment changes over time to identify emerging issues
- **Dashboard** — Interactive HTML dashboard with charts and filters
- **CSV/JSON Export** — Export analyzed data for further processing
- **Batch Processing** — Handle thousands of reviews efficiently

## 📁 Project Structure

```
ai-customer-insights/
├── pipeline/
│   ├── __init__.py
│   ├── sentiment.py            # Sentiment analysis engine
│   ├── categorizer.py          # Topic categorization
│   ├── trends.py               # Trend detection
│   └── preprocessor.py         # Text cleaning & normalization
├── dashboard/
│   ├── index.html              # Interactive dashboard
│   ├── style.css
│   └── charts.js
├── data/
│   └── sample_reviews.csv      # Sample dataset
├── main.py                     # CLI entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/saleh-alkhrisat/ai-customer-insights.git
cd ai-customer-insights
pip install -r requirements.txt

# Analyze sample data
python main.py --input data/sample_reviews.csv --output results/

# Or analyze a single text
python main.py --text "The product quality is amazing but shipping was slow"
```

## 📝 License

MIT License
