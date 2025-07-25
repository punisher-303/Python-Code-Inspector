# 🐞 Python Code Inspector - AI Bug Detection Tool

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/punisher-303/Python-Code-Inspector)
[![Hugging Face](https://img.shields.io/badge/🤗%20Demo-Spaces-yellow)](https://huggingface.co/spaces/punisher-303/python-code-inspector)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

An intelligent AI tool that detects common Python bugs and suggests fixes using machine learning and pattern matching.

![Python Code Inspector Demo](demo_screenshot.png) <!-- Replace with your actual screenshot -->

## 🏆 Hackathon Features

- **AI-Powered Analysis**: Combines ML with rule-based corrections
- **Real-Time Feedback**: Instant bug detection as you type
- **Educational**: Explains why changes are needed
- **Production-Ready**: Clean Gradio interface with system monitoring

## 🚀 Quick Start

1. Try the live demo:
   [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/punisher-303/python-code-inspector)

2. Run locally:
   ```bash
   git clone https://github.com/punisher-303/Python-Code-Inspector.git
   cd Python-Code-Inspector
   pip install -r requirements.txt
   python app.py


   🛠️ How It Works
https://architecture.png <!-- Add your diagram here -->

Code Tokenization: Custom parser breaks down Python syntax

Feature Extraction: TF-IDF with n-grams (1-4) captures patterns

ML Classification: Random Forest predicts bug probability

Smart Corrections: Rule-based fixes with context


# Example Bug Detection
def add(a, b): return a - b  # Detected (98% confidence)
# Suggested Fix:
def add(a, b): return a + b


💡 Key Features
Feature	Description
20+ Bug Patterns	Wrong operators, missing colons, unsafe code
Confidence Scoring	Visual 0-100% confidence indicator
Detailed Analysis	JSON output with technical details
System Monitoring	CPU/RAM usage tracking
📊 Model Performance
Accuracy: 92.5% (cross-validated)

Inference Speed: <100ms

Training Data: 200 curated examples

Vocabulary Size: 500 tokens

🛠️ Development

# Install dependencies
pip install -r requirements.txt

# Train new model
python train.py

# Run tests
python -m pytest tests/


🤝 Contributing
Found a bug we missed?

Fork the repository

Add your test case to train.py

Submit a pull request

📜 License
MIT License - See LICENSE for details.

👨‍💻 Team
Punisher-303

Add Teammates
