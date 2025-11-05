# Text Classification using DistilBERT (DBpedia-14)

## Overview
This project fine-tunes **DistilBERT** on the **DBpedia-14** dataset for text classification into 14 categories, such as *Company*, *Artist*, and *Film*.  
The model achieves high accuracy and outputs detailed evaluation metrics, including per-class F1-scores and a confusion matrix.

---

## Model and Dataset
- **Model**: `distilbert-base-uncased` (Hugging Face Transformers)
- **Dataset**: DBpedia-14 (title + content fields)
- **Text Input Format**: `title [SEP] content`
- **Sequence Length**: 512 tokens
- **Labels**: 14 classes (Company, Artist, Film, etc.)

---

## Project Structure

Text-Classification-Using-DistilBERT/
│
├── Data/
│   └── dataset.parquet          # Dataset used for training and evaluation
│
├── train.ipynb                  # Jupyter Notebook for model training
├── evaluate.py                  # Script for evaluating and generating metrics
├── best_model.pt                # Saved fine-tuned model weights
├── requirements.txt             # Required dependencies
└── README.md                    # Project documentation
