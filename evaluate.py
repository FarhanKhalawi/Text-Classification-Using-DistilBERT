import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.amp import autocast

# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------
# DBpedia-14 class names (fallback)
# ----------------------
DBPEDIA14 = [
    "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
    "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
    "Plant","Album","Film","WrittenWork"
]


# ----------------------
# Infer class names from test parquet (prefer label_name if available)
# ----------------------
def infer_class_names_from_parquet(pq_path):
    try:
        df = pd.read_parquet(pq_path, columns=["label","label_name"])
        if "label" in df.columns and "label_name" in df.columns:
            m = df[["label","label_name"]].drop_duplicates().sort_values("label")
            names = m["label_name"].tolist()
            if len(names) == m["label"].nunique():
                return names
    except Exception:
        pass
    return DBPEDIA14


# ----------------------
# Build DataLoader for parquet file
# ----------------------
def build_test_loader(parquet_path, tokenizer, max_len=512, batch_size=16):
    df = pd.read_parquet(parquet_path)

    df["title"] = df["title"].fillna("") if "title" in df.columns else ""
    df["content"] = df["content"].fillna("") if "content" in df.columns else ""
    df["text"] = df["title"] + " [SEP] " + df["content"]

    def tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )

    cols = ["text", "label"] if "label" in df.columns else ["text"]
    ds = Dataset.from_pandas(df[cols], preserve_index=False).map(tok, batched=True)

    keep_cols = ["input_ids", "attention_mask"] + (["label"] if "label" in df.columns else [])
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    ds.set_format(type="torch", columns=keep_cols)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, df


# ----------------------
# Infer num_labels from checkpoint
# ----------------------
def infer_num_labels_from_state(state_dict, fallback=14):
    if "classifier.weight" in state_dict:
        return state_dict["classifier.weight"].shape[0]
    if "classifier.bias" in state_dict:
        return state_dict["classifier.bias"].shape[0]
    return fallback


# ----------------------
# Evaluation function
# ----------------------
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            if "label" in batch:
                golds.extend(batch["label"].cpu().tolist())

    return preds, golds


# ----------------------
# Heatmap 
# ----------------------
def plot_heatmap(mat, title, out_file):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# ----------------------
# Main
# ----------------------
def main():

    parser = argparse.ArgumentParser(description="Evaluate DistilBERT test set + attention visualizations")

    # DEFAULTS so user runs without arguments
    parser.add_argument("--test_parquet", default="Data/test_set.parquet")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--layer_id", type=int, default=3)
    parser.add_argument("--head_id", type=int, default=2)

    parser.add_argument("--sentence", type=str,
        default="A robot may not injure a human being or, through inaction, allow a human being to come to harm.")

    parser.add_argument("--cm_png", default="confusion_matrix.png")
    parser.add_argument("--attn_head_png", default="attn_post_head.png")
    parser.add_argument("--attn_avg_png", default="attn_post_avg.png")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load class names (confusion matrix)
    class_names = infer_class_names_from_parquet(args.test_parquet)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    test_loader, test_df = build_test_loader(
        args.test_parquet, tokenizer, args.max_len, args.batch_size
    )

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint missing: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device)

    if "label" in test_df.columns:
        num_labels = test_df["label"].nunique()
        print(f"Detected labels from test set: {num_labels}")
    else:
        num_labels = infer_num_labels_from_state(state)
        print(f"Inferred labels from checkpoint: {num_labels}")

    id2label = {i: class_names[i] for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager"
    ).to(device)

    model.load_state_dict(state, strict=True)
    print(f"✓ Loaded checkpoint: {args.checkpoint}")

    # ----------------------
    # Evaluation
    # ----------------------
    preds, golds = evaluate(model, test_loader, device)

    print(f"\nTest accuracy: {accuracy_score(golds, preds):.4f}\n")

    print(classification_report(
        golds, preds, target_names=[id2label[i] for i in range(num_labels)], digits=4
    ))

    # Pretty confusion matrix
    cm = confusion_matrix(golds, preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in range(num_labels)])
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=True)
    plt.title("DBpedia-14 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.cm_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {args.cm_png}")

    # ----------------------
    # Post-training attention
    # ----------------------
    toks = tokenizer(args.sentence, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}

    with torch.no_grad():
        out = model(**toks, output_attentions=True)

    attentions = [a.squeeze(0).cpu().numpy() for a in out.attentions]

    L, H = args.layer_id, args.head_id
    head_map = attentions[L][H]
    avg_map = attentions[L].mean(axis=0)

    plot_heatmap(head_map, f"Post-training L{L} H{H}", args.attn_head_png)
    plot_heatmap(avg_map, f"Post-training L{L} (avg heads)", args.attn_avg_png)

    print(f"Saved attention heatmaps: {args.attn_head_png}, {args.attn_avg_png}")


if __name__ == "__main__":
    main()
