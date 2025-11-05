import os, sys, argparse, time
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, ConfusionMatrixDisplay)
from tqdm.auto import tqdm

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DBPEDIA14 = ["Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
             "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
             "Plant","Album","Film","WrittenWork"]

def infer_class_names(pq):
    try:
        df = pd.read_parquet(pq, columns=["label","label_name"])
        if "label_name" in df.columns:
            m = df[["label","label_name"]].drop_duplicates().sort_values("label")
            names = m["label_name"].tolist()
            if len(names) == m["label"].nunique(): return names
    except Exception: pass
    return DBPEDIA14

def load_model_and_tokenizer(test_file_hint):
    print("→ Loading tokenizer…", flush=True)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    class_names = infer_class_names(test_file_hint)
    print(f"→ Using {len(class_names)} classes.", flush=True)
    print("→ Building model skeleton…", flush=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(class_names),
        id2label={i:n for i,n in enumerate(class_names)},
        label2id={n:i for i,n in enumerate(class_names)},
    )
    if not os.path.exists("best_model.pt"):
        print("❌  best_model.pt not found.", flush=True); sys.exit(1)
    print("→ Loading fine-tuned weights from best_model.pt …", flush=True)
    state = torch.load("best_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"✅  Weights loaded ({len(state)} tensors)", flush=True)
    model.eval()
    return model, tok, class_names

class ParquetTextDataset(Dataset):
    def __init__(self, df):
        self.texts = (df["title"].fillna("") + " [SEP] " + df["content"].fillna("")).tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else None
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        d = {"text": self.texts[i]}
        if self.labels is not None: d["label"] = int(self.labels[i])
        return d

def make_collate_fn(tok, max_len=512):
    def collate(batch):
        texts = [b["text"] for b in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "label" in batch[0]:
            out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out
    return collate

@torch.no_grad()
def evaluate_model(model, tok, test_file, class_names, batch_size=32, limit=None, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n🖥️  Device: {device}", flush=True); model.to(device)
    t0 = time.time(); print(f"→ Reading parquet: {test_file}", flush=True)
    df = pd.read_parquet(test_file, engine="pyarrow")
    if limit: df = df.iloc[:limit].copy(); print(f"  • Limit: {len(df):,} rows", flush=True)
    print(f"✅  Loaded {len(df):,} rows in {time.time()-t0:.2f}s\n", flush=True)
    print("First rows:"); print(df.head(3)[["title","label"]], flush=True)
    ds = ParquetTextDataset(df); has_labels = ds.labels is not None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(tok))
    print(f"\n🚀 Starting evaluation | samples={len(df):,} | batch_size={batch_size}\n", flush=True)
    preds, golds = [], []
    for batch in tqdm(loader, desc="Evaluating", mininterval=1.0):
        ids = batch["input_ids"].to(device); attn = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        if has_labels: golds.extend(batch["labels"].cpu().tolist())
    if has_labels:
        idx = list(range(len(class_names)))
        acc = accuracy_score(golds, preds)
        pr, rc, f1, sup = precision_recall_fscore_support(golds, preds, labels=idx, average=None, zero_division=0)
        cm = confusion_matrix(golds, preds, labels=idx)
        print(f"\n🎯  Test Accuracy: {acc:.4f}\n", flush=True)
        print("Per-class metrics:", flush=True)
        for i, n in enumerate(class_names):
            print(f"{i:2d} {n:22s}  P={pr[i]:.4f}  R={rc[i]:.4f}  F1={f1[i]:.4f}  (n={sup[i]})", flush=True)
        fig, ax = plt.subplots(figsize=(12,10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(include_values=True, xticks_rotation=45, cmap="Blues", ax=ax, colorbar=True)
        plt.title("DBpedia-14 Confusion Matrix"); plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("\n🖼️  Saved confusion_matrix.png", flush=True)
    else:
        out = pd.DataFrame({"title": df.get("title", pd.Series([None]*len(preds))),
                            "pred_label_id": preds,
                            "pred_label_name": [class_names[i] for i in preds]})
        out.to_parquet("predictions.parquet", index=False)
        print("💾  Saved predictions.parquet (no labels).", flush=True)

def main():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned DistilBERT")
    p.add_argument("test_file", help="Path to parquet (e.g., Data/dataset.parquet)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Evaluate on first N rows")
    args = p.parse_args()
    if not os.path.exists(args.test_file):
        print(f"\n❌  File not found: {args.test_file}\nExample: python evaluate.py Data/dataset.parquet\n", flush=True)
        sys.exit(1)
    print(f"✅  Using test file: {args.test_file}\nLoading model and tokenizer…\n", flush=True)
    model, tok, names = load_model_and_tokenizer(args.test_file)
    print("✅  Model and tokenizer loaded.\n", flush=True)
    evaluate_model(model, tok, args.test_file, names, batch_size=args.batch_size, limit=args.limit)
    print("\n✅  Done — check confusion_matrix.png or predictions.parquet\n", flush=True)

if __name__ == "__main__":
    main()