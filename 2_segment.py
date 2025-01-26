import glob
import json
import sys

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

labels_structure = {
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}
labels = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]


class TextSegmenter:
    def __init__(self, model_path):
        nltk.download("punkt")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.model = self.model.to("cuda")
        self.model.eval()
        self.model.config.output_hidden_states = True

    def get_probs_and_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to("cuda")
            outputs = self.model(**inputs)
            probs = [float(p) for p in torch.sigmoid(outputs.logits).cpu().numpy()[0]]
            embedding = [
                float(e) for e in outputs.hidden_states[-1][0, 0].cpu().numpy()
            ]  # CLS token
            return probs, embedding

    def split_to_sentences(self, text):
        return sent_tokenize(text)

    def compute_gain(self, parent_probs, segment_probs):
        return (max(segment_probs[0]) + max(segment_probs[1])) / 2 - max(parent_probs)

    def segment_recursively(self, text):
        sentences = self.split_to_sentences(text)
        if len(sentences) < 2 or len(text) < 600:
            probs, embedding = self.get_probs_and_embedding(text)
            return [(text, probs, embedding)]

        parent_probs, _ = self.get_probs_and_embedding(text)
        best_gain = 0
        best_segments = None

        for split_idx in range(1, len(sentences)):
            segment1 = " ".join(sentences[:split_idx])
            segment2 = " ".join(sentences[split_idx:])

            if len(segment1) >= 300 and len(segment2) >= 300:
                probs1, _ = self.get_probs_and_embedding(segment1)
                probs2, _ = self.get_probs_and_embedding(segment2)
                gain = self.compute_gain(parent_probs, [probs1, probs2])

                if gain > best_gain:
                    best_gain = gain
                    best_segments = (segment1, segment2)

        if best_segments is None:
            probs, embedding = self.get_probs_and_embedding(text)
            return [(text, probs, embedding)]

        seg1_text, seg2_text = best_segments
        segments1 = self.segment_recursively(seg1_text)
        segments2 = self.segment_recursively(seg2_text)
        return segments1 + segments2


def print_result(item, threshold=0.35):
    print(f"\n---- Text [{item['id']}] ----")
    print(f"True label: {item['label']}")

    text_pred_labels = [
        labels[i] for i, p in enumerate(item["text_probs"]) if p > threshold
    ]
    print(f"Pred label: {', '.join(text_pred_labels)}")

    for j, seg in enumerate(item["segments"], 1):
        pred_labels = [labels[i] for i, p in enumerate(seg["probs"]) if p > threshold]
        print(f"Segment {j} [{', '.join(pred_labels)}]: {seg['text'][:1000]}...")


def main(model_path, dataset_path):
    all_data = []
    for tsv_file in glob.glob(f"{dataset_path}/*.tsv"):
        df = pd.read_csv(
            tsv_file,
            sep="\t",
            header=None,
            names=["label", "text"],
            na_values="",
            keep_default_na=False,
        )
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)

    segmenter = TextSegmenter(model_path=model_path)

    with open("segmentations.jsonl", "w", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            full_probs, full_embedding = segmenter.get_probs_and_embedding(row["text"])
            segments = segmenter.segment_recursively(row["text"])
            result = {
                "id": i,
                "label": row["label"],
                "text_probs": full_probs,
                "text_embedding": full_embedding,
                "segments": [
                    {"text": text, "probs": probs, "embedding": emb}
                    for text, probs, emb in segments
                ],
            }
            print_result(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <model_path> <dataset_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
