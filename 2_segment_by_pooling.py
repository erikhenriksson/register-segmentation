import glob
import json
import sys

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import labels


class TextSegmenter:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.model = self.model.to("cuda")
        self.model.eval()

    def get_embeddings_and_predict(self, text):
        """Get token embeddings and prediction for full text in one pass"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to("cuda")

            # Get hidden states and prediction
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get last hidden state and attention mask
            hidden_states = outputs.hidden_states[-1][0]  # Remove batch dimension
            attention_mask = inputs["attention_mask"][0]

            # Get the prediction for full text
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            return hidden_states, attention_mask, probs

    def truncate_text(self, text):
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        if len(tokens) > 2048:
            text = self.tokenizer.decode(tokens[:2048], skip_special_tokens=True)
        return text

    def mean_pool_and_predict(
        self, hidden_states, attention_mask, start_token, end_token
    ):
        """Apply mean pooling to a segment and get prediction"""
        with torch.no_grad():
            # Get segment embeddings and mask
            segment_states = hidden_states[start_token:end_token]
            segment_mask = attention_mask[start_token:end_token]

            # Apply mean pooling
            segment_mask = segment_mask.unsqueeze(-1)
            pooled = (segment_states * segment_mask).sum(dim=0) / segment_mask.sum()

            # Get prediction through classification head
            pooled = pooled.unsqueeze(0)  # Add batch dimension
            logits = self.model.classifier(self.model.head(pooled))
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            return probs

    def segment_recursively(self, text):
        # First, get token embeddings and full text prediction
        hidden_states, attention_mask, parent_probs = self.get_embeddings_and_predict(
            text
        )

        # Get token to char mapping for segmentation
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping

        # Base case
        if len(text) < 1000:
            return [
                (text, parent_probs, None)
            ]  # None for embedding since we're not using it

        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [(text, parent_probs, None)]

        best_gain = 0
        best_segments = None
        best_segment_probs = None

        # Try different split points
        for split_idx in range(1, len(sentences)):
            segment1 = " ".join(sentences[:split_idx])
            segment2 = " ".join(sentences[split_idx:])

            if len(segment1) >= 500 and len(segment2) >= 500:
                # Find token indices for segments
                seg1_end = next(
                    i
                    for i, (_, end) in enumerate(offset_mapping)
                    if end >= len(segment1)
                )
                seg2_start = seg1_end

                # Get predictions for segments using mean pooling
                probs1 = self.mean_pool_and_predict(
                    hidden_states, attention_mask, 0, seg1_end
                )
                probs2 = self.mean_pool_and_predict(
                    hidden_states, attention_mask, seg2_start, len(attention_mask)
                )

                gain = self.compute_gain(parent_probs, [probs1, probs2])

                if gain > best_gain:
                    best_gain = gain
                    best_segments = (segment1, segment2)
                    best_segment_probs = (probs1, probs2)

        if best_segments is None:
            return [(text, parent_probs, None)]

        # Recursively segment
        seg1_text, seg2_text = best_segments
        seg1_probs, seg2_probs = best_segment_probs

        return self.segment_recursively(seg1_text) + self.segment_recursively(seg2_text)


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


def get_last_processed_id():
    try:
        with open("segmentations.jsonl", "r", encoding="utf-8") as f:
            last_line = None
            for line in f:
                last_line = line
            if last_line:
                return json.loads(last_line)["id"]
    except FileNotFoundError:
        pass
    return -1


def main(model_path, dataset_path, output_path):
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

    last_id = get_last_processed_id()
    segmenter = TextSegmenter(model_path=model_path)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = segmenter.truncate_text(row["text"])
            full_probs, full_embedding = segmenter.get_probs_and_embedding(text)
            segments = segmenter.segment_recursively(text)
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
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <model_path> <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
