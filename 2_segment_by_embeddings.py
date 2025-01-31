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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.model.eval()
        self.min_tokens = 128  # Set minimum token length
        self.threshold = 0.35  # Set threshold for segmenting

    def safe_mean_pooling(self, hidden_states, attention_mask):
        """Safe mean pooling that handles edge cases to prevent infinite values"""
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        attention_mask = attention_mask.unsqueeze(-1)
        mask_sum = attention_mask.sum(dim=0) + epsilon

        # Ensure the mask sum is at least epsilon to prevent explosion
        mask_sum = torch.maximum(
            mask_sum,
            torch.tensor(epsilon, device=mask_sum.device, dtype=hidden_states.dtype),
        )

        # Compute weighted sum and divide by mask sum
        weighted_sum = (hidden_states * attention_mask).sum(dim=0)
        pooled = weighted_sum / mask_sum

        # Clip any extreme values
        pooled = torch.clamp(pooled, min=-100.0, max=100.0)

        # Ensure output is in float16 to match model dtype
        pooled = pooled.to(dtype=torch.float16)

        return pooled

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

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]
            attention_mask = inputs["attention_mask"][0]
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            return hidden_states, attention_mask, probs

    def compute_gain(self, parent_probs, segment_probs):
        """Compute gain for segment split, ensuring each segment has at least one register"""
        max_seg1 = max(segment_probs[0])
        max_seg2 = max(segment_probs[1])
        max_parent = max(parent_probs)

        # Check that each segment has at least one probability above threshold
        has_register_seg1 = any(prob > self.threshold for prob in segment_probs[0])
        has_register_seg2 = any(prob > self.threshold for prob in segment_probs[1])

        if (
            max_seg1 > max_parent
            and max_seg2 > max_parent
            and has_register_seg1
            and has_register_seg2
        ):
            return (max_seg1 + max_seg2) / 2 - max_parent
        return 0

    def compute_gain_2(self, parent_probs, segment_probs):
        """Compute gain for segment split, ensuring each segment has at least one register and different registers from parent"""
        # Get registers for parent and segments using threshold
        parent_registers = {
            i for i, prob in enumerate(parent_probs) if prob > self.threshold
        }
        seg1_registers = {
            i for i, prob in enumerate(segment_probs[0]) if prob > self.threshold
        }
        seg2_registers = {
            i for i, prob in enumerate(segment_probs[1]) if prob > self.threshold
        }

        # Check max probabilities
        max_seg1 = max(segment_probs[0])
        max_seg2 = max(segment_probs[1])
        max_parent = max(parent_probs)

        # Return 0 if:
        # - Either segment doesn't have higher max prob than parent
        # - Either segment has no registers above threshold
        # - Both segments have exactly the same registers as parent
        if (
            max_seg1 <= max_parent
            or max_seg2 <= max_parent
            or not seg1_registers
            or not seg2_registers
            or (
                seg1_registers == parent_registers
                and seg2_registers == parent_registers
            )
        ):
            return 0

        return (max_seg1 + max_seg2) / 2 - max_parent

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
        with torch.no_grad():
            segment_states = hidden_states[start_token:end_token]
            segment_mask = attention_mask[start_token:end_token]

            # Use safe pooling
            pooled = self.safe_mean_pooling(segment_states, segment_mask)
            pooled = pooled.unsqueeze(0)

            logits = self.model.classifier(self.model.head(pooled))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            return probs

    def get_token_count(self, text):
        """Helper method to get token count for a text segment"""
        return len(self.tokenizer.encode(text)) - 2  # Subtract 2 for special tokens

    def compute_gain_semantic(self, parent_embedding, seg1_embedding, seg2_embedding):
        """
        Compute gain for segment split using only embedding dissimilarity.

        Args:
            parent_embedding: Hidden state embedding for parent segment
            seg1_embedding: Hidden state embedding for first child segment
            seg2_embedding: Hidden state embedding for second child segment
        """

        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

        # Calculate similarities
        seg1_parent_sim = cosine_similarity(parent_embedding, seg1_embedding)
        seg2_parent_sim = cosine_similarity(parent_embedding, seg2_embedding)
        seg1_seg2_sim = cosine_similarity(seg1_embedding, seg2_embedding)

        # We want:
        # 1. High dissimilarity between segments (1 - seg1_seg2_sim should be high)
        # 2. Some dissimilarity with parent (but not too much, as segments should still be related)
        semantic_gain = (1 - seg1_seg2_sim) - 0.5 * (
            (seg1_parent_sim + seg2_parent_sim) / 2
        )

        # Only split if segments are sufficiently different from each other
        if seg1_seg2_sim > 0.9:  # High similarity threshold
            return 0

        return (
            semantic_gain if semantic_gain > 0.1 else 0
        )  # Adjusted threshold for semantic-only approach

    def segment_recursively(self, text):
        """
        Recursively segment text based on semantic similarity.
        """
        # Get embeddings and probabilities for full text
        hidden_states, attention_mask, parent_probs = self.get_embeddings_and_predict(
            text
        )

        # Get embedding for full text
        full_text_embedding = self.safe_mean_pooling(hidden_states, attention_mask)
        full_text_embedding = full_text_embedding.cpu().numpy().tolist()

        # Check minimum token count
        if self.get_token_count(text) < self.min_tokens:
            return [(text, parent_probs, full_text_embedding)]

        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [(text, parent_probs, full_text_embedding)]

        best_gain = 0.1  # Adjusted threshold for semantic-only approach
        best_segments = None

        for split_idx in range(1, len(sentences)):
            segment1 = " ".join(sentences[:split_idx])
            segment2 = " ".join(sentences[split_idx:])

            # Check token counts
            if (
                self.get_token_count(segment1) >= self.min_tokens
                and self.get_token_count(segment2) >= self.min_tokens
            ):

                # Get token indices for segments
                seg1_tokens = self.tokenizer.encode(segment1, add_special_tokens=False)
                seg1_end = len(seg1_tokens)
                seg2_start = seg1_end

                # Get embeddings for segments
                seg1_states = hidden_states[:seg1_end]
                seg1_mask = attention_mask[:seg1_end]
                seg1_embedding = self.safe_mean_pooling(seg1_states, seg1_mask)
                seg1_embedding = seg1_embedding.cpu().numpy().tolist()

                seg2_states = hidden_states[seg2_start : len(attention_mask)]
                seg2_mask = attention_mask[seg2_start : len(attention_mask)]
                seg2_embedding = self.safe_mean_pooling(seg2_states, seg2_mask)
                seg2_embedding = seg2_embedding.cpu().numpy().tolist()

                # Get probabilities for segments (needed for return value)
                probs1 = self.mean_pool_and_predict(
                    hidden_states, attention_mask, 0, seg1_end
                )
                probs2 = self.mean_pool_and_predict(
                    hidden_states, attention_mask, seg2_start, len(attention_mask)
                )

                # Compute gain using only semantic similarity
                gain = self.compute_gain_semantic(
                    full_text_embedding, seg1_embedding, seg2_embedding
                )

                if gain > best_gain:
                    best_gain = gain
                    best_segments = (
                        (segment1, probs1, seg1_embedding),
                        (segment2, probs2, seg2_embedding),
                    )

        if best_segments is None:
            return [(text, parent_probs, full_text_embedding)]

        # Recursively segment the best segments found
        return self.segment_recursively(best_segments[0][0]) + self.segment_recursively(
            best_segments[1][0]
        )

    def print_result(self, item):
        print(f"\n---- Text [{item['id']}] ----")
        print(f"True label: {item['label']}")

        text_pred_labels = [
            labels[i] for i, p in enumerate(item["text_probs"]) if p > self.threshold
        ]
        print(f"Pred label: {', '.join(text_pred_labels)}")

        for j, seg in enumerate(item["segments"], 1):
            pred_labels = [
                labels[i] for i, p in enumerate(seg["probs"]) if p > self.threshold
            ]
            print(f"Segment {j} [{', '.join(pred_labels)}]: {seg['text']}")
            print("---")


def get_last_processed_id(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
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

    last_id = get_last_processed_id(output_path)
    segmenter = TextSegmenter(model_path=model_path)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = segmenter.truncate_text(row["text"])
            hidden_states, attention_mask, full_probs = (
                segmenter.get_embeddings_and_predict(text)
            )
            text_embedding = (hidden_states * attention_mask.unsqueeze(-1)).sum(
                dim=0
            ) / attention_mask.sum()
            text_embedding = text_embedding.cpu().numpy().tolist()

            # Get initial segments
            segments = segmenter.segment_recursively(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in full_probs.tolist()],
                "text_embedding": text_embedding,
                "segments": (
                    [
                        {
                            "text": text,
                            "probs": [round(x, 4) for x in probs.tolist()],
                            "embedding": emb,
                        }
                        for text, probs, emb in segments
                    ]
                ),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            segmenter.print_result(result)
            f.flush()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <model_path> <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
