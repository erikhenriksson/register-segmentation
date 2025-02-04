import sys
import json
import glob
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from scipy.spatial.distance import cosine

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 8192  #
    classification_threshold: float = (
        0.35  # Increase for more confident register assignments
    )
    min_sentences: int = 3  # Decrease to allow shorter segments
    max_sentences: int = 150  # Increase to handle longer coherent sections
    window_sentences: int = 7  # Increase for more context
    stride: int = 3  # Increase to reduce computational overhead
    merge_threshold: float = 0.25  # Lower to detect subtle register shifts


class MultiScaleSegmenter:
    def __init__(self, model_path: str, config: MultiScaleConfig = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )
        self.config = config or MultiScaleConfig()

        # Cache for token representations and offset mappings
        self.token_embeddings = None
        self.attention_mask = None
        self.tokens = None
        self.offset_mapping = None

        # Add cache for register probabilities
        self._prob_cache = {}
        self._embedding_cache = {}

    def get_span_embedding(self, start_token: int, end_token: int) -> torch.Tensor:
        """Get mean-pooled embedding for token span using cached embeddings."""
        cache_key = (start_token, end_token)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document before get_span_embedding")

        # Use safe mean pooling on the span
        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        embedding = self.safe_mean_pooling(span_embeddings, span_mask)

        # Cache the embedding
        self._embedding_cache[cache_key] = embedding
        return embedding

    def get_register_probs(
        self, text: str = None, start_token: int = None, end_token: int = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Get register probabilities and embedding for a text span."""
        if self.token_embeddings is None:
            if text is None:
                raise ValueError(
                    "Must either provide text or call prepare_document first"
                )
            self.prepare_document(text)
            start_token = 0
            end_token = len(self.token_embeddings)

        # If no span specified, use entire sequence
        if start_token is None or end_token is None:
            start_token = 0
            end_token = len(self.token_embeddings)

        # Check cache
        cache_key = (start_token, end_token)
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key], self._embedding_cache[cache_key]

        # Get span embedding
        span_embedding = self.get_span_embedding(start_token, end_token)

        # Get probabilities
        with torch.no_grad():
            hidden = self.model.head(span_embedding.unsqueeze(0))
            logits = self.model.classifier(hidden)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        # Cache results
        self._prob_cache[cache_key] = probs

        return probs, span_embedding

    def prepare_document(self, text: str):
        """Run model once for whole document and cache results."""
        # Clear caches when preparing new document
        self._prob_cache = {}
        self._embedding_cache = {}

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        inputs = {
            k: v.to("cuda") if k != "offset_mapping" else v for k, v in inputs.items()
        }

        self.tokens = inputs["input_ids"][0]
        outputs = self.model(**inputs)
        self.token_embeddings = outputs.hidden_states[-1][0].detach()
        self.attention_mask = inputs["attention_mask"][0]
        self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray, torch.Tensor]]:
        """Main entry point for text segmentation."""
        text = self.truncate_text(text)

        sent_detector = PunktSentenceTokenizer()
        sent_char_spans = list(sent_detector.span_tokenize(text))
        sentences = [text[s:e] for s, e in sent_char_spans]

        self.prepare_document(text)

        offset_mapping = np.array(self.offset_mapping)

        sent_spans = []
        for char_start, char_end in sent_char_spans:
            token_start = None
            token_end = None
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end == 0:
                    continue
                if token_start is None and tok_end > char_start:
                    token_start = i
                if tok_start < char_end:
                    token_end = i + 1
                else:
                    break
            if token_start is None or token_end is None:
                token_start, token_end = 0, 0
            sent_spans.append((int(token_start), int(token_end)))

        if not sent_spans:
            probs, embedding = self.get_register_probs()
            return [(text, probs, embedding)]

        segments = self.segment_recursive(text, sentences, sent_spans)

        if len(segments) == 1:
            probs, embedding = self.get_register_probs()
            segments = [(text, probs, embedding)]

        return segments

    def segment_recursive(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> List[Tuple[str, np.ndarray, torch.Tensor]]:
        """Recursively segment text using binary splitting."""
        if len(sentences) < 2 * self.config.min_sentences:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(
                start_token=start_token, end_token=end_token
            )
            return [(span_text, probs, embedding)]

        split_idx, score = self.find_best_split(text, sentences, sent_spans)

        if score < self.config.min_register_diff or split_idx is None:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(
                start_token=start_token, end_token=end_token
            )
            return [(span_text, probs, embedding)]

        left_segments = self.segment_recursive(
            text, sentences[:split_idx], sent_spans[:split_idx]
        )
        right_segments = self.segment_recursive(
            text, sentences[split_idx:], sent_spans[split_idx:]
        )

        return left_segments + right_segments


def main(model_path, dataset_path, output_path):
    config = MultiScaleConfig()
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
    segmenter = MultiScaleSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            segments = segmenter.segment_text(text)

            # Get document-level predictions
            text_probs, text_embedding = segmenter.get_register_probs(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in text_probs.tolist()],
                "text_embedding": text_embedding.tolist(),
                "segments": [
                    {
                        "text": text,
                        "probs": [round(x, 4) for x in probs.tolist()],
                        "embedding": emb.tolist(),
                    }
                    for text, probs, emb in segments
                ],
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            segmenter.print_result(result)
            f.flush()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <model_path> <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
