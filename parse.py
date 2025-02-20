import json
import trankit
from typing import Dict, Any, List
from tqdm import tqdm
import torch


def batch_items(items: List[Dict], batch_size: int):
    """Yield batch_size items at a time."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def process_texts_streaming(
    input_file: str,
    full_text_output: str,
    segment_output: str,
    lang="english",
    batch_size=32,
):
    """Process texts using trankit with GPU support and batch processing."""
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    device = "gpu" if use_gpu else "cpu"
    print(f"Using device: {device}")

    # Initialize trankit with GPU
    nlp = trankit.Pipeline(lang, cache_dir="./cache", gpu=use_gpu)

    with open(input_file, "r", encoding="utf-8") as f_in, open(
        full_text_output, "w", encoding="utf-8"
    ) as f_full, open(segment_output, "w", encoding="utf-8") as f_segment:

        # Process in batches
        batch_texts = []
        batch_items = []
        batch_segments = []

        for line in tqdm(f_in, desc="Reading and processing"):
            item = json.loads(line)
            batch_texts.append(" ".join([seg["text"] for seg in item["segments"]]))
            batch_items.append(item)

            # When batch is full or at end of file, process it
            if len(batch_texts) >= batch_size:
                try:
                    # Process full texts in batch
                    parsed_batch = nlp.posdep(batch_texts, is_sent=False)

                    # Write results for each item in batch
                    for item, parsed in zip(batch_items, parsed_batch):
                        # Write full text entry
                        full_text_entry = {
                            "id": item["id"],
                            "text_probs": item["text_probs"],
                            "parsed": parsed,
                        }
                        f_full.write(
                            json.dumps(full_text_entry, ensure_ascii=False) + "\n"
                        )

                        # Process segments for this item
                        segment_texts = [seg["text"] for seg in item["segments"]]
                        if segment_texts:  # Only process if there are segments
                            parsed_segments = nlp.posdep(segment_texts, is_sent=False)

                            for segment, parsed_seg in zip(
                                item["segments"], parsed_segments
                            ):
                                segment_entry = {
                                    "id": item["id"],
                                    "text": segment["text"],
                                    "probs": (
                                        segment["probs"][-1]
                                        if segment["probs"]
                                        else None
                                    ),
                                    "parsed": parsed_seg,
                                }
                                f_segment.write(
                                    json.dumps(segment_entry, ensure_ascii=False) + "\n"
                                )

                    # Flush both files
                    f_full.flush()
                    f_segment.flush()

                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

                # Clear batches
                batch_texts = []
                batch_items = []
                batch_segments = []

        # Process remaining items
        if batch_texts:
            try:
                parsed_batch = nlp.posdep(batch_texts, is_sent=False)
                for item, parsed in zip(batch_items, parsed_batch):
                    # Write full text entry
                    full_text_entry = {
                        "id": item["id"],
                        "text_probs": item["text_probs"],
                        "parsed": parsed,
                    }
                    f_full.write(json.dumps(full_text_entry, ensure_ascii=False) + "\n")

                    # Process segments
                    segment_texts = [seg["text"] for seg in item["segments"]]
                    if segment_texts:
                        parsed_segments = nlp.posdep(segment_texts, is_sent=False)
                        for segment, parsed_seg in zip(
                            item["segments"], parsed_segments
                        ):
                            segment_entry = {
                                "id": item["id"],
                                "text": segment["text"],
                                "probs": (
                                    segment["probs"][-1] if segment["probs"] else None
                                ),
                                "parsed": parsed_seg,
                            }
                            f_segment.write(
                                json.dumps(segment_entry, ensure_ascii=False) + "\n"
                            )
            except Exception as e:
                print(f"Error processing final batch: {str(e)}")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "s_hierarchy.jsonl"
    FULL_TEXT_OUTPUT = "full_text_parsed.jsonl"
    SEGMENT_OUTPUT = "segment_parsed.jsonl"
    LANGUAGE = "english"
    BATCH_SIZE = 32  # Adjust based on your GPU memory

    process_texts_streaming(
        INPUT_FILE, FULL_TEXT_OUTPUT, SEGMENT_OUTPUT, LANGUAGE, BATCH_SIZE
    )
