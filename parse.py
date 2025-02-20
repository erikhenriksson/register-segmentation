import json
import trankit
from typing import Dict, Any
from tqdm import tqdm  # For progress tracking


def process_texts_streaming(
    input_file: str, full_text_output: str, segment_output: str, lang="english"
):
    """Process texts using trankit and save results to JSONL files incrementally."""
    # Initialize trankit
    nlp = trankit.Pipeline(lang, gpu=True)

    # Open both output files in write mode
    with open(input_file, "r", encoding="utf-8") as f_in, open(
        full_text_output, "w", encoding="utf-8"
    ) as f_full, open(segment_output, "w", encoding="utf-8") as f_segment:

        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))

        # Reset file pointer
        f_in.seek(0)

        # Process each line
        for line in tqdm(f_in, total=total_lines, desc="Processing texts"):
            item = json.loads(line)

            try:
                # Process full text
                parsed = nlp.posdep(
                    " ".join([segment["text"] for segment in item["segments"]])
                )

                # Create and write full text entry immediately
                full_text_entry = {
                    "id": item["id"],
                    "text_probs": item["text_probs"],
                    "parsed": parsed,
                }
                f_full.write(json.dumps(full_text_entry, ensure_ascii=False) + "\n")
                f_full.flush()  # Ensure it's written to disk

                # Process and write each segment immediately
                for segment in item["segments"]:
                    parsed_segment = nlp.posdep(segment["text"])

                    # Get the last probability array for the segment
                    segment_probs = segment["probs"][-1] if segment["probs"] else None

                    segment_entry = {
                        "id": item["id"],
                        "text": segment["text"],
                        "probs": segment_probs,
                        "parsed": parsed_segment,
                    }
                    f_segment.write(
                        json.dumps(segment_entry, ensure_ascii=False) + "\n"
                    )
                    f_segment.flush()  # Ensure it's written to disk

            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
                continue  # Skip to next item if there's an error


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "s_hierarchy.jsonl"  # Your input file
    FULL_TEXT_OUTPUT = "full_text_parsed.jsonl"
    SEGMENT_OUTPUT = "segment_parsed.jsonl"
    LANGUAGE = "english"  # Or whatever language your texts are in

    # Process the texts
    process_texts_streaming(INPUT_FILE, FULL_TEXT_OUTPUT, SEGMENT_OUTPUT, LANGUAGE)
