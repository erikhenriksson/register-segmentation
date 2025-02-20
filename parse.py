import json
import trankit
from typing import Dict, List, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_texts(
    input_file: str, full_text_output: str, segment_output: str, lang="english"
):
    """Process texts using trankit and save results to JSONL files."""
    # Initialize trankit
    nlp = trankit.Pipeline(lang)

    # Load input data
    data = load_jsonl(input_file)

    # Process full texts
    full_text_results = []
    segment_results = []

    for item in data:
        # Process full text
        parsed = nlp.posdep(" ".join(segment["text"] for segment in item["segments"]))

        # Create full text entry with probabilities and parsed info
        full_text_entry = {
            "id": item["id"],
            "text_probs": item["text_probs"],
            "parsed": parsed,
        }
        full_text_results.append(full_text_entry)

        # Process segments
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
            segment_results.append(segment_entry)

    # Save results
    save_jsonl(full_text_results, full_text_output)
    save_jsonl(segment_results, segment_output)


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "s_merged.jsonl"  # Your input file
    FULL_TEXT_OUTPUT = "full_text_parsed.jsonl"
    SEGMENT_OUTPUT = "segment_parsed.jsonl"
    LANGUAGE = "english"  # Or whatever language your texts are in

    # Process the texts
    process_texts(INPUT_FILE, FULL_TEXT_OUTPUT, SEGMENT_OUTPUT, LANGUAGE)
