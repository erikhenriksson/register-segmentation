import json
import numpy as np
from typing import List, Dict, Any

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]
THRESHOLD = 0.7  # Threshold for considering a label as active


def get_active_registers(probs: List[float]) -> List[str]:
    """Get active registers based on probability threshold."""
    return [LABELS[i] for i, prob in enumerate(probs) if prob > THRESHOLD]


def are_registers_identical(regs1: List[str], regs2: List[str]) -> bool:
    """Check if two lists of registers are identical."""
    return set(regs1) == set(regs2)


def merge_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """
    Merge multiple embeddings by taking their average while ensuring (1, 1024) shape.
    """
    if not embeddings:
        return []

    # Convert to numpy array and ensure 2D shape
    embeddings_array = np.array([np.array(emb).reshape(-1) for emb in embeddings])
    # Take mean along first axis (across embeddings)
    mean_embedding = np.mean(embeddings_array, axis=0)
    # Reshape to ensure (1, 1024) shape and convert to list of lists
    return mean_embedding.reshape(1, -1).tolist()


def merge_probs(leaf_probs: List[List[float]]) -> List[List[float]]:
    """
    Merge leaf probabilities by averaging them.
    Returns a list containing only the averaged probabilities.
    """
    if not leaf_probs:
        return []
    # Average the leaf probabilities
    avg_probs = [[round(x, 8) for x in np.mean(leaf_probs, axis=0).tolist()]]
    return avg_probs


def process_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            entry = json.loads(line.strip())

            # Ensure text_embedding has correct shape
            text_embedding = np.array(entry["text_embedding"]).reshape(1, -1).tolist()

            # Process segments
            merged_segments = []
            current_texts = []
            current_leaf_probs = []
            current_embeddings = []
            current_registers = None

            for segment in entry["segments"]:
                # Ensure segment embedding has correct shape
                segment["embedding"] = (
                    np.array(segment["embedding"]).reshape(1, -1).tolist()
                )

                # Get the last probability list (leaf probabilities)
                leaf_probs = segment["probs"][-1]
                active_registers = get_active_registers(leaf_probs)

                if current_registers is None:
                    # First segment
                    current_texts = [segment["text"]]
                    current_leaf_probs = [leaf_probs]
                    current_embeddings = [segment["embedding"]]
                    current_registers = active_registers
                else:
                    if are_registers_identical(current_registers, active_registers):
                        # Collect segments for merging
                        current_texts.append(segment["text"])
                        current_leaf_probs.append(leaf_probs)
                        current_embeddings.append(segment["embedding"])
                    else:
                        # Save current merged segment and start new one
                        merged_segments.append(
                            {
                                "text": " ".join(current_texts),
                                "probs": merge_probs(current_leaf_probs),
                                "embedding": merge_embeddings(current_embeddings),
                            }
                        )
                        # Start new segment
                        current_texts = [segment["text"]]
                        current_leaf_probs = [leaf_probs]
                        current_embeddings = [segment["embedding"]]
                        current_registers = active_registers

            # Add the last segment if it exists
            if current_registers is not None:
                merged_segments.append(
                    {
                        "text": " ".join(current_texts),
                        "probs": merge_probs(current_leaf_probs),
                        "embedding": merge_embeddings(current_embeddings),
                    }
                )

            # Create output in the same format as input
            result = {
                "id": entry["id"],
                "label": entry["label"],
                "text_probs": entry["text_probs"],
                "text_embedding": text_embedding,
                "segments": merged_segments,
            }

            # Write to output file
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl> <output_jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_file(input_path, output_path)
    print("Processing complete!")
