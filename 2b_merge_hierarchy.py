import json
import sys

# Define labels
LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]

# Get indices
LY_INDEX = LABELS.index("LY")
ID_INDEX = LABELS.index("ID")
SP_INDEX = LABELS.index("SP")
OP_INDEX = LABELS.index("OP")
IN_INDEX = LABELS.index("IN")
IP_INDEX = LABELS.index("IP")
HI_INDEX = LABELS.index("HI")

# Threshold
THRESHOLD = 0.70


def should_skip_first_level_op(probs):
    """Check if any of the blocking registers are active in the probabilities."""
    blocking_indices = [LY_INDEX, IN_INDEX, ID_INDEX, IP_INDEX, HI_INDEX]
    return any(probs[idx] >= THRESHOLD for idx in blocking_indices)


def process_file(input_path: str, output_path: str):
    """Process JSONL file and update the final probability list for each segment."""
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            # Parse the JSON object
            record = json.loads(line.strip())

            # Process each segment
            for segment in record["segments"]:
                prob_chain = segment["probs"]
                if not prob_chain:  # Skip if empty
                    continue

                # Get final probability list
                final_probs = prob_chain[-1]

                # Find maximum ID and SP probabilities across all levels
                max_id_prob = max(prob_list[ID_INDEX] for prob_list in prob_chain)
                max_sp_prob = max(prob_list[SP_INDEX] for prob_list in prob_chain)

                # Update ID and SP in the final list if the max is higher
                if max_id_prob > final_probs[ID_INDEX]:
                    final_probs[ID_INDEX] = max_id_prob
                if max_sp_prob > final_probs[SP_INDEX]:
                    final_probs[SP_INDEX] = max_sp_prob
                """
                # For OP, check if we should skip first level
                if should_skip_first_level_op(final_probs):
                    # Skip first level, only look at levels 1 onwards
                    max_op_prob = (
                        max(prob_list[OP_INDEX] for prob_list in prob_chain[1:])
                        if len(prob_chain) > 1
                        else final_probs[OP_INDEX]
                    )
                else:
                    # Use all levels
                    max_op_prob = max(prob_list[OP_INDEX] for prob_list in prob_chain)

                if max_op_prob > final_probs[OP_INDEX]:
                    final_probs[OP_INDEX] = max_op_prob
                """
                # Update the chain
                prob_chain[-1] = final_probs

                # Update the segment's probability chain
                segment["probs"] = prob_chain

            # Write the updated record
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl> <output_jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_file(input_path, output_path)
    print("Processing complete!")
