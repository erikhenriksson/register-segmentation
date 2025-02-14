import json
import os
import time

labels = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]

evaluation_instructions = """Evaluation Options:

5 (Perfect): Correct/almost correct
4 (Good): Mostly correct
3 (Ok): Notable issues
2 (Poor): Major issues
1 (Wrong): Incorrect
"""


class SegmentEvaluator:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_segments(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def get_last_evaluated_position(self, evaluator_id):
        output_file = os.path.join(self.output_dir, f"evaluations_{evaluator_id}.jsonl")
        if not os.path.exists(output_file):
            return 0
        with open(output_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")
        # Small delay to ensure screen is cleared
        time.sleep(0.2)

    def display_segment(self, segment, current_pos, total_segments):
        """Display a single segment with formatting"""
        self.clear_screen()
        print(f"Evaluating segment {current_pos} of {total_segments}\n")
        print("=" * 80)
        print(f"Text ID: {segment['id']} [Manual label: {segment['label']})")
        print("=" * 80)

        for i, seg in enumerate(segment["segments"], 1):
            # Handle hierarchical probabilities
            register_chain = []
            if "registers" not in seg:
                for prob_array in seg[
                    "probs"
                ]:  # Iterate through each probability array in the hierarchy
                    # For each level, get all labels that exceed threshold
                    level_labels = [
                        labels[j] for j, prob in enumerate(prob_array) if prob > 0.7
                    ]
                    if level_labels:  # Only add non-empty levels
                        register_chain.append(level_labels)

                # Join the chain with ">" separators
                register_set = set(
                    [item for sublist in register_chain for item in sublist]
                )
                # pred_label = " > ".join(register_chain)
                pred_label = " ".join([x for x in labels if x in register_set])
                pred_label += "\n" + " > ".join([" ".join(x) for x in register_chain])
                pred_label = " ".join(register_chain[-1])
            else:
                pred_label = " ".join([x for x in labels if x in seg["registers"]])

            print(f"\nSegment {i} [{pred_label}]:")
            print(f"Text: {seg['text']}")
            print("-" * 80)

        print(evaluation_instructions)

    def get_single_evaluation(self, evaluation_type):
        """Get a single evaluation score"""
        while True:
            prompt = f"Your evaluation for {evaluation_type.upper()} (1-5 or q): "
            choice = input(prompt).strip().lower()

            if choice == "q":
                return None
            try:
                score = int(choice)
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass

            print("\nInvalid input. Please enter a single digit (1-5), or q.")

    def get_sequential_evaluation(self):
        """Get label and segment evaluations sequentially"""
        # First get label evaluation
        label_score = self.get_single_evaluation("labels")
        if label_score is None:
            return None

        # Then get segment evaluation
        segment_score = self.get_single_evaluation("segments")
        if segment_score is None:
            return None

        self.clear_screen()

        # Return both scores as a tuple
        return (label_score, segment_score)

    def save_evaluation(self, evaluator_id, segment_id, evaluation):
        output_file = os.path.join(self.output_dir, f"evaluations_{evaluator_id}.jsonl")
        evaluation_record = {
            "segment_id": segment_id,
            "label_score": evaluation[0],
            "segment_score": evaluation[1],
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(evaluation_record) + "\n")

    def run_evaluation(self):
        self.clear_screen()
        print("Welcome to the Segmentation Evaluator")
        evaluator_id = input("Name: ").strip()

        segments = self.load_segments()
        start_position = self.get_last_evaluated_position(evaluator_id)

        if start_position >= len(segments):
            self.clear_screen()
            print("\nYou have completed all available segments!")
            return

        self.clear_screen()
        print(f"\nStarting from position {start_position + 1} of {len(segments)}")
        input("Press Enter to continue...")

        try:
            for i, segment in enumerate(segments[start_position:], start_position):
                self.display_segment(segment, i + 1, len(segments))
                evaluation = self.get_sequential_evaluation()

                if evaluation is None:  # User quit
                    self.clear_screen()
                    break

                self.save_evaluation(evaluator_id, segment["id"], evaluation)

        except KeyboardInterrupt:
            self.clear_screen()
            print("\nEvaluation interrupted. Progress has been saved.")
            return

        self.clear_screen()
        print("\nThank you for completing the evaluation!")
        print(f"Your evaluations have been saved in: {self.output_dir}")


def main():
    evaluator = SegmentEvaluator(
        input_file="sample.jsonl",
        output_dir=".",
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
