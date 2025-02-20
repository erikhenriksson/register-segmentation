import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Iterator, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iterate_jsonl(filename: str) -> Iterator[Dict]:
    """Iterate over JSONL file line by line"""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def batch_encode_texts(texts: List[str], tokenizer, model, batch_size: int = 32):
    """Encode a batch of texts and return their embeddings"""
    with torch.no_grad():
        input_data = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        batch_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    return batch_vectors.cpu().numpy().tolist()


def process_batch(
    batch_texts: List[str],
    batch_records: List[Dict],
    tokenizer,
    model,
    output_file: str,
):
    """Process a batch of texts and write results to output file"""
    if not batch_texts:
        return

    embeddings = batch_encode_texts(batch_texts, tokenizer, model)

    with open(output_file, "a", encoding="utf-8") as f:
        for record, embedding in zip(batch_records, embeddings):
            result = {
                "text": record["text"],
                "probs": record["probs"],
                "embedding": embedding,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def process_data(
    input_file: str, full_text_output: str, segment_output: str, batch_size: int = 32
):
    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "NovaSearch/stella_en_400M_v5", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "NovaSearch/stella_en_400M_v5",
        trust_remote_code=True,
        use_memory_efficient_attention=False,
        unpad_inputs=False,
    )
    model.cuda()
    model.eval()

    # Clear output files if they exist
    open(full_text_output, "w").close()
    open(segment_output, "w").close()

    # Process full texts
    logger.info("Processing full texts...")
    batch_texts = []
    batch_records = []

    for item in tqdm(iterate_jsonl(input_file)):
        full_text = " ".join(segment["text"] for segment in item["segments"])

        batch_texts.append(full_text)
        batch_records.append({"text": full_text, "probs": item["text_probs"]})

        if len(batch_texts) >= batch_size:
            process_batch(
                batch_texts, batch_records, tokenizer, model, full_text_output
            )
            batch_texts = []
            batch_records = []

    # Process remaining full texts
    if batch_texts:
        process_batch(batch_texts, batch_records, tokenizer, model, full_text_output)

    # Process segments
    exit()
    logger.info("Processing segments...")
    batch_texts = []
    batch_records = []

    for item in tqdm(iterate_jsonl(input_file)):
        for segment in item["segments"]:
            batch_texts.append(segment["text"])
            batch_records.append(
                {
                    "text": segment["text"],
                    "probs": segment["probs"][
                        -1
                    ],  # Taking the last item from probs list
                }
            )

            if len(batch_texts) >= batch_size:
                process_batch(
                    batch_texts, batch_records, tokenizer, model, segment_output
                )
                batch_texts = []
                batch_records = []

    # Process remaining segments
    if batch_texts:
        process_batch(batch_texts, batch_records, tokenizer, model, segment_output)

    logger.info("Processing completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate semantic embeddings for texts and segments"
    )
    parser.add_argument(
        "--input",
        default="s_merged.jsonl",
        type=str,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--full-output",
        type=str,
        default="full_text_semantic.jsonl",
        help="Output file for full text embeddings",
    )
    parser.add_argument(
        "--segment-output",
        type=str,
        default="segment_semantic.jsonl",
        help="Output file for segment embeddings",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )

    args = parser.parse_args()

    process_data(args.input, args.full_output, args.segment_output, args.batch_size)
