from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain
import os
from transformers.testing_utils import CaptureLogger
import logging
from squeakily.filter import check_char_repetition, check_flagged_words
from squeakily.clean import remove_empty_lines, normalize_whitespace
from squeakily.filter import minhash_dedup
from squeakily.core import Pipeline
import transformers
from fire import Fire
import shutil

logger = logging.getLogger(__name__)


def main(dataset_dir, data_files, data_cache_dir, save_dir,
         do_tokenize=True,
         block_size=512,
         tokenizer_name=""):
    # block_size = 512
    # dataset_dir = "/lustre07/scratch/gagan30/arocr/meta-llama/fin_project/fin_data"
    # dataset_config_name = "pretraining"
    # data_cache_dir = "/lustre07/scratch/gagan30/arocr/cache"
    preprocessing_num_workers = os.cpu_count()
    streaming = False
    # validation_split_percentage = 0.00001
    # do_tokenize = True

    print(f"Loading dataset from {dataset_dir}"
          f" with config {data_files}"
          f" and cache dir {data_cache_dir}"
          f" and preprocessing num workers {preprocessing_num_workers}"
          f" and streaming {streaming}"
          f" and do_tokenize {do_tokenize}"
          f" and block_size {block_size}"
          f" and tokenizer_name {tokenizer_name}")

    raw_dataset = load_dataset(dataset_dir, data_files=data_files,
                               cache_dir=data_cache_dir, split="train",
                               num_proc=preprocessing_num_workers,
                               keep_in_memory=False,
                               streaming=streaming)

    print("Original Dataset:", raw_dataset)

    datasources = [
        {
            "dataset": raw_dataset,
            "name": "pretraining",
            "columns": ["text"],
            "filters": [check_char_repetition, check_flagged_words],
            "cleaners": [remove_empty_lines, normalize_whitespace],
        },
    ]

    pipeline = Pipeline(datasources)
    pipeline.run(global_filters=[minhash_dedup])

    raw_dataset = pipeline.datasources[0]["dataset"]

    print("Deduped Dataset:", raw_dataset)

    if do_tokenize:
        tok_logger = transformers.utils.logging.get_logger(
            "transformers.tokenization_utils_base")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples["text"])
                # clm input could be much much longer than block_size
                if "Token indices sequence length is longer than the" in cl.out:
                    tok_logger.warning(
                        "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                        " before being passed to the model."
                    )
            return output

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=["text", "meta_data", "__id__"],
            load_from_cache_file=True,
            keep_in_memory=False,
            # cache_file_names = {k: os.path.join(data_cache_dir, f'tokenized_fin_pretrain.arrow') for k in raw_dataset},
            desc="Running tokenizer on dataset",
        )

        print("Tokenized Dataset:", tokenized_dataset)
        grouped_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=True,
            keep_in_memory=False,
            # cache_file_names = {k: os.path.join(data_cache_dir, f'grouped_fin_pretrain.arrow') for k in tokenized_dataset},
            desc=f"Grouping texts in chunks of {block_size}",
        )
        print("Grouped Dataset:", grouped_datasets)

        lm_datasets = grouped_datasets
    else:
        lm_datasets = raw_dataset

    print("Final Dataset:", lm_datasets)

    # save_dir = "/lustre07/scratch/gagan30/arocr/meta-llama/fin_project/pretrain_data"
    print(f"Saving pretraining data to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    save_at = os.path.join(save_dir)
    if os.path.exists(save_at):
        shutil.rmtree(save_at, ignore_errors=True)
    save_workers = preprocessing_num_workers if preprocessing_num_workers < len(
        lm_datasets) else len(lm_datasets)
    lm_datasets.save_to_disk(save_at, num_proc=save_workers)
    # lm_datasets["train"].save_to_disk(os.path.join(data_cache_dir, f'fin_pretrain_train'))


if __name__ == "__main__":
    Fire(main)
