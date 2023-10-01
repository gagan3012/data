import os
import json
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count

directory_path = '/lustre07/scratch/gagan30/arocr/meta-llama/data/modelzoo/modelzoo/transformers/data_processing/slimpajama/ArabicText-2022/008.txt'


def process_file(filename):
    txt_file_path = filename

    # Read the text file
    with open(txt_file_path, 'r') as infile:
        lines = infile.readlines()

    # Determine the output JSONL filename
    jsonl_file_path = os.path.splitext(txt_file_path)[0] + '.jsonl'

    # Write to the JSONL file
    with open(jsonl_file_path, 'w') as outfile:
        for line in lines:
            obj = {"text": line.strip()}
            outfile.write(json.dumps(obj) + '\n')

    # Delete the original .txt file
    os.remove(txt_file_path)


if __name__ == "__main__":
    files = glob(directory_path)

    # Use 4 cores (or all available cores if less than 4)
    num_cores = max(4, cpu_count())

    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_file, files), total=len(files)))

    print("Conversion completed!")

import pandas as pd
from glob import glob
from tqdm import tqdm
# Specify the directory (for this example, it's the current directory)
directory_path = 'ArabicText-2022/008.txt'

# Entry to add
meta_entry = {"redpajama_set_name": "ar"}

for filename in tqdm(glob(directory_path)):
    df = pd.read_json(filename, lines=True)
    df['meta'] = [meta_entry for _ in range(len(df))]
    df = df.dropna()
    # print(df.head())
    df.to_json(filename, orient='records', lines=True)

import jsonlines
import argparse
from os import path, remove


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split a JSONL file into smaller chunks.')
    parser.add_argument('input_file', type=str,
                        help='Path to the input JSONL file.')
    parser.add_argument('output_dir', type=str,
                        help='Directory to store the output JSONL files.')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Number of records per chunk.')
    return parser.parse_args()


def split_jsonl(input_file, output_dir, chunk_size):
    with jsonlines.open(input_file) as reader:
        records = list(reader)

    total_records = len(records)
    num_chunks = (total_records // chunk_size) + \
        (1 if total_records % chunk_size > 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_records = records[start_idx:end_idx]
        output_file = path.join(output_dir, f'008_chunk_{i}.jsonl')

        with jsonlines.open(output_file, mode='w') as writer:
            for record in chunk_records:
                writer.write(record)

    print(f'Split {total_records} records into {num_chunks} chunks.')

    remove(input_file)

if __name__ == '__main__':
    args = parse_args()
    split_jsonl(args.input_file, args.output_dir, args.chunk_size)
