import os
import json
from tqdm import tqdm
from glob import glob
import jsonlines
from multiprocessing import Pool, cpu_count

CHUNK_SIZE = 1000


def txt_to_records(txt_file_path):
    """Read a .txt file and return a list of dictionaries."""
    with open(txt_file_path, 'r') as infile:
        lines = infile.readlines()

    return [{"text": line.strip()} for line in lines]


def split_and_save(records, output_dir, base_filename):
    """Split a list of records and save them in smaller chunks."""
    total_records = len(records)
    num_chunks = (total_records // CHUNK_SIZE) + \
        (1 if total_records % CHUNK_SIZE > 0 else 0)

    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        chunk_records = records[start_idx:end_idx]
        output_file = os.path.join(
            output_dir, f'{base_filename}_chunk_{i}.jsonl')

        with jsonlines.open(output_file, mode='w') as writer:
            for record in chunk_records:
                writer.write(record)


def process_file(args_tuple):
    txt_file, target_dir = args_tuple
    records = txt_to_records(txt_file)
    base_filename = os.path.splitext(os.path.basename(txt_file))[0]
    split_and_save(records, target_dir, base_filename)
    os.remove(txt_file)


def jsonize(args):
    txt_files = glob(os.path.join(args.data_dir, '*.txt'))

    # Create the target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)

    args_tuples = [(txt_file, args.target_dir) for txt_file in txt_files]

    # Number of cores for parallel processing
    num_cores = cpu_count()

    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_file, args_tuples), total=len(txt_files)))

    print(f"Processing of {args.data_dir} completed!")
