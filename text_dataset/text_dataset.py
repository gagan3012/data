from collections import defaultdict
import datasets
from glob import glob
import json
import pandas as pd
import os
import re


def organize_files(file_paths):
    organized_files = defaultdict(lambda: defaultdict(list))
    for path in file_paths:
        parts = path.split("/")
        task_name = parts[-2]  # Change this index based on your path structure
        split_type = parts[-1].split(".")[0]
        organized_files[task_name][split_type].append(path)
    return {k: dict(v) for k, v in organized_files.items()}


def read_large_file(file_object, block_size=1024*1024*1024):
    """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k."""
    while True:
        data = file_object.read(block_size)
        if not data:
            break
        yield data


clean_text_pattern = re.compile(r'[^a-zA-Z0-9\s]')


def clean_text(text):
    return clean_text_pattern.sub('', text)


class TextDataConfig(datasets.BuilderConfig):
    def __init__(self, name, data_files, **kwargs):
        super(TextDataConfig, self).__init__(name=name, **kwargs)
        self.data_files = data_files


class TextData(datasets.GeneratorBasedBuilder):
    """Text dataset."""

    def __init__(self, data_dir, **kwargs):
        super(TextData, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.data_files = glob(os.path.join(data_dir, "*.txt"))

    def _info(self):
        features = datasets.Features({
            "text": datasets.Value("string"),
        })
    
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(self.data_files)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={
                                        "filepaths": downloaded_files})
        ]
        
    def _generate_examples(self, filepaths):
            yield from self.generate_examples_pretrain(filepaths)
        

    def generate_examples_pretrain(self, filepaths):
        key = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for chunk in read_large_file(f):
                    for text in chunk.strip().split("\n"):
                        text = clean_text(text)
                        yield key, {
                            "text": text,
                        }
                        key += 1