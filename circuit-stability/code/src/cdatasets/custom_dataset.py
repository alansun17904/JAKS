"""arith_dataset.py
Generates a dataset of random arithmetic problems of various lengths with
different operations and outputs them as a json file.
"""


import random
import json
import string
from pathlib import Path
import numpy as np
from functools import partial
import re


from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate


from torch.utils.data import DataLoader

class CustomDataset(BaseDataset):
    description = "Solve the following arithmetic problems."
    data_file = "custom.json"

    def __init__(self, n=5, append_ans=True):
        super().__init__()
        #self.append_ans = append_ans
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples =  []
        self._labels = []
        manual_mode = True

        # Manual single-example mode
        if manual_mode:
            clean_str = "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 1 1 11 11\\nPossible next steps:\\n"
            label_str = '14 − 2 = 12 (left: 8 8 12)'
            corrupted_str = "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 6 7 8 9\\nPossible next steps:\\n"

            self._examples = [{"input": clean_str, "target": label_str}]
            self._clean_examples = [clean_str]
            self._corrupted_examples = [corrupted_str if corrupted_str else clean_str]
            self._labels = [label_str]

    @property
    def examples(self):
        return self._examples

    def get_questions(self):

        if self._examples:
            return

        with open(Path(__file__).parent / "data" / self.data_file, encoding="utf-8") as f:
            task = json.load(f)  # list[dict]

        self._examples = []
        for ex in task:
            single_input = ex["Input"].strip().replace("\n", "").replace("\t", "")
            single_target = ex["labels"][1]

            self._examples.append({"input": single_input, "target": single_target})

        random.shuffle(self._examples)
        self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter):
        if formatter.name == "chain-of-thought":
            raise NotImplementedError(
                "Chain-of-thought not supported for arithmetic problems."
            )
         # Manual mode — nothing to format
        if self._examples and self._clean_examples:
            return

        Qs = [v["input"] for v in self._examples]
        As = [""] * len(self._examples) # just a filler to not break code

        self._clean_examples = [
            formatter.format(self.description, ex["input"], questions=Qs, answers=As)
            for ex in self._examples
        ]

        self._labels = [ex["target"] for ex in self._examples]  # <-- list of lists
        self._corrupted_examples = self._clean_examples[:]
        random.shuffle(self._corrupted_examples)

        #Qs, As = [v["input"] for v in self._examples], [
        #    v["target"] for v in self._examples
        #]


    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )
