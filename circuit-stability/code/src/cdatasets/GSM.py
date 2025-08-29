## built hopefully similar to custom_dataset.py


import random
import json
import re
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader, Subset

class GSM8KDataset(BaseDataset):
    description = """You are solving grade school math word problems. Solve the problem step by step and provide the final answer."""
    data_file = "gsm8k.json"  # You'll need to download/prepare this file


    def __init__(self, n=5, append_ans=True):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []
        manual_mode = False    

        if manual_mode:
                clean_str = "Natalie has 3 apples. She buys 5 more apples from the store. How many apples does she have now?"
                label_str = "8"
                corrupted_str = "Natalie has 3 apples. She buys 5 more apples from the store. She now has 13 apples."
                self._examples = [{"input": clean_str, "target": label_str}]
                self._clean_examples = [clean_str]
                self._corrupted_examples = [corrupted_str if corrupted_str else clean_str]
                self._labels = [label_str]


    def examples(self):
        return self._examples

    def get_questions(self):
        if self._examples:
            return None

        with open(Path(__file__).parent / "data" / self.data_file, encoding="utf-8") as f:
            data = json.load(f)


        if isinstance(data, dict):
            task = data.get("train")  #GSM8K comes with train test split I believe
            if task is None:
                # common alternates: 'test', 'data'
                task = data.get("test") or data.get("data")
                if task is None:
                    # fallback: first list value in the dict
                    task = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            task = data

        self._examples = []
        for ex in task:
            if "question" in ex and "answer" in ex:
                q = ex["question"].strip().replace("\n", " ").replace("\t", " ")
                a = self._extract_answer(ex["answer"])
                self._examples.append({"input": q, "target": a})


            elif "Input" in ex and "Output" in ex:
                q = ex["Input"].strip().replace("\n", " ").replace("\t", " ")
                a = str(ex["Output"]).strip()
                self._examples.append({"input": q, "target": a})

        print(self._examples)
    

    def _extract_answer(self, answer_text: str) -> str:
        if not answer_text:
            return ""
        txt = str(answer_text).replace("−", "-").replace("$", "")
        m = re.search(r"####\s*(.+)$", txt.strip())
        if m:
            short = m.group(1).strip().replace(",", "")
            return short
        nums = re.findall(r"[+\-]?\d+(?:\.\d+)?", txt.replace(",", ""))
        return nums[-1] if nums else txt.strip()
    

    def format_questions(self, formatter: PromptFormatter):
        if formatter.name == "chain-of-thought":
            raise NotImplementedError("Chain-of-thought not supported for GSM8K.")

        if self._examples and self._clean_examples:
            return None

        Qs = [v["input"] for v in self._examples]
        As = [""] * len(self._examples) 

        self._clean_examples = [
            formatter.format(self.description, ex["input"], questions=Qs, answers=As)
            for ex in self._examples
        ]

        self._labels = [ex["target"] for ex in self._examples]
        print(self._labels)

        corrupted_prompt = (
            "Q: A class has 3 students and 4 more join. How many now?\n"
            "Possible steps:\n"
            "3 + 4 = 8\n"
            "Final answer: 8"
        )
        self._corrupted_examples = [corrupted_prompt] * len(self._clean_examples)

    def to_dataloader(self, model, batch_size: int, collate_fn=None, indices=None):
        collate_fn = partial(generic_collate, model)
        ds = self if indices is None else Subset(self, indices)
        return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )