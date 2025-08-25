"""custom_dataset.py
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

from torch.utils.data import DataLoader, Subset

class CustomDataset(BaseDataset):
    description = """You are solving the Game of 24. Given 4 numbers At each step, calculate the next best step"""
    data_file = "custom.json"

    def __init__(self, data_file=None, n=5, append_ans=True):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []
        
        # Load data from file if specified
        if data_file:
            self.load_data(data_file)
        else:
            # Create simple arithmetic examples for circuit analysis
            self.create_arithmetic_examples()

    def load_data(self, filename):
        """Load data from JSON file in the data directory"""
        data_path = Path(__file__).parent / "data" / filename
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Handle ToT data structure with data_entry and steps
            if isinstance(data, dict) and "data_entry" in data and "steps" in data:
                self.load_tot_data(data)
                return
                
            # Handle nested list structure from test.json
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                # Flatten the nested structure
                examples = []
                for group in data:
                    examples.extend(group)
                data = examples
            
            for item in data[:self.n]:  # Limit to n examples
                if isinstance(item, dict) and "Prompt" in item:
                    prompt = item["Prompt"]
                    # Extract target from the prompt (simple heuristic)
                    target = "24"  # Default for Game of 24
                    self._examples.append({"input": prompt, "target": target})
                    self._clean_examples.append(prompt)
                    self._corrupted_examples.append(prompt)
                    self._labels.append(target)
                    
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            self.create_arithmetic_examples()
            
    def load_tot_data(self, data):
        """Load ToT-format data with thought variations"""
        data_entry = data["data_entry"]
        steps = data["steps"]
        
        # Process each step's thought variations
        for step_data in steps:
            step_num = step_data["step"]
            prompt = step_data.get("Prompt", "")
            thought_variations = step_data.get("thought_variation", {})
            
            # Create examples from thought variations
            for variation_text, scores in thought_variations.items():
                if variation_text.strip():  # Skip empty variations
                    # Include the thought variation in the input so circuits can differentiate
                    clean_input = prompt + variation_text
                    
                    # Create corrupted version with different wording/structure
                    corrupted_prompt = prompt.replace("Input:", "Problem:").replace("Possible next steps:", "Available moves:")
                    corrupted_input = corrupted_prompt + variation_text
                    
                    # Use a completion target (like "correct" or the next step)
                    target = "correct"  # Simple binary target for now
                    
                    self._examples.append({
                        "input": clean_input, 
                        "target": target,
                        "step": step_num,
                        "variation": variation_text
                    })
                    self._clean_examples.append(clean_input)
                    self._corrupted_examples.append(corrupted_input)
                    self._labels.append(target)
    
    def create_arithmetic_examples(self):
        """Create simple arithmetic examples for circuit analysis"""
        examples = [
            {"input": "2 + 3 =", "target": "5"},
            {"input": "4 * 6 =", "target": "24"}, 
            {"input": "8 - 3 =", "target": "5"},
            {"input": "12 / 4 =", "target": "3"},
            {"input": "7 + 8 =", "target": "15"}
        ]
        
        for ex in examples[:self.n]:
            self._examples.append(ex)
            self._clean_examples.append(ex["input"])
            self._corrupted_examples.append(ex["input"])
            self._labels.append(ex["target"])

    @property
    def examples(self):
        return self._examples

    def get_questions(self):

        if self._examples:
            return None

        with open(Path(__file__).parent / "data" / self.data_file, encoding="utf-8") as f:
            task = json.load(f)  # list[dict]

        self._examples = []
        for ex in task:
            single_input = ex["Input"].strip().replace("\n", "").replace("\t", "")
            #for thoughts in ex["labels"]:
            #    combined = single_input + thoughts
            #    print(thoughts)
            self._examples.append({"input": single_input, "target": ""})
            #    print(self._examples)
        
        print(self._examples)

        #random.shuffle(self._examples)
        #self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter):
        if formatter.name == "chain-of-thought":
            raise NotImplementedError(
                "Chain-of-thought not supported for arithmetic problems."
            )
        # Manual mode — nothing to format
        if self._examples and self._clean_examples:
            return None

        Qs = [v["input"] for v in self._examples]
        As = [""] * len(self._examples) # just a filler to not break code

        self._clean_examples = [
            formatter.format(self.description, ex["input"], questions=Qs, answers=As)
            for ex in self._examples
        ]
        #print(self._clean_examples)

        self._labels = [ex["target"] for ex in self._examples]  # <-- list of lists
        print(self._labels)
        corrupted_prompt = "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 2 6 11 13\\nPossible next steps:\\n5 % 13 = ?! (left: 6 9 56)"
        #self._corrupted_examples = self._clean_examples[:]
        self._corrupted_examples = [corrupted_prompt] * len(self._clean_examples)

        #random.shuffle(self._corrupted_examples)

        Qs, As = [v["input"] for v in self._examples], [
            v["target"] for v in self._examples
        ]


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
