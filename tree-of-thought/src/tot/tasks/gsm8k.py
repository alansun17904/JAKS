import os
import re
import json

from tot.tasks.base import Task, DATA_PATH
from tot.prompts.gsm8k import (
    standard_prompt,
    cot_prompt,
    propose_prompt, 
    value_prompt,
    value_last_step_prompt
)

_NUM_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?")

def _finish_prompt(input_text, steps_so_far):
    steps = (steps_so_far or "").strip()
    return (
        'Finish the solution briefly and end with exactly: Final answer: <number>\n'
        'Do NOT restate the problem. Keep to 1–3 short lines before the final answer.\n'
        f"Problem:\n{input_text}\n"
        "Steps so far:\n"
        f"{steps}\n"
    )


def _short_answer_from_rationale(answer_text):
    if not answer_text:
        return ""
    txt = str(answer_text).replace("−", "-").replace(",", "").replace("$", "")
    m = re.search(r"####\s*(.+)$", txt)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(txt)
    return nums[-1].strip() if nums else txt.strip()

def _extract_predicted_answer(output):
    if not output:
        return ""
    txt = output.strip().replace("−", "-").replace(",", "").replace("$", "")
    m = re.search(r"####\s*([+\-]?\d+(?:\.\d+)?)", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"final\s*answer\s*:\s*([+\-]?\d+(?:\.\d+)?)", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(txt)
    return nums[-1].strip() if nums else ""


def _num_equal_strict(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return (a or "").strip() == (b or "").strip()

def _make_state_for_propose(problem, steps):
    steps = (steps or "").strip()
    if steps:
        return f"Problem:\n{problem}\n\nSteps so far:\n{steps}\n"
    return f"Problem:\n{problem}\n\nSteps so far:\n"

def _is_final_step(y):
    if not y:
        return False
    yl = y.lower()
    return ("####" in yl) or ("final answer" in yl)


class GSM8KTask(Task):
    def __init__(self, file="gsm8k.json"):
        super().__init__()
        path = os.path.join(DATA_PATH, "gsm8k", file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "train" not in data or not isinstance(data["train"], list):
            raise ValueError(f"Expected {file} to be a dict with a 'train' list at {path}.")

        self.data = data["train"]
        self.value_cache = {}
        self.steps = 6           # maybe increase or decrease later
        self.stops = ["\n"] * 4

    def __len__(self):
        return len(self.data)

    def get_input(self, idx):
        q = self.data[idx]["question"]
        print(f"To Debug: Current entry (GSM8K): {q[:120]}{'...' if len(q) > 120 else ''}")
        return q

    def test_output(self, idx, output):
        gold_full = self.data[idx]["answer"]
        gold_short = _short_answer_from_rationale(gold_full)
        pred_short = _extract_predicted_answer(output)

        if not pred_short or not gold_short:
            return {"r": 0}
        return {"r": int(_num_equal_strict(pred_short, gold_short))}

    @staticmethod
    def standard_prompt_wrap(x, y=""):
        return standard_prompt.format(input=x) + (y or "")

    @staticmethod
    def cot_prompt_wrap(x, y=""):
        return _finish_prompt(x, y)

    @staticmethod
    def propose_prompt_wrap(x, y=""):
        if _is_final_step(y) or (y or "").strip().count("\n") >= 3:
            # no extra "Steps:" here
            return _finish_prompt(x,y)

        state = _make_state_for_propose(x, y or "")
        return propose_prompt.format(input=state)
    @staticmethod
    def value_prompt_wrap(x, y):
        """
        Produce a rating prompt for partial vs. final steps.
        """
        if _is_final_step(y):
            ans = _extract_predicted_answer(y)
            return value_last_step_prompt.format(input=x, answer=ans)
        state = _make_state_for_propose(x, y or "")
        return value_prompt.format(input=state)

    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        """
        Convert textual ratings to a numeric value, mirroring Game24Task's ad-hoc map.
        """
        y_text = (y or "").strip()
        if len(y_text.split("\n")) <= 1 and "answer" not in y_text.lower() and "####" not in y_text:
            return 0.0
        last_lines = [str(v).strip().split("\n")[-1].lower() for v in value_outputs]
        score_map = {"impossible": 0.001, "likely": 1.0, "sure": 20.0}
        return sum(score_map.get(name.strip(), 0.0) for name in last_lines)