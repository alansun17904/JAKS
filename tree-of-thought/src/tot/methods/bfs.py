"""
Implements value-based, vote-based, and proposal-based search.
"""

import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import json
from pathlib import Path
import subprocess
import re

# Strip chat special tokens like <|eot_id|>
_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")

# One-line schema for propose steps
_SC_ASSIGN = re.compile(r'^ASSIGN:\s*[A-Za-z_]\w*\s*=\s*[-+*/().\dA-Za-z_\s]+$')
_SC_EQ     = re.compile(r'^EQ:\s*[-+*/().\dA-Za-z_\s]+\s*=\s*[-+*/().\dA-Za-z_\s]+$')

# Extract the LHS variable name in ASSIGN lines
_LHS_RE = re.compile(r'^\s*ASSIGN:\s*([a-zA-Z_]\w*)\s*=', re.IGNORECASE)

json_thought = {}
all_entries = []
_BAD_VARS = {"tmp", "temp", "step", "var", "test"}
_EQ_IDENTITY_NUM = re.compile(r'^EQ:\s*([+\-]?\d+(?:\.\d+)?)\s*=\s*\1\s*$', re.IGNORECASE)


def _is_identity_eq(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if _EQ_IDENTITY_NUM.match(s):
        return True
    if not s.lower().startswith("eq:"):
        return False
    try:
        body = s[3:].strip()
        if "=" not in body:
            return False
        lhs, rhs = body.split("=", 1)
        lhs = lhs.strip().replace(" ", "")
        rhs = rhs.strip().replace(" ", "")
        return lhs != "" and lhs == rhs
    except Exception:
        return False

def _mentions_bad_var(s: str) -> bool:
    words = re.findall(r'[A-Za-z_]\w*', s)
    return any(w.lower() in _BAD_VARS for w in words)

def _fallback_one_liner(lines, prompt_text, max_len=120):
    """
    Last-resort, task-agnostic fallback: pull a short, clean next-step line
    from whatever the model returned, so we ALWAYS have at least one proposal.
    """
    for raw in (lines or []):
        s = (raw or "").strip()
        if not s:
            continue
        # strip any special chat tokens
        s = _TOKEN_RE.sub("", s).strip()
        # drop prompt echo
        if s in prompt_text or s.lower().startswith(("problem:", "steps so far:")):
            continue
        # drop 'next step:' prefix if present
        if s.lower().startswith("next step:"):
            s = s.split(":", 1)[1].strip()
        # keep only first physical line
        s = s.splitlines()[0].strip()
        if not s:
            continue
        # keep it short
        if len(s) > max_len:
            s = s[:max_len].rstrip()
        # trim to ~16 words to avoid rambles
        words = s.split()
        if len(words) > 16:
            s = " ".join(words[:16])
        # reject extremely generic or prompty fragments
        bad_starts = ("given the problem", "do not", "output exactly", "next step")
        if s.lower().startswith(bad_starts):
            continue
        return s
    # Absolute last resort: schema-conformant harmless step (avoid using a "step" var)
    return "EQ: 1=1"


def _keep_one_schema_line(s: str, fallback: str, max_len: int = 60) -> str:
    """
    Enforce we only keep a single, short line matching the ASSIGN/EQ schema.
    Also reject ASSIGNments to a variable literally named 'step'.
    """
    if not s:
        return fallback
    # strip special tokens and clamp to first line
    line = _TOKEN_RE.sub("", s).strip().splitlines()[0].strip()
    # drop trivial prompt echoes
    if not line or line.lower().startswith(("problem:", "steps so far:", "next step:", "given the problem", "do not")):
        return fallback
    # hard length cap
    if len(line) > max_len:
        line = line[:max_len].rstrip()
    # schema check
    if _SC_ASSIGN.match(line):
        m = _LHS_RE.match(line)
        if m and m.group(1).lower() == "step":
            return fallback
        return line
    if _SC_EQ.match(line):
        return line
    return fallback


def _sanitize_proposals(lines, prompt_text, prior_text=None, max_len=120):
    """
    Clean & dedupe:
    - keep only first physical line
    - enforce ASSIGN:/EQ: schema
    - drop prompt echoes, identities, repeats, and junk vars (tmp/step/var/test)
    - NO FALLBACK LINE (returns [] if nothing good)
    """
    prompt_lines = set(l.strip() for l in prompt_text.splitlines() if l.strip())
    prior = (prior_text or "").lower()
    cleaned, seen = [], set()

    for ln in (lines or []):
        s = (ln or "").strip()
        if not s:
            continue

        # strip chat tokens like <|eot_id|>
        s = _TOKEN_RE.sub("", s).strip()

        # exact prompt echo skip
        if s in prompt_lines:
            continue

        # clamp to a single schema line
        one = _keep_one_schema_line(s, "", max_len=min(60, max_len))
        if not one:
            continue

        # drop identity equations and junk vars
        if _is_identity_eq(one):
            continue
        if _mentions_bad_var(one):
            continue

        # drop ASSIGN: step=...
        m = _LHS_RE.match(one)
        if m and m.group(1).lower() == "step":
            continue

        sl = one.lower()
        if sl in seen:
            continue
        if sl in prior:
            continue

        seen.add(sl)
        cleaned.append(one)

    # IMPORTANT: return [] if nothing good, no synthetic fallback
    return cleaned


def _looks_final(s: str) -> bool:
    if not s:
        return False
    t = s.lower()
    return ("####" in t) or ("final answer" in t) or ("final_answer:" in t)


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    """
    Ask the LLM to rate how promising the current step is
    on the scale (impossible/likely/sure).
    """
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]

    value_outputs = gpt(
        value_prompt,
        n=n_evaluate_sample,
        stop=None,
        json=thought_dict,
        x=x,
        proposals=False
    )
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    """
    Call get_value for each candidate, with a small local cache.
    """
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = local_value_cache[y]
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_proposals(task, x, y, n_generate_sample, thought_dict):
    propose_prompt = task.propose_prompt_wrap(x, y)
    thought_dict["Prompt"] = propose_prompt
    print(thought_dict)

    raw_lists = gpt(
        propose_prompt,
        n=n_generate_sample,
        stop=None,
        json=thought_dict,
        x=x,
        proposals=True,
        task=type(task).__name__
    )

    flat = list(itertools.chain(*raw_lists))
    proposals = _sanitize_proposals(flat, propose_prompt, prior_text=y)
    print(f"To debug: thought variations: {proposals}")

    # If everything was filtered out, DO NOT append junk — keep current y
    if not proposals:
        print("To debug: no clean proposals; carrying forward current partial solution unchanged.")
        return [y]

    for step_line in proposals:
        thought_dict["thought_variation"][step_line] = []

    print(f"To debug: {thought_dict}")
    return [y + step_line + '\n' for step_line in proposals]


def get_cot_completions(task, x, y, n_generate_sample, thought_dict):
    """
    Use CoT prompt to produce full solutions (often final).
    """
    cot = task.cot_prompt_wrap(x, y or "")
    thought_dict["Prompt"] = cot
    print(thought_dict)

    outs = gpt(
        cot,
        n=n_generate_sample,
        stop=None,
        json=thought_dict,
        x=x,
        proposals=False,
        task=type(task).__name__,
    )
    return [out if out.endswith("\n") else out + "\n" for out in outs]


# ---- JSON logging helpers --------------------------------------------------

def thought_to_json(dictionary, filename):
    sanitized_filename = re.sub(r'[<>:"/\\|?*,]', '_', filename)
    if len(sanitized_filename) > 100:
        sanitized_filename = sanitized_filename[:100]
    p = Path("circuit-stability/code/src/cdatasets/data") / sanitized_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(dictionary, f, indent=4)
        f.write("\n")


def get_circuit_scores(task, x, y):
    return None


# ---- Main loop -------------------------------------------------------------

def solve(args, task, idx, to_print=True):
    """
    Main BFS search loop for generating and selecting candidate solutions step by step.
    At each step, generates, evaluates, and selects candidates according to the specified methods.
    """
    global gpt
    global json_thought
    global thought_dict

    try:
        import random, torch
        random.seed(0); np.random.seed(0)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    except Exception:
        pass

    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    ys = ['']
    infos = []

    json_thought = {"data_entry": str(x), "steps": []}

    for step in range(task.steps):
        thought_dict = {
            "step": step,
            "Prompt": None,
            "raw_output_prop": [],
            "raw_output_eval": [],
            "thought_variation": {}
        }

        # generation
        if args.method_generate == 'propose':
            new_ys_nested = [get_proposals(task, x, y, args.n_generate_sample, thought_dict) for y in ys]
        elif args.method_generate == 'cot':
            new_ys_nested = [get_cot_completions(task, x, y, args.n_generate_sample, thought_dict) for y in ys]
        else:
            new_ys_nested = [ys]

        print(f"To debug: {thought_dict}")

        new_ys = list(itertools.chain(*new_ys_nested))
        ids = list(range(len(new_ys)))
        print(f"To debug: ids: {ids}")

        # log to JSON as we go
        json_thought["steps"].append(thought_dict)
        name = json_thought["data_entry"].replace(" ", ",")
        thought_to_json(json_thought, f"{name}.json")

        # selection scores
        if args.method_select == 'first':
            values = [0.0] * len(new_ys)  # no judging, always take first
        else:
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # attach scores to thought variations
        for elist, eval in zip(thought_dict["thought_variation"].values(), values):
            elist.append(eval)

        # choose next beams
        if args.method_select == 'first':
            select_ids = [0]
        elif args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda z: values[z], reverse=True)[:args.n_select_sample]
        else:
            select_ids = [0]

        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        print(f"To Debug: New selected{select_new_ys}")

        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

        if any(_looks_final(y) for y in ys):
            break

    if to_print:
        print(ys)
    return ys, {'steps': infos}
