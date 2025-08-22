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


json_thought = {}
all_entries = []



def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    """
    To ask the llm how well the current step is on the scale of (sure/maybe/impossible)
    
    Args:
        task: The task object providing prompt and cache utilities.
        x: The input string.
        y: The candidate output string.
        n_evaluate_sample: number of times to ask the llm
        cache_value: Whether to cache the value result.
    Returns:
        float: The value score for the candidate output.
    """

    # goes to /tot/tasks/{task}.py to get the respective tasks's wrap methods
    value_prompt = task.value_prompt_wrap(x, y) # eg: Evaluate if given numbers can reach 24 (sure/likely/impossible)
    # if cached value is available then use cached
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    # Send to gpt to run the inference loop
    value_outputs = gpt(value_prompt,
                        n=n_evaluate_sample,
                        stop=None,
                        json=thought_dict,
                        x=x,
                        proposals = False)
    # goes to /tot/tasks/{task}.py to get the respective tasks's unwrap methods
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    """
    Actual function that calls the inference loop i.e get_values() for each variation 

    Args:
        task: The task object.
        x: The input string.
        ys: List of step ith variations.
        n_evaluate_sample: number of times to ask the llm
        cache_value: Whether to cache value results.
    Returns:
        list: Value scores for each candidate output.
    """
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_proposals(task, x, y): 
    """
    Ask LLM to propose a set of first steps based on few-shot prompting.

    Args:
        task: The task object.
        x: The input string.
        y: The current output string (partial solution).
    Returns:
        list: New candidate outputs (proposals) as continuations of y.
    """
    # In /tot/tasks/{task}.py gets the respective method 
    propose_prompt = task.propose_prompt_wrap(x, y)

    thought_dict["Prompt"] = propose_prompt

    print(thought_dict)

    # each line is a variation
    proposals = gpt(propose_prompt,
                    n=2, stop=None,
                    json = thought_dict,
                    x = x,
                    proposals = True,
                    task = type(task).__name__ )

    print(f"To debug: thought variations: {proposals}")


    proposals = list(itertools.chain(*proposals))  # this flattens it into a single list

    for proposal in proposals:
        thought_dict["thought_variation"][proposal] = []

    print(f"To debug: {thought_dict}")

    # store each variation in list along with previous variation
    return [y + _ + '\n' for _ in proposals]

### adding a function for writing to json over here and calling it at the end of solve
def thought_to_json(dictionary, filename):
    p = Path("circuit-stability/code/src/cdatasets/data") / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(dictionary, f, indent=4)
        f.write("\n")

def get_circuit_scores(task, x, y):
    """
    This function is used to get the best scored thought based on its corresponding circuit metrics.
    :param task:
    :param x:
    :param y:
    :return:
    """

    return None

def solve(args, task, idx, to_print=True):
    """
    Main BFS search loop for generating and selecting candidate solutions step by step.
    At each step, generates, evaluates, and selects candidates according to the specified methods.

    Args:
        args: all the CLI args.
        task: The task object.
        idx: Index of the input to solve.
        to_print: Whether to print intermediate results.
    Returns:
        tuple: (final candidate outputs, log info dictionary)
    """
    # the main model generation loop; goes to /tot/models.py
    global gpt
    # make the json_thought object global
    global json_thought
    global thought_dict
    # just adds these args without actually calling the function
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # gets input at current index
    ys = ['']  # current output candidates / thought variations
    infos = []

    """
    SCHEMA FOR JSON
[
  {
    "data_entry": "1 1 11 11",
    "steps": [
      {
        "step": 0,
        "prompt": 0,
        "raw_output": "11 - 1 = 10 (left: 1 11 10)",
        "thought_variation": {
          "var1": { "heuristic": 0.82, "circuit_stability": 13.68 },
          "var2": { "heuristic": 0.55, "circuit_stability": -7.90 }
        }
      },
      {
        "step": 1,
        "prompt": 1,
        "raw_output": "10 + 1 = 11 (left: 11 11)",
        "thought_variation": {
          "var1": { "heuristic": 0.73, "circuit_stability": 4.90 }
        }
      }
    ]
  },
  {
    "data_entry": "2 8 8 14",
    "steps": [
      {
        "step": 0,
        "prompt": 0,
        "raw_output": "2 + 8 = 10 (left: 8 10 14)",
        "thought_variation": {
          "var1": { "heuristic": 0.90, "circuit_stability": 4.12 }
        }
      }
    ]
  }
]

    """
    # Structure of JSON object
    json_thought["data_entry"] = str(x)
    json_thought["steps"] = []

    for step in range(task.steps): # each class instance has an attribute steps allowed to complete given task
        thought_dict = {}


        thought_dict = {"step": step,
                        "Prompt": None,
                        "raw_output_prop": [],
                        "raw_output_eval": [],
                        "thought_variation": {}
                        }

        # Ask model to propose variations of first step /tot/prompts/{task}.py propose_prompt
        if args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]

        #print(f"To debug: new_ys: {new_ys}")

        # the list of list for concurrent step's variation generation is a list of list


        print(f"To debug: {thought_dict}")

        new_ys = list(itertools.chain(*new_ys)) # this flattens it into a single list
        #print(f"To debug: flattened new_ys: {new_ys}")

        ids = list(range(len(new_ys)))
        print(f"To debug: ids: {ids}")

        # TODO: Add additional cli args for circuit discovery as well
        # TODO: Make this into a function and call the circuit_discovery script with params
        # TODO: If above is done we have to change path too.
        # Edit params at circuit-stability/code/src/scripts/naive_run.sh

        
        #TODO: Circuit selection goes inside the first if

        # evaluation method
        if args.method_evaluate == 'circuits':
            subprocess.run(["bash",
                            "circuit-stability/code/src/scripts/naive_run.sh"],
                           check=True)
            values = get_circuit_scores(task, x, new_ys)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        #Append score for each variation
        for elist, eval in zip(thought_dict["thought_variation"].values(), values):
            elist.append(eval)

        # selection
        if args.method_select == 'sample':
            print(f"To debug:Values {values}")
            rng = np.random.default_rng(42)  # optional seed
            n, lo, hi = len(values), 0.001, 20
            values = list(rng.uniform(lo, hi, size=n))
            print(f"To debug:Values {values}")
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()

        elif args.method_select == 'greedy':
            # sorted based on values
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        print(f"To Debug: New selected{select_new_ys}")

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

        # Append each thought's data to the json
        json_thought["steps"].append(thought_dict)

    name = json_thought["data_entry"].replace(" ", ",")
    thought_to_json(json_thought, f'{name}.json')
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

