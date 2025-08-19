"""
Implements value-based, vote-based, and proposal-based search.
"""

import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import json
from pathlib import Path
from datetime import datetime

json_thought = {}


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
                        json=json_thought,
                        x=x,)
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

def get_votes(task, x, ys, n_evaluate_sample):
    """
    Use LLM to decide which is the most promising step given instructions and list of steps

    Args:
        task: The task object.
        x: The input string.
        ys: List of step ith variations.
        n_evaluate_sample: number of times to ask the llm
    Returns:
        list: Vote-based value scores for each candidate output.
    """
    # vote_prompt_wrap is only present in task: /tot/tasks/text.py
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
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

    json_thought[str(x)]["Prompt"] = propose_prompt

    print(json_thought)

    # each line is a variation
    proposals = gpt(propose_prompt,
                    n=2, stop=None,
                    json = json_thought,
                    x = x,
                    proposals = True)

    print(f"To debug: thought variations: {proposals}")


    proposals = list(itertools.chain(*proposals))  # this flattens it into a single list

    for proposal in proposals:
        json_thought[str(x)]["thought_variation"][proposal] = []

    #print(f"To debug: {json_thought}")


    # store each variation in list along with previous variation
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    """
    Ask llm to directly answer the question using a standard prompt

    Args:
        task: The task object.
        x: The input string.
        y: The current output string (partial solution).
        n_generate_sample: Number of samples to generate.
        prompt_sample: Type of prompt ('standard' or 'cot').
        stop: Stop token(s) for LLM generation.
    Returns:
        list: New candidate outputs as continuations of y.
    """
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]



### adding a function for writing to json over here and calling it at the end of solve
def thought_to_json(dictionary, filename):
    p = Path("circuit-stability/code/src/cdatasets/data") / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(dictionary, f, indent=4)
        f.write("\n")

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
    # just adds these args without actually calling the function
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # gets input at current index
    ys = ['']  # current output candidates / thought variations
    infos = []

    """
    json_thought = { 'data entry': { 'step' : 'int',
                                     'prompt' : 'int',
                                     `raw_output` : `str`,
                                     'thought_variation' : {variation : [score1 = heuristic, score2 = circuit stability]} }
    """

    # Structure of JSON object
    json_thought[str(x)] = { "step" : "",
                             "Prompt" : "",
                             "raw_output_prop": [],
                             "raw_output_eval": [],
                             "thought_variation" : {"" : []}}

    for step in range(task.steps): # each class instance has an attribute steps allowed to complete given task
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        # Ask model to propose variations of first step /tot/prompts/{task}.py propose_prompt
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]


        print(f"To debug: new_ys: {new_ys}")

        # the list of list for concurrent step's variation generation is a list of list
        json_thought[str(x)]["step"] = str(step)

        print(f"To debug: {json.dumps(json_thought, indent=4)}")
        
        new_ys = list(itertools.chain(*new_ys)) # this flattens it into a single list
        print(f"To debug: flattened new_ys: {new_ys}")

        ids = list(range(len(new_ys)))

        # evaluation method
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        #Append score for each variation TODO remove comment line to add each eval to each thought variation
        """
        for elist, eval in zip(json_thought[str(x)]["thought_variation"].values(), values):
            elist.append(eval)
        """
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()

        elif args.method_select == 'greedy':
            # sorted based on values
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    ### calling the function from earlier
    ### using timestamps just so that we can use the code over and over and not have to change anything
    name = f"{datetime.now():%Y%m%d-%H%M%S}"
    thought_to_json(json_thought, f"json_thought_{name}.json")


    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    """
    Simpler baseline: generate samples in one shot without iterative search or evaluation.

    Args:
        args: Namespace of arguments controlling the search.
        task: The task object.
        idx: Index of the input to solve.
        to_print: Whether to print intermediate results.
    Returns:
        tuple: (final candidate outputs, empty log dictionary)
    """
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}