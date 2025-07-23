"""
Parses CLI arguments, sets up logging, and runs the main generation loop via run(args).

devs note: If it is your first time please pick an example of game24 and loop through the whole pipeline.
"""

import os
import json
import argparse
from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import gpt_usage

def run(args):
    """
    Runs the main inference loop from the desired start_index 
    to the stop_index of a gven task (all args are passed through CLI).

    Args:
        args (argparse.Namespace): Parsed command-line arguments controlling the pipeline.

    Returns:
        None
    """
    # fetches respective task attributes from /tot/tasks/{task_name}.py
    task = get_task(args.task) # task is a instance of the respective task class
    logs, cnt_avg, cnt_any = [], 0, 0

    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # from the specified start_index to end_index
    for i in range(args.task_start_index, args.task_end_index):
        if args.naive_run:
            # just standard prompt and answer type generation
            ys, info = naive_solve(args, task, i) 
        else:
            # the propose method uses this
            ys, info = solve(args, task, i)


        # TODO: IDK IF WE NEED LOGS SO I DIDN'T GET INTO IT
        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    """
    Each and every arg is worded out in JAKS/src/tree-of-thought-llm/README.md

    Returns:
        argparse.Namespace : all the CLI args passed 
    """
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-2', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o'], default='gpt-2')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)