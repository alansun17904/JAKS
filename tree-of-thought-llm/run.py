"""
Parses CLI arguments, sets up logging, and runs the main generation loop via run(args).

devs note: If it is your first time please pick an example of game24 and loop through the whole pipeline.
"""

import argparse
from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve

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

    # from the specified start_index to end_index
    for i in range(args.task_start_index, args.task_end_index):
        if args.naive_run:
            # just standard prompt and answer type generation
            ys, info = naive_solve(args, task, i) 
        else:
            # the propose method uses this
            ys, info = solve(args, task, i)


        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos })
        print(info)

def parse_args():
    """
    Each and every arg is worded out in JAKS/src/tree-of-thought-llm/README.md

    Returns:
        argparse.Namespace : all the CLI args passed 
    """
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt2', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'meta-llama/llama-3.2-3B-Instruct'], default='gpt2')
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