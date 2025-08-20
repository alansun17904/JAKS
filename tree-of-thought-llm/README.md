# Tree of Thoughts (ToT) for thought generation

# Technical Constraints for thought generation:
1. GPT-2 has a max input token length of **_1024_**; does limit generation for the task 'text'.
2.


## Setup
1. Get hugginface access token here: https://huggingface.co/settings/tokens (for gated models) (https://transformerlensorg.github.io/TransformerLens/content/getting_started.html)

2. Follow below command to get `tot` package:

```bash
git clone https://github.com/princeton-nlp/tree-of-thought-llm
cd tree-of-thought-llm
# for mac use 
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# for others
pip install -r requirements.txt 

pip install -e .  # install `tot` package
```

**After setting up the tot package**
1. Create an `.env` file inside the `JAKS` folder 
2. Create an `HF_TOKEN` variable and add your huggingface access token

## File Structure

```bash
## File Structure

```{bash}
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt            
├── run.py                      # contains the main entry point 
├── scripts                     # Has bash scripts to run each task 
│   ├── crosswords  
│   │   ├── cot_sampling.sh
│   │   ├── search_crosswords-dfs.ipynb    # Has depth first search
│   │   └── standard_sampling.sh
│   ├── game24
│   │   ├── bfs.sh
│   │   ├── cot_sampling.sh
│   │   └── standard_sampling.sh
│   └── text
│       ├── bfs.sh
│       ├── cot_sampling.sh
│       └── standard_sampling.sh
├── setup.py
└── src
    └── tot
        ├── __init__.py
        ├── data                 # Dataset in .csv .json .txt format for each task
        │   ├── 24
        │   │   └── 24.csv
        │   ├── crosswords
        │   │   ├── mini0505_0_100_5.json
        │   │   └── mini0505.json
        │   └── text
        │       └── data_100_random_text.txt
        ├── methods              # Implements the BFS algorithm
        │   └── bfs.py
        ├── models.py            # Has the function that calls the actual completion
        ├── hf_llm.py            # Has hugging face class (just in case)
        ├── t_lens_generate.py   # has inference generation through transformer lens
        ├── prompts              # Has prompts as 'str' for each tasks
        │   ├── crosswords.py
        │   ├── game24.py
        │   └── text.py
        └── tasks                # Has the class definition of each task (task dependent methods)
            ├── __init__.py
            ├── base.py
            ├── crosswords.py
            ├── game24.py
            └── text.py

```

## Quick Start To Run Locally For One Inference

run `JAKS/tree-of-thought-llm/scripts/game24/bfs.sh`

**_OPTIONAL:_** The above script just runs for the first index entry in the dataset of game24. Could be changed according to your will. 

## flow of code 
1. You run the respective script from the package.
2. Calls the run.py 
    2.1 Run.py has two main components:
    2.2 Calls get_task() -> respective instance of a task class which has all required processing functions
    2.3 Loops from `start_idx` to `stop_idx` and calls `naive_solve` or `solve`; based on input args
3. If `solve` (TOT + BFS) function is called in a loop w.r.t each tasks number of steps (eg: game24 is 4):
    3.1 If `--method_generate` is propose
         3.1.1 Calls `get_proposals` -> wraps the input with `propose_prompt` and calls gpt for inference
         3.1.2 Gpt calls hf_generate method from LLM class in a loop of `--n_generate_sample` to generate output texts; 
4. Depending on the `--method_evaluate` we call `get_votes` or `get_values` respectively:
    4.1 This has respective prompt wrap to vote each thought variations respectively
5. We then select thought variations based on `sample`(random sampling) or `greedy`
   5.1 Based on this we return the selected new `ys` (I believe it's the plural form of y )


## Paper Experiments

Run experiments via ``sh scripts/{game24, text, crosswords}/{standard_sampling, cot_sampling, bfs}.sh``, except in crosswords we use a DFS algorithm for ToT, which can be run via ``scripts/crosswords/search_crosswords-dfs.ipynb``.

The very simple ``run.py`` implements the ToT + BFS algorithm, as well as the naive IO/CoT sampling. Some key arguments:

- ``--naive_run``: if True, run naive IO/CoT sampling instead of ToT + BFS.
-  ``--prompt_sample`` (choices=[``standard``, ``cot``]): sampling prompt
- ``--method_generate`` (choices=[``sample``, ``propose``]): thought generator, whether to sample independent thoughts (used in Creative Writing) or propose sequential thoughts (used in Game of 24)
- ``--method_evaluate`` (choices=[``value``, ``vote``]): state evaluator, whether to use the value states independently (used in Game of 24) or vote on states together (used in Creative Writing)
- ``--n_generate_sample``: number of times to prompt for thought generation
- ``--n_evaluate_sample``: number of times to prompt for state evaluation
- ``--n_select_sample``: number of states to keep from each step (i.e. ``b`` in the paper's ToT + BFS algorithm)


## How to Add A New Task
Setting up a new task is easy, and mainly involves two steps.
* Set up a new task class in ``tot/tasks/`` and task files in ``tot/data/``. See ``tot/tasks/game24.py`` for an example. Add the task to ``tot/tasks/__init__.py``.
* Set up task-specific prompts in ``tot/prompts/``. See ``tot/prompts/game24.py`` for an example. Depending on the nature of the task, choose ``--method_generate`` (choices=[``sample``, ``propose``]) and ``--method_evaluate`` (choices=[``value``, ``vote``]) and their corresponding prompts. 

## Citations
Please cite the paper and star this repo if you use ToT and find it interesting/useful, thanks! Feel free to contact shunyuyao.cs@gmail.com or open an issue if you have any questions.

```bibtex
@misc{yao2023tree,
      title={{Tree of Thoughts}: Deliberate Problem Solving with Large Language Models}, 
      author={Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},
      year={2023},
      eprint={2305.10601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
