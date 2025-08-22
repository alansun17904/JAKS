"""
Provides wrappers for OpenAI's ChatCompletion API with backoff, token/cost tracking, and prompt handling utilities.
"""

import os

from .t_lens_generate import get_tlens_model

def gpt(prompt,
        model="gpt2",
        temperature=0.7,
        max_tokens=50,
        n=1,
        stop=None,
        json = None,
        x = None,
        proposals = False,
        task = None) -> list:
    """
    Generate completions from a prompt using OpenAI's chat models.
    Args:
        prompt (str): The user prompt to send to the model.
        model (str): Model name (default: 'gpt-4').
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate.
        n (int): Number of completions to generate.
        stop (str or list): Optional stop sequence(s).
    Returns:
        list: List of generated completions (strings).
    """
    append_raw_output = False
    # Check if model is gpt-2
    if model.lower() not in ['gpt-3.5-turbo', 'gpt-4o']:
        outputs = []
        print(model)
        # Lazy import so that it only initiates once
        t_lens = get_tlens_model(model_id = model)
        # generate variations using hf_models
        while n > 0: # n is the number of generations we want, each generation has x variations
            raw_output_text = t_lens.generate(prompt,  # TODO maybe make the prompt look like messages below
                                          temperature=temperature,
                                          max_tokens=max_tokens,
                                          )
            n -= 1
            #print(f"To debug:raw output {raw_output_text}")

            if proposals:
                # Put raw output into json
                if append_raw_output:
                    json["raw_output_prop"].append(raw_output_text)
                # If we look at the prompt anything after 'Possible next steps:' are variations
                if task == "Game24Task":
                    variation = raw_output_text.strip().split("Possible next steps:")[-1]
                    # append everything except the last line, bc the stop rn is no.of token; so not guaranteed that last variation is complete
                    outputs.append(variation.split("\n")[:-1])

            if not proposals:
                if append_raw_output:
                    json[x]["raw_output_eval"].append(raw_output_text)
                #print(raw_output_text)
                outputs.append(raw_output_text)
            #print(f"To debug: {outputs}")
        return outputs
