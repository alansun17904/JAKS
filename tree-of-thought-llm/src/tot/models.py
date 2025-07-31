"""
Provides wrappers for OpenAI's ChatCompletion API with backoff, token/cost tracking, and prompt handling utilities.
"""

import os
import openai
import backoff

from .t_lens_generate import get_tlens_model

completion_tokens = prompt_tokens = 0

# get openAPI key for out of the box TOT
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
# Optionally set custom OpenAI API base
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    # just tries to hit API again if error occurs using exponential backoff 
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-2", temperature=0.7, max_tokens=500, n=1, stop=None) -> list:
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
    # Check if model is gpt-2
    if model.lower() not in ['gpt-3.5-turbo', 'gpt-4o']:
        outputs = []
        # Lazy import so that it only initiates once
        t_lens = get_tlens_model(model_id = model)
        # generate variations using hf_models
        while n > 0: # n is the number of generations we want, each generation has x variations
            raw_output_text = t_lens.generate(prompt,  # TODO maybe make the prompt look like messages below
                                          temperature=temperature,
                                          max_tokens=max_tokens,
                                          )
            n -= 1
            outputs.append(raw_output_text)

        print("To Debug: Encoded tensors:\n" + "\n".join(str(_) for _ in outputs))
        return outputs
    
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """
    Generate completions from a list of chat messages using OpenAI's chat models.
    Handles batching and tracks token usage.
    Args:
        messages (list): List of message dicts from above for the chat API.
        model (str): Model name (default: 'gpt-4').
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate.
        n (int): Number of completions to generate.
        stop (str or list): Optional stop sequence(s).
    Returns:
        list: List of generated completions (strings).
    """
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        """
        Since API call limits requests to 20 we batch it and send; eg :50 generations 
        will run thrice each time subtracting 20 requests.
        """
        cnt = min(n, 20)
        n -= cnt
        # output a JSON object that for eg: "choices": [{"message": {"content": "output 1"}}, ...],
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        # appends the 'content' string from the 'JSON output' to the list
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
