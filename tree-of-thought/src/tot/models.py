"""
Provides wrappers for OpenAI's ChatCompletion API with backoff, token/cost tracking, and prompt handling utilities.
"""

import os
from functools import lru_cache
from .t_lens_generate import get_tlens_model
DEFAULT_MAX_TOKENS = 64
DEFAULT_STOPS = ["\n\n", "Final answer:", "FINAL_ANSWER:"]

@lru_cache(maxsize=8)
def _get_tlens(model_id: str):
    return get_tlens_model(model_id=model_id)

def gpt(prompt,
        model="gpt2",
        temperature=0.7,
        max_tokens=DEFAULT_MAX_TOKENS,
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
    stops = stop if stop is not None else DEFAULT_STOPS
    tlens = _get_tlens(model)
    outputs = []
    
    for _i in range(max(1, n)):
        raw = tlens.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        
        if proposals:
            lines = []
            text = raw.strip()

            # --- TASK-SPECIFIC PARSING (compat with your old code) ---
            if task == "Game24Task":
                # Keep lines after the marker
                part = text.split("Possible next steps:", 1)[-1]
                lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
                # (Optionally drop the very last line if you fear truncation)
                # if lines: lines = lines[:-1]

            elif task == "GSM8KTask":
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            else:
                # default: split on lines
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            outputs.append(lines)

            # Optional raw logging
            if isinstance(json, dict) and isinstance(json.get("raw_output_prop"), list):
                json["raw_output_prop"].append(raw)

        else:
            outputs.append(raw)
            if isinstance(json, dict) and isinstance(json.get("raw_output_eval"), list):
                json["raw_output_eval"].append(raw)

    return outputs
