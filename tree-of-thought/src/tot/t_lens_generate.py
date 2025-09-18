
"""
This script contains the LLM class, which loads and uses
a Hugging Face Causal Language Model for text generation.
"""


from pathlib import Path

from transformer_lens import HookedTransformer

import os
import torch


hf_token = os.getenv("HF_TOKEN")

hf_model = None

def get_tlens_model(model_id):
    """
    Lazy initiation of class LLM; making sure its initiated only once.
    """
    global hf_model
    if hf_model is None:
        hf_model = LLM(model_id=model_id)
    return hf_model

class LLM():
    def __init__(self, model_id = 'gpt2', args = None,):
        """
        A wrapper class to load a Causal Language Model and generate text from prompts.

        Attributes:
            device (str): The computation device ("cuda", "mps", or "cpu").
            model (AutoModelForCausalLM): The language model instance.
            tokenizer (AutoTokenizer): The tokenizer instance.
            args (list): Optional list of arguments for future use.
        """


        self.args = args

        #Select device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("To debug: Using MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"



        self.model = HookedTransformer.from_pretrained(
            model_id,
            device = self.device,
        )


    def generate(self,prompt,max_tokens : int = 250,temperature : float = 0.6,):
        toks = self.model.to_tokens(prompt, prepend_bos=True)
        # TODO: MAke sure variations are different

        gen = self.model.generate(
        toks,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=temperature,
        )
        new = gen[0, toks.shape[1]:]  # keep only newly generated tokens
        return self.model.to_string(new)



