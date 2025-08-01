
"""
This script contains the LLM class, which loads and uses
a Hugging Face Causal Language Model for text generation.
"""

from dotenv import load_dotenv
from transformer_lens import HookedTransformer

import os


load_dotenv(dotenv_path="../.env")

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

        # Select device
        #if torch.backends.mps.is_available():
        #    self.device = "mps"
        #    print("To debug: Using MPS")
        #elif torch.cuda.is_available():
        #    self.device = "cuda"
        #else:
        #    self.device = "cpu"


        self.model = HookedTransformer.from_pretrained(model_id)


    def generate(self,
                                 prompt,
                                 max_tokens : int = 500,
                                 temperature : float = 0.6,
                                 ):

        output = self.model.generate(
            input=prompt,
            max_new_tokens=max_tokens,
            do_sample=True,  # enable sampling
            top_k=50,  # restrict to top 50 tokens
            top_p=0.95,  # restrict to tokens covering 95% of prob mass
            temperature=temperature,  # control randomness
            return_type="str"  # return a string
        )
        return output



