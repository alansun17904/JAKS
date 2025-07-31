
"""
This script contains the LLM class, which loads and uses
a Hugging Face Causal Language Model for text generation.
"""

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import torch
import os


load_dotenv(dotenv_path="../.env")

hf_token = os.getenv("HF_TOKEN")

hf_model = None

def get_hf_model(model_id):
    """
    Lazy initiation of class LLM; making sure its initiated only once.
    """
    global hf_model
    if hf_model is None:
        hf_model = hf_LLM(model_id=model_id)
    return hf_model

class hf_LLM():
    def __init__(self, model_id = 'openai-community/gpt2', args = None,):
        """
        A wrapper class to load a Causal Language Model and tokenizer,
        set the appropriate device, and generate text from prompts.

        Attributes:
            device (str): The computation device ("cuda", "mps", or "cpu").
            model (AutoModelForCausalLM): The language model instance.
            tokenizer (AutoTokenizer): The tokenizer instance.
            args (list): Optional list of arguments for future use.
        """
        if model_id.lower() in ['gpt-2', 'gpt2']:
            model_id = "openai-community/gpt2"

        #Get config file first just to test for network errors or anything else
        try:
            config = AutoConfig.from_pretrained(model_id)
        except OSError as e:
            raise RuntimeError(
                f"Could not download config for {model_id}: {e}"
                )

        # Select device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("To debug: Using MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # MPS prefers float32
            trust_remote_code=True
        ).to(self.device)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Store any additional arguments
        self.args = args


    def encode(self, prompt: str):
        """
        Tokenizes and encodes the prompt into tensors; moved to the model device.

        Args:
            prompt (str): The input text prompt.

        Returns:
            torch.Tensor: Encoded inputs ready for generation.
        """
        return self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

    def generate(self,
                 prompt: str,
                 max_tokens : int = 500,
                 temperature : float = 0.6,
                  ):
        """
        Generates text from a prompt using the loaded model.

        Args:
            prompt (str): The input text prompt.

        Returns:
            str: The generated text continuation.
        """
        print(f"To Debug: prompt before passing into the model: {prompt}")
        # Encode prompt
        inputs = self.encode(prompt)

        if self.device == "mps" and "attention_mask" in inputs:
            # to makesure the attention masks use dtype int32 bc that is supported by pytorch on mps
            inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.int32)

        # Generate continuation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens= max_tokens,
            do_sample=True,
            temperature= temperature
        )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return generated_text



