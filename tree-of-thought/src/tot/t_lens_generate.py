"""
Hybrid loader for TLens (open/base models) and HF Transformers (instruct/gated models).
"""

import os
import torch
from transformer_lens import HookedTransformer

# Optional aliases → canonical HF repo ids
ALIASES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
}

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

_hf_singleton = None  # keep same interface as before


def get_tlens_model(model_id):
    global _hf_singleton
    if _hf_singleton is None:
        _hf_singleton = LLM(model_id=model_id)
    return _hf_singleton


class LLM:
    def __init__(self, model_id="gpt2", args=None):
        self.args = args

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        model_id = ALIASES.get(model_id, model_id)
        self.model_id = model_id
        self.use_hf = ("/" in model_id)  # HF path for Llama/Mistral/etc.

        self.model = None               # TransformerLens model
        self.hf_model = None            # HF Transformers model
        self.tokenizer = None
        self.has_chat_template = False

        if self.use_hf:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                token=HF_TOKEN,
                trust_remote_code=True,
            )

            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=HF_TOKEN,
                trust_remote_code=True,
            )

            self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        else:
            # TransformerLens path (gpt2/pythia/etc.)
            self.model = HookedTransformer.from_pretrained(
                model_id,
                device=self.device,
                hf_access_token=HF_TOKEN,
                trust_remote_code=True,
            )

    def generate(self, prompt, max_tokens: int = 250, temperature: float = 0.6):
        if self.use_hf:
            # Use chat template if available (for Instruct models)
            if self.has_chat_template:
                prompt_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = prompt

            enc = self.tokenizer(prompt_text, return_tensors="pt")
            enc = {k: v.to(self.hf_model.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.hf_model.generate(
                    **enc,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_ids = out[0][enc["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # TransformerLens path
        toks = self.model.to_tokens(prompt, prepend_bos=True)
        gen = self.model.generate(
            toks,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=temperature,
        )
        new = gen[0, toks.shape[1]:]  # only newly generated tokens
        return self.model.to_string(new)
