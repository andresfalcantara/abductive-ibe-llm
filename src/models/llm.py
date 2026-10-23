from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if dtype is None:
            if self.device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, trust_remote_code=trust_remote_code
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
            )
        self.model = model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        use_chat_template: bool = True,
    ) -> str:
        
        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            add_special_tokens = False
        else:
            text = prompt
            add_special_tokens = True

        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).to(self.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    # Teacher-forced scoring
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def token_logprobs(self, context: str, continuation: str) -> list[float]:

        if not context:
            raise ValueError("context must be non-empty (the first token of a sequence has no logprob)")
        if not continuation:
            raise ValueError("continuation must be non-empty")

        text = context + continuation
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(self.device)
        ctx_len = len(context)

        positions = []
        for i, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if start < ctx_len < end:
                raise ValueError(
                    "A token straddles the context/continuation boundary. "
                    "End the context with a newline or start the continuation with a space."
                )
            if start >= ctx_len:
                positions.append(i)

        if not positions:
            raise ValueError("No continuation tokens found after tokenization")
        if positions[0] == 0:
            raise ValueError("First token of the sequence cannot be scored (no preceding context)")

        logits = self.model(input_ids).logits[0]
        pos = torch.tensor(positions, device=logits.device)
        
        logp = torch.log_softmax(logits[pos - 1].float(), dim=-1)
        targets = input_ids[0, pos]
        return logp.gather(1, targets.unsqueeze(1)).squeeze(1).tolist()

    def mean_logprob(self, context: str, continuation: str) -> float:
        lps = self.token_logprobs(context, continuation)
        return sum(lps) / len(lps)