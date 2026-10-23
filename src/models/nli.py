from __future__ import annotations

from typing import Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLI:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        idx = [int(i) for i, lab in self.model.config.id2label.items() if lab.lower().startswith("entail")]
        if len(idx) != 1:
            raise ValueError(f"Could not identify a unique entailment label in {self.model.config.id2label}")
        self.entail_idx = idx[0]

    @torch.inference_mode()
    def p_entail_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(probs[:, self.entail_idx].tolist())
        return out

    def p_entail(self, premise: str, hypothesis: str) -> float:
        return self.p_entail_batch([(premise, hypothesis)])[0]