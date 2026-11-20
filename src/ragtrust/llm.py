"""Interface to the language model.

All generation goes through OllamaLLM.chat(), so every setting that affects
the output is stored in one place (and can be saved with the results).
To use another backend later (vLLM, an API), write a class with the same
chat() method.
"""

import warnings
from dataclasses import asdict, dataclass

import ollama


@dataclass
class OllamaLLM:
    model: str = "llama3.1:8b"
    temperature: float = 0.0  # 0 = always the most likely token
    seed: int = 0
    num_ctx: int = 4096       # max prompt + answer tokens; Ollama silently cuts longer prompts
    num_predict: int = 32     # max answer tokens
    think: bool | None = None  # "thinking" models only: False = no hidden reasoning; None = model default

    def chat(self, messages: list[dict], format: dict | None = None) -> str:
        """Send the messages to the model and return its answer text.

        format: optional JSON schema; Ollama then forces the answer to be valid JSON
        that follows it ("structured outputs").
        """
        # Rough guard against silent truncation: English is ~4 characters per token.
        approx_tokens = sum(len(m["content"]) for m in messages) / 4
        if approx_tokens > 0.8 * self.num_ctx:
            warnings.warn(f"Prompt is ~{approx_tokens:.0f} tokens, close to num_ctx={self.num_ctx}.")

        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        extra = {}
        if self.think is not None:
            extra["think"] = self.think
        if format is not None:
            extra["format"] = format
        response = ollama.chat(model=self.model, messages=messages, options=options, **extra)
        text = response["message"]["content"].strip()
        if not text:
            # An empty answer would silently be scored as wrong, so make it visible.
            warnings.warn(f"Empty answer from {self.model} (done_reason={response.get('done_reason')}).")
        return text

    def settings(self) -> dict:
        """All settings as a dict, to save next to the results."""
        return asdict(self)
