from typing import Optional
from .load_model import load_model
from transformers import TextStreamer
from .format import ALPACA_PROMPT


class code:
    def __init__(self, model=None, tokenizer=None, cache_dir: Optional[str] = None):
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_model(cache_dir=cache_dir)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.streamer = TextStreamer(self.tokenizer)

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        inputs = self.tokenizer(
            [
                ALPACA_PROMPT.format(
                    instruction="You are an expert in GIS and Python programming, specializing in producing high-quality and concise Python code, and do not comment the code.",  # instruction
                    input=f"Please generate the code for {prompt} using Python and make sure the syntax is correct.",  # input
                    output="",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            streamer=self.streamer,
            max_new_tokens=max_new_tokens
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

