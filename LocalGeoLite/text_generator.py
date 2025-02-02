from typing import Optional
from .load_model import load_model
from .format import ALPACA_PROMPT
from transformers import TextStreamer

class text:
    
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
                    instruction="You are a GIS expert specializing in geospatial analysis, remote sensing, and coordinate systems. Based on the given input, provide a concise and accurate response related to geographic data processing, GPS applications, spatial analysis, or map projections.",  # instruction
                    input=f"Please answer the question: {prompt} and make sure you answer is concise and professionally.",  # input
                    output="",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")


        outputs = self.model.generate(
            **inputs,
            streamer=self.streamer,
            max_new_tokens=max_new_tokens,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
