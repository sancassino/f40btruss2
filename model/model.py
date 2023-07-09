import torch
import transformers
from typing import Dict

CHECKPOINT = "tiiuae/falcon-40b-instruct"

class Model:
    def __init__(self, data_dir: str, config: Dict, secrets: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=CHECKPOINT,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate_response(self, prompt: str) -> str:
        with torch.no_grad():
            try:
                data = self.pipeline(
                    prompt,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    max_new_tokens=300
                )[0]
                return data["generated_text"]

            except Exception as exc:
                return str(exc)

