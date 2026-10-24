import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LLMWrapper:
    """Unified interface for loading models with optional quantization and PEFT adapters."""
    
    def __init__(self, config: dict):
        self.cfg = config
        self.model, self.tokenizer = self._load_model_and_tokenizer(self.cfg)

    @staticmethod
    def _load_model_and_tokenizer(cfg: dict):
        if "model" not in cfg or "base_model_id" not in cfg["model"]:
            raise KeyError("Configuration must contain a 'model' section with 'base_model_id'.")

        bnb_config = None
        if cfg.get("quantization", {}).get("load_in_4bit"):
            compute_dtype = getattr(torch, cfg["quantization"].get("bnb_4bit_compute_dtype", "float16"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=cfg["quantization"]["load_in_4bit"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=cfg["quantization"].get("bnb_4bit_quant_type", "nf4")
            )

        torch_dtype = getattr(torch, cfg["model"].get("torch_dtype", "float16"))
        device_map = cfg["model"].get("device_map", "cuda")
        if device_map == "auto" and torch.cuda.is_available():
            device_map = "cuda:0"

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_model_id"],
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        if cfg["model"].get("adapter_id"):
            model = PeftModel.from_pretrained(base_model, cfg["model"]["adapter_id"])
        else:
            model = base_model

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_id"])
        return model, tokenizer

    def generate(self, prompt: str, decoding_params: dict) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert pedagogical assistant and objective evaluator. "
                    "Your task is to provide ONLY direct, constructive feedback to the student based on the grading rubric. "
                    "Do NOT repeat the student answer. Do NOT include any section titles, headings (such as '### EXAMPLE' or 'Summary Assessment:'), preambles, or labels. "
                    "Start directly with the first sentence of the feedback text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print(f"Message: ", messages)

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decoding_params.get("max_new_tokens", 256),
                temperature=decoding_params.get("temperature", 0.1),
                top_p=decoding_params.get("top_p", 0.95),
                do_sample=decoding_params.get("temperature", 0.1) > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_len = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()