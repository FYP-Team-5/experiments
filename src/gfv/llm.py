import time
from typing import Dict, List, Tuple

Messages = List[Dict[str, str]]


def is_out_of_memory(error: BaseException) -> bool:
    """True for torch.OutOfMemoryError / torch.cuda.OutOfMemoryError without importing torch."""
    return type(error).__name__ == "OutOfMemoryError"


def adapter_key_map(saved_keys, model_keys) -> Dict[str, str]:
    """Maps each weight name in a saved adapter to the name the loaded model uses.

    Qwen3.5 loaded as the multimodal model names its text layers
    `model.language_model.layers.N`; loaded with AutoModelForCausalLM, as here, they are
    `model.layers.N`. An adapter trained on the first (SmuFypTeam5/GradingQlora since
    commit 6ad11d55) matches nothing in the second, and PEFT only warns: its LoRA layers
    stay at their zero initialisation and the model grades like the base model.
    Raises if a saved weight has no match or a LoRA weight in the model gets none.
    """
    def normalise(key):
        return key.replace("language_model.", "")

    by_name = {normalise(key): key for key in model_keys}
    mapping = {key: by_name[normalise(key)] for key in saved_keys if normalise(key) in by_name}
    unmatched = sorted(set(saved_keys) - set(mapping))
    uncovered = sorted(set(model_keys) - set(mapping.values()))
    if unmatched or uncovered:
        raise ValueError(
            f"The adapter does not fit the model: {len(unmatched)} saved weights match no layer "
            f"(e.g. {unmatched[:2]}), {len(uncovered)} LoRA weights get no value (e.g. {uncovered[:2]})."
        )
    return mapping


def load_adapter(base_model, adapter_id: str):
    """PeftModel.from_pretrained, then a check that every adapter weight really loaded.
    Weights saved under the multimodal layer names are renamed to the model's own."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils import load_peft_weights

    model = PeftModel.from_pretrained(base_model, adapter_id)
    saved = load_peft_weights(adapter_id)
    mapping = adapter_key_map(saved, get_peft_model_state_dict(model))
    renamed = sum(name != target for name, target in mapping.items())
    if renamed:
        print(f"Adapter {adapter_id}: renaming {renamed} weights to the model's layer names", flush=True)
        result = set_peft_model_state_dict(model, {mapping[name]: value for name, value in saved.items()})
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"Adapter weights not loaded: {result.unexpected_keys[:2]}")

    loaded = get_peft_model_state_dict(model)
    wrong = []
    for name, value in saved.items():
        current = loaded[mapping[name]].detach().cpu()
        if not torch.equal(current, value.to(current.dtype).cpu()):
            wrong.append(name)
    if wrong:
        raise ValueError(f"Adapter {adapter_id}: {len(wrong)} of {len(saved)} weights differ from the file "
                         f"after loading (e.g. {wrong[:2]}).")
    print(f"Adapter {adapter_id}: all {len(saved)} weights loaded", flush=True)
    return model


def clear_cuda_cache():
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LLMWrapper:
    """Unified interface for loading models with optional quantization and PEFT adapters."""

    def __init__(self, config: dict):
        self.cfg = config
        self.model, self.tokenizer = self._load_model_and_tokenizer(self.cfg)

    @staticmethod
    def _load_model_and_tokenizer(cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if "model" not in cfg or "base_model_id" not in cfg["model"]:
            raise KeyError("Configuration must contain a 'model' section with 'base_model_id'.")

        bnb_config = None
        if cfg.get("quantization", {}).get("load_in_4bit"):
            compute_dtype = getattr(torch, cfg["quantization"].get("bnb_4bit_compute_dtype", "float16"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=cfg["quantization"]["load_in_4bit"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=cfg["quantization"].get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=cfg["quantization"].get("bnb_4bit_use_double_quant", False),
            )

        dtype = getattr(torch, cfg["model"].get("torch_dtype", "float16"))
        device_map = cfg["model"].get("device_map", "auto")
        if device_map == "auto" and torch.cuda.is_available():
            device_map = "cuda:0"

        print(f"Loading {cfg['model']['base_model_id']} (adapter: {cfg['model'].get('adapter_id')})", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_model_id"],
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if cfg["model"].get("adapter_id"):
            model = load_adapter(base_model, cfg["model"]["adapter_id"])
        else:
            model = base_model
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_id"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left padding so every row in a batch continues straight after its own prompt.
        tokenizer.padding_side = "left"
        return model, tokenizer

    def render(self, messages: Messages, enable_thinking: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def generate_batch(self, message_lists: List[Messages], decoding_params: dict,
                       enable_thinking: bool = False) -> Tuple[List[str], float]:
        """Generates one reply per chat. Returns the replies and the wall-clock seconds for the batch."""
        import torch

        prompts = [self.render(messages, enable_thinking) for messages in message_lists]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_width = inputs["input_ids"].shape[1]
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        do_sample = decoding_params.get("do_sample", decoding_params.get("temperature", 0) > 0)
        sampling = {"do_sample": do_sample}
        if do_sample:
            sampling["temperature"] = decoding_params.get("temperature", 0.7)
            if "top_p" in decoding_params:
                sampling["top_p"] = decoding_params["top_p"]

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decoding_params.get("max_new_tokens", 256),
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
            )
        elapsed = time.perf_counter() - started

        replies = self.tokenizer.batch_decode(outputs[:, input_width:].cpu(), skip_special_tokens=True)
        return [reply.strip() for reply in replies], elapsed

    def generate(self, messages: Messages, decoding_params: dict, enable_thinking: bool = False) -> str:
        replies, _ = self.generate_batch([messages], decoding_params, enable_thinking)
        return replies[0]
