"""The grading adapter from commit 6ad11d55 names its weights after Qwen3.5's multimodal
layout (`model.language_model.layers.N`). Loaded onto AutoModelForCausalLM
(`model.layers.N`), PeftModel.from_pretrained matched none of them and the "QLoRA" run
graded like the base model. These tests rebuild that with a tiny random model."""
import os

import pytest

from src.gfv.llm import adapter_key_map

PREFIX = "base_model.model.model."


def test_key_map_renames_multimodal_names():
    model_keys = [PREFIX + "layers.0.mlp.up_proj.lora_A.weight", PREFIX + "layers.0.mlp.up_proj.lora_B.weight"]
    saved = [key.replace(PREFIX, PREFIX + "language_model.") for key in model_keys]
    assert adapter_key_map(saved, model_keys) == dict(zip(saved, model_keys))
    assert adapter_key_map(model_keys, model_keys) == dict(zip(model_keys, model_keys))


def test_key_map_rejects_an_adapter_that_does_not_fit():
    model_keys = [PREFIX + "layers.0.mlp.up_proj.lora_A.weight"]
    with pytest.raises(ValueError, match="does not fit"):
        adapter_key_map([PREFIX + "layers.0.self_attn.q_proj.lora_A.weight"], model_keys)
    with pytest.raises(ValueError, match="does not fit"):
        adapter_key_map([], model_keys)


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    return LlamaForCausalLM(config).eval()


def test_multimodal_named_adapter_loads_onto_causal_lm(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file, save_file

    from src.gfv.llm import load_adapter

    # Train-like adapter: random LoRA B so the adapter changes the output.
    trained = peft.get_peft_model(tiny_model(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"]))
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    saved_dir = tmp_path / "adapter"
    trained.save_pretrained(saved_dir)
    weights_path = os.path.join(saved_dir, "adapter_model.safetensors")
    save_file({key.replace(PREFIX, PREFIX + "language_model."): value
               for key, value in load_file(weights_path).items()}, weights_path)

    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = trained(inputs).logits
        base = tiny_model()(inputs).logits

        # The bug: plain PEFT loading leaves the adapter at zero, so the output is the base model's.
        plain = peft.PeftModel.from_pretrained(tiny_model(), str(saved_dir))
        assert torch.allclose(plain(inputs).logits, base)

        fixed = load_adapter(tiny_model(), str(saved_dir))
        assert torch.allclose(fixed(inputs).logits, expected)
        assert not torch.allclose(expected, base)


def test_adapter_for_another_model_is_refused(tmp_path):
    pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from src.gfv.llm import load_adapter

    other = peft.get_peft_model(tiny_model(), peft.LoraConfig(r=4, target_modules=["q_proj"]))
    other.save_pretrained(tmp_path / "adapter")
    # Same target modules but one layer fewer in the model: a weight with no layer to go to.
    model = tiny_model()
    model.model.layers = model.model.layers[:1]
    model.config.num_hidden_layers = 1
    with pytest.raises(ValueError, match="does not fit"):
        load_adapter(model, str(tmp_path / "adapter"))
