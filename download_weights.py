from typing import NamedTuple, Optional
import os
import ml_dtypes
import jax.numpy as jnp
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from unittest.mock import patch

MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'# Model configuration constants

config = AutoConfig.from_pretrained(MODEL_ID)
print(config)
HEAD_DIM = config.hidden_size // config.num_attention_heads  # = 64
KV_HEAD_DIM = config.hidden_size // config.num_key_value_heads  # = 192

def print_tensor_info(tensor: torch.Tensor, name: str):
    """Helper function to debug tensor shapes and sizes"""
    print(f"\n{name} tensor info:")
    print(f"Shape: {tensor.shape}")
    print(f"Size: {tensor.numel()}")
    print(f"HEAD_DIM: {HEAD_DIM}")
    print(f"KV_HEAD_DIM: {KV_HEAD_DIM}")

def reverse_permute(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Reverse the permutation of attention weights.
    For q: shape=[576, 576] -> view=[9, 64, 576] -> transpose -> reshape=[576, 576]
    For k: shape=[192, 576] -> view=[3, 64, 576] -> transpose -> reshape=[192, 576]
    """
    if n_heads == config.num_attention_heads:  # Query heads (576, 576)
        return (tensor
                .view(config.num_attention_heads, HEAD_DIM, config.hidden_size)
                .transpose(1, 2)
                .reshape(config.hidden_size, config.hidden_size))
    else:  # Key heads (192, 576)
        head_dim = tensor.shape[0] // config.num_key_value_heads  # = 64
        return (tensor
                .view(config.num_key_value_heads, head_dim, config.hidden_size)
                .transpose(1, 2)
                .reshape(config.num_key_value_heads * head_dim, config.hidden_size))

def translate_key(in_key: str) -> str:
    """Translate HuggingFace key names to your format"""
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        replacements = {
            'input_layernorm': 'attention_norm',
            'mlp.down_proj': 'feed_forward.w2',
            'mlp.gate_proj': 'feed_forward.w1',
            'mlp.up_proj': 'feed_forward.w3',
            'post_attention_layernorm': 'ffn_norm',
            'self_attn.k_proj': 'attention.wk',
            'self_attn.o_proj': 'attention.wo',
            'self_attn.q_proj': 'attention.wq',
            'self_attn.v_proj': 'attention.wv',
            'embed_tokens': 'tok_embeddings',
            'norm': 'norm'
        }
        for old, new in replacements.items():
            if out_key.endswith(old):
                out_key = out_key.replace(old, new)
                break
        else:
            if not out_key == 'norm':
                print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def download_weights(model_id: str = MODEL_ID, out_dir: Path = Path('weights/1B-Instruct')):
    device = torch.device("cpu")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            offload_folder="/tmp/offload",
            device_map='cpu'
        )
        
        with torch.no_grad():
            state_dict = hf_model.state_dict()
            for hf_name, param in state_dict.items():
                print(f'\nProcessing {hf_name}: shape={param.shape}')
                name = translate_key(hf_name)
                param = param.cpu()

                if name.endswith('wq.weight'):
                    print_tensor_info(param, "Query weight")
                    param = reverse_permute(param, n_heads=config.num_attention_heads)
                elif name.endswith('wk.weight'):
                    print_tensor_info(param, "Key weight")
                    param = reverse_permute(param, n_heads=config.num_key_value_heads)

                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                jnp.save(f'{out_dir}/{name}.npy', bf16_out)

    del hf_model
    del state_dict

download_weights()