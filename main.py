
from typing import (
    NamedTuple, 
    Optional, 
    Tuple, 
    List, 
    AbstractSet, 
    Collection, 
    Dict, 
    Iterator, 
    Literal, 
    Sequence, 
    Union
)
import torch
import torch.nn.functional as F
import torch.nn as nn
import tyro
import os
import ml_dtypes
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from logging import getLogger
import json
import math
from enum import Enum
os.system('clear')
logger = getLogger(__name__)

# The following constants remain unchanged
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

MODEL_ID = 'HuggingFaceTB/SmolLM-135M-Instruct' #No need for tokens as Smollm is not gated!


def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

params = {
    "dim": 576,
    "n_layers": 30,
    "n_heads": 9,
    "n_kv_heads": 3,
    "vocab_size": 49152,
    "norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,  # Inferred from "rope_scaling": null
    "max_seq_len": 2048,  # Inferred from "max_position_embeddings"
}

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool


LLAMA_1B_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)



jax.clear_caches()
torch.cuda.empty_cache()



class LayerWeights(NamedTuple):
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  w3: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  norm: torch.Tensor
  output: torch.Tensor
  layer_weights: List[LayerWeights]

def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 30):
  w = {}
  layer_weights = []
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
      #print(f'JAX output (first 30): {jax_weight.flatten()[:30]}')
      np_weight = np.array(jax_weight).astype(np.float32)
      weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
      w[name] = weight.to(device)
    for i in range(n_layers):
      layer_weights.append(LayerWeights(
        wq=w[f'layers.{i}.attention.wq.weight'],
        wk=w[f'layers.{i}.attention.wk.weight'],
        wv=w[f'layers.{i}.attention.wv.weight'],
        wo=w[f'layers.{i}.attention.wo.weight'],
        w1=w[f'layers.{i}.feed_forward.w1.weight'],
        w2=w[f'layers.{i}.feed_forward.w2.weight'],
        w3=w[f'layers.{i}.feed_forward.w3.weight'],
        ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
        attention_norm=w[f'layers.{i}.attention_norm.weight'],
      ))

    xfmr_weights = XfmrWeights(
      tok_embeddings=w['tok_embeddings.weight'],
      norm=w['norm.weight'],
      output=w['output.weight'],
      layer_weights=layer_weights
    )

    return xfmr_weights



class Tokenizer:
    """
    Tokenizing and encoding/decoding text using a tokenizer.json file.
    """

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 17

    def __init__(self, tokenizer_path: str):
        """
        Initializes the Tokenizer with a tokenizer.json file.

        Args:
            tokenizer_path (str): The path to the tokenizer.json file.
        """
        assert os.path.isfile(tokenizer_path), tokenizer_path

        self.model = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        special_tokens = [
            '<|endoftext|>',
            '<|im_start|>',
            '<|im_end|>',
            "<repo_name>",
            "<reponame>",
            "<file_sep>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<jupyter_script>",
            "<empty_output>"
        ]

        self.special_tokens = {token: self.model.convert_tokens_to_ids(token) for token in special_tokens}

        self.n_words: int = self.model.vocab_size
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens['<|im_start|>']
        self.eos_id: int = self.special_tokens['<|im_end|>']
        self.eot_id: int = self.special_tokens['<|im_start|>']
        self.eom_id: int = self.special_tokens['<|im_end|>']
        self.python_tag_id = self.special_tokens['<jupyter_code>']
        self.pad_id: int = self.special_tokens['<|endoftext|>']
        self.stop_tokens = [
            self.special_tokens['<|im_start|>'],
            self.special_tokens['<|im_end|>'],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal['all'], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal['all'], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.
        """
        if allowed_special is None:
            allowed_special = set()
        assert isinstance(s, str)

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr, add_special_tokens=False))
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.model.decode(t)

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]





class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super(KVCache, self).__init__()
        # Initialize k and v as buffers to ensure they're part of the module state
        self.register_buffer(
            'k',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )
        self.register_buffer(
            'v',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Creates a new KVCache instance with initialized k and v tensors."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    def update(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        layer_idx: int,
        cur_pos: int,
        n_rep: int
    ):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (torch.Tensor): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (torch.Tensor): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        # Ensure xk and xv have the correct device and dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        # Update the k and v tensors in the specified layer and position
        insert_len = xk.size(1)  # Assuming xk shape is (bsz, insert_len, kv_heads, head_dim)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self

    def clear(self):
        """Resets the k and v caches to zeros."""
        self.k.zero_()
        self.v.zero_()





class AttnStats(NamedTuple):
    entropy: torch.Tensor  # (bsz, n_layers, num_heads)
    varentropy: torch.Tensor  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return self.entropy.sum(dim=-1, keepdim=False)  # Average across heads

    @property
    def std_error(self):
        return torch.sqrt(torch.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: torch.Tensor, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
        new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1))**2, dim=-1)

        # Update entropy and varentropy tensors
        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self



def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
  return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    # Check if x is 2D or 3D and adjust accordingly
    if x.dim() == 2:
        bsz = 1
        seq_len, dim = x.shape
        x = x.unsqueeze(0)  # Add batch dimension
    else:
        bsz, seq_len, dim = x.shape

    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).view(bsz, seq_len, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).view(bsz, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).view(bsz, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = xq.permute(0, 2, 1, 3)  # (bs, n_heads, seqlen, head_dim)
    keys = keys.permute(0, 2, 3, 1)  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = values.permute(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(x.dtype)
    output = torch.matmul(scores.to(values.dtype), values)
    output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
    out = F.linear(output, layer_weights.wo)

    # If input was 2D, remove the batch dimension from the output
    if x.dim() == 2:
        out = out.squeeze(0)

    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats


class SamplerConfig:
    def __init__(self):
        self.temperature = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03

        self.low_logits_entropy_threshold = 0.6
        self.medium_logits_entropy_threshold = 1.584
        self.high_logits_entropy_threshold = 2.17

        self.low_logits_varentropy_threshold = 3.28
        self.medium_logits_varentropy_threshold = 3.85
        self.high_logits_varentropy_threshold = 6.18

        self.low_attention_entropy_threshold = 8.989
        self.medium_attention_entropy_threshold = 8.99
        self.high_attention_entropy_threshold = 8.991

        self.low_attention_varentropy_threshold = 5.212
        self.medium_attention_varentropy_threshold = 5.9125
        self.high_attention_varentropy_threshold = 6.92

        self.low_agreement_threshold = 2e-06
        self.medium_agreement_threshold = 4e-06
        self.high_agreement_threshold = 5e-06

        self.low_interaction_strength_threshold = 0.2
        self.medium_interaction_strength_threshold = 0.247
        self.high_interaction_strength_threshold = 0.264

        self.high_entropy_attention_offset = 1.3
        self.high_entropy_attention_coefficient = 0.2

        self.low_entropy_interaction_strength_offset = 1.2
        self.low_entropy_interaction_strength_coefficient = 0.3

        self.high_entropy_varentropy_attention_offset = 2.0
        self.high_entropy_varentropy_attention_coefficient = 0.5

        self.n_adaptive_samples = 5

        self.adaptive_temperature_logits_coefficient = 0.3
        self.adaptive_temperature_attention_coefficient = 0.2
        self.adaptive_temperature_agreement_coefficient = 0.2
        self.adaptive_top_p_coefficient = 0.1
        self.adaptive_top_k_interaction_coefficient = 0.3
        self.adaptive_top_k_agreement_coefficient = 0.2
        self.adaptive_min_p_coefficient = 0.5
        self.adaptive_score_logits_entropy_coefficient = 0.1
        self.adaptive_score_attention_entropy_coefficient = 0.2
        self.adaptive_score_logits_varentropy_coefficient = 0.3
        self.adaptive_score_attention_varentropy_coefficient = 0.4
        self.adaptive_score_agreement_coefficient = 0.5
        self.adaptive_score_interaction_strength_coefficient = 0.6






class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(
    logits: torch.Tensor,
    temperature: float,
    min_p: float,
    top_p: float = None,  # Made optional
    top_k: int = None,      # Made optional
    generator: torch.Generator = None
) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)
        probs = F.softmax(logit, dim=-1)

    # Apply top-k sampling if top_k is provided
    if top_k is not None:
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        probs_sort = torch.flip(top_k_probs, dims=[-1])
        probs_idx = torch.flip(top_k_indices, dims=[-1])
    else:
        probs_sort = probs
        probs_idx = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0).repeat(bsz, 1)

    # Apply top-p sampling if top_p is provided
    if top_p is not None:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        probs_sort = probs_sort * (1 - mask)
        probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)

    next_token = multinomial_sample_one(probs_sort, generator)
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)

    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

def sample(
    gen_tokens: torch.Tensor,
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    cfg: 'SamplerConfig',  # Assuming SamplerConfig is defined elsewhere
    clarifying_question_token: int = 2564,
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)
) -> Tuple[torch.Tensor, SamplerState]:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold):
        sampler_state = SamplerState.FLOWING
        sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        return sampled_token, sampler_state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold):
        sampler_state = SamplerState.TREADING
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:, -1], torch.tensor([clarifying_question_token], device=device)).any():
            sampled_token = torch.tensor([[clarifying_question_token]], dtype=torch.int32, device=device)
            return torch.tensor([[999999]]).to(device), sampler_state
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent
            sampled_token = _sample(
                logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_p,
                generator=generator
            )
            return sampled_token, sampler_state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (ent < cfg.high_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold):
        sampler_state = SamplerState.EXPLORING
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        sampled_token = _sample(
            logits,
            temperature=min(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=top_k_adj,
            min_p=cfg.min_p,
            generator=generator
        )
        return  torch.tensor([[999999]]).to(device), sampler_state

    # High Entropy, High Varentropy: "resampling in the mist"
    elif (ent > cfg.medium_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold):
        sampler_state = SamplerState.RESAMPLING
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attn_ent)
        sampled_token = _sample(
            logits,
            temperature=max(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            generator=generator
        )
        return sampled_token, sampler_state

    # Middle ground: use adaptive sampling
    else:
        sampler_state = SamplerState.ADAPTIVE
        logits_uncertainty = ent + vent
        attn_uncertainty = attn_ent + attn_vent

        temperature = cfg.temperature * (
            1 +
            cfg.adaptive_temperature_logits_coefficient * ent +
            cfg.adaptive_temperature_attention_coefficient * attn_ent -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        # top_p and top_k are removed from adaptive sampling
        # top_p = torch.clamp(
        #     (cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent)).clone().detach(),
        #     0.1,
        #     1.0
        # )
        # top_k = int(torch.clamp(
        #     torch.round(torch.tensor(cfg.top_k) * (
        #         1 +
        #         cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() -
        #         cfg.adaptive_top_k_agreement_coefficient * agreement.item()
        #     )),
        #     min=1,
        #     max=100
        # ).item())
        # Removed coefficient-based min_p adjustments
        # min_p = torch.clamp(
        #     (cfg.min_p * (1 - cfg.adaptive_min_p_coefficient * vent)).clone().detach(),
        #     0.01,
        #     0.5
        # )
        min_p = cfg.min_p  # Use only min_p without coefficients

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = _sample(
                logits,
                temperature=temperature,
                # top_p=top_p,  # Removed
                # top_k=top_k,  # Removed
                min_p=min_p,
                generator=generator
            )
            samples.append(sample)

        def score_sample(sample):
            # Ensure sample is a 1D tensor of indices
            sample_indices = sample.view(-1).to(torch.long)

            # Create one-hot encoding
            one_hot = F.one_hot(sample_indices, num_classes=logits.shape[-1])

            # Calculate log probability
            log_probs = F.log_softmax(logits[:, -1], dim=-1)
            log_prob = torch.sum(log_probs * one_hot, dim=-1)

            confidence_score = (
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        sampled_token = samples[best_sample_idx]
        return sampled_token, sampler_state




def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 2048  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask

class EntropixModel:
    def __init__(self):
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = load_weights()
        self.tokenizer = Tokenizer('tokenizer.json')
        self.sampler_config = SamplerConfig()
        self.generator = torch.Generator(device=device).manual_seed(1337)

    def visualize_token_entropy_varentropy(self, metrics_data, generated_tokens):
        # Extract data
        entropies = np.array(metrics_data['logits_entropy'])
        varentropies = np.array(metrics_data['logits_varentropy'])
        attention_entropies = np.array(metrics_data['attention_entropy'])
        attention_varentropies = np.array(metrics_data['attention_varentropy'])

        # Ensure all arrays have the same length
        min_length = min(len(entropies), len(varentropies), len(attention_entropies), len(attention_varentropies), len(generated_tokens))
        entropies = entropies[:min_length]
        varentropies = varentropies[:min_length]
        attention_entropies = attention_entropies[:min_length]
        attention_varentropies = attention_varentropies[:min_length]
        generated_tokens = generated_tokens[:min_length]

        positions = np.arange(min_length)

        # Create hover text
        hover_text = [
            f"Token: {self.tokenizer.decode([token]) or 'Unknown'}<br>"
            f"Position: {i}<br>"
            f"Logits Entropy: {entropies[i]:.4f}<br>"
            f"Logits Varentropy: {varentropies[i]:.4f}<br>"
            f"Attention Entropy: {attention_entropies[i]:.4f}<br>"
            f"Attention Varentropy: {attention_varentropies[i]:.4f}"
            for i, token in enumerate(generated_tokens)
        ]

        # Create the 3D scatter plot
        fig = go.Figure()

        # Add logits entropy/varentropy scatter
        fig.add_trace(go.Scatter3d(
            x=entropies,
            y=varentropies,
            z=positions,
            mode='markers',
            marker=dict(
                size=5,
                color=entropies,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Logits Entropy", x=0.85),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Logits Entropy/Varentropy'
        ))

        # Add attention entropy/varentropy scatter
        fig.add_trace(go.Scatter3d(
            x=attention_entropies,
            y=attention_varentropies,
            z=positions,
            mode='markers',
            marker=dict(
                size=5,
                color=attention_entropies,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Attention Entropy", x=1.0),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Attention Entropy/Varentropy'
        ))

        # Calculate the limits for x, y, and z
        logits_x_min, logits_x_max = min(entropies), max(entropies)
        logits_y_min, logits_y_max = min(varentropies), max(varentropies)
        attention_x_min, attention_x_max = min(attention_entropies), max(attention_entropies)
        attention_y_min, attention_y_max = min(attention_varentropies), max(attention_varentropies)
        z_min, z_max = min(positions), max(positions)

        # Function to create threshold planes
        def create_threshold_plane(threshold, axis, color, name, data_type):
            if data_type == 'logits':
                x_min, x_max = logits_x_min, logits_x_max
                y_min, y_max = logits_y_min, logits_y_max
            else:  # attention
                x_min, x_max = attention_x_min, attention_x_max
                y_min, y_max = attention_y_min, attention_y_max

            if axis == 'x':
                return go.Surface(
                    x=[[threshold, threshold], [threshold, threshold]],
                    y=[[y_min, y_max], [y_min, y_max]],
                    z=[[z_min, z_min], [z_max, z_max]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=name,
                    visible=False
                )
            elif axis == 'y':
                return go.Surface(
                    x=[[x_min, x_max], [x_min, x_max]],
                    y=[[threshold, threshold], [threshold, threshold]],
                    z=[[z_min, z_min], [z_max, z_max]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=name,
                    visible=False
                )

        # Add threshold planes
        thresholds = [
            ('logits_entropy', 'x', [
                (self.sampler_config.low_logits_entropy_threshold, 'rgba(255, 0, 0, 0.2)'),
                (self.sampler_config.medium_logits_entropy_threshold, 'rgba(0, 255, 0, 0.2)'),
                (self.sampler_config.high_logits_entropy_threshold, 'rgba(0, 0, 255, 0.2)')
            ], 'logits'),
            ('logits_varentropy', 'y', [
                (self.sampler_config.low_logits_varentropy_threshold, 'rgba(255, 165, 0, 0.2)'),
                (self.sampler_config.medium_logits_varentropy_threshold, 'rgba(165, 42, 42, 0.2)'),
                (self.sampler_config.high_logits_varentropy_threshold, 'rgba(128, 0, 128, 0.2)')
            ], 'logits'),
            ('attention_entropy', 'x', [
                (self.sampler_config.low_attention_entropy_threshold, 'rgba(255, 192, 203, 0.2)'),
                (self.sampler_config.medium_attention_entropy_threshold, 'rgba(0, 255, 255, 0.2)'),
                (self.sampler_config.high_attention_entropy_threshold, 'rgba(255, 255, 0, 0.2)')
            ], 'attention'),
            ('attention_varentropy', 'y', [
                (self.sampler_config.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
                (self.sampler_config.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
                (self.sampler_config.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)')
            ], 'attention')
        ]

        for threshold_type, axis, threshold_list, data_type in thresholds:
            for threshold, color in threshold_list:
                fig.add_trace(create_threshold_plane(threshold, axis, color, f'{threshold_type.replace("_", " ").title()} Threshold: {threshold}', data_type))

        # Create buttons for toggling views
        buttons = [
            dict(
                label='Show All',
                method='update',
                args=[{'visible': [True] * len(fig.data)}]
            ),
            dict(
                label='Hide All',
                method='update',
                args=[{'visible': [True, True] + [False] * (len(fig.data) - 2)}]
            ),
            dict(
                label='Logits Only',
                method='update',
                args=[{'visible': [True, False] + [True if i < 6 else False for i in range(len(fig.data) - 2)]}]
            ),
            dict(
                label='Attention Only',
                method='update',
                args=[{'visible': [False, True] + [True if i >= 6 else False for i in range(len(fig.data) - 2)]}]
            )
        ]

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Entropy',
                yaxis_title='Varentropy',
                zaxis_title='Token Position',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title='',
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.1,
                xanchor='left',
                yanchor='top',
                pad={"r": 10, "t": 10},
                showactive=True,
                buttons=buttons
            )],
            autosize=True,
            legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        )
 
        export_data = {
            "tokens": [self.tokenizer.decode([token]) for token in generated_tokens],
            "logits_entropy": metrics_data['logits_entropy'],
            "logits_varentropy": metrics_data['logits_varentropy'],
            "attention_entropy": metrics_data['attention_entropy'],
            "attention_varentropy": metrics_data['attention_varentropy'],
            "thresholds": {
                "logits_entropy": {
                    "low": self.sampler_config.low_logits_entropy_threshold,
                    "medium": self.sampler_config.medium_logits_entropy_threshold,
                    "high": self.sampler_config.high_logits_entropy_threshold
                },
                "logits_varentropy": {
                    "low": self.sampler_config.low_logits_varentropy_threshold,
                    "medium": self.sampler_config.medium_logits_varentropy_threshold,
                    "high": self.sampler_config.high_logits_varentropy_threshold
                },
                "attention_entropy": {
                    "low": self.sampler_config.low_attention_entropy_threshold,
                    "medium": self.sampler_config.medium_attention_entropy_threshold,
                    "high": self.sampler_config.high_attention_entropy_threshold
                },
                "attention_varentropy": {
                    "low": self.sampler_config.low_attention_varentropy_threshold,
                    "medium": self.sampler_config.medium_attention_varentropy_threshold,
                    "high": self.sampler_config.high_attention_varentropy_threshold
                }
            }
        }

        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entropy_data_{timestamp}.json"

        # Save the data to a file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Data exported to {filename}")

        return fig

    def generate(self, prompt, max_tokens=516, debug=True, summarise=False):
        # Initialize lists to store metrics
        metrics_data = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attention_entropy': [],
            'attention_varentropy': []
        }
        status = 'generating'
        sampler_states = []
        generated_tokens = []

        with torch.inference_mode():
            tokens = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", bos=True, eos=False, allowed_special='all')
            tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(self.model_params.head_dim, self.model_params.max_seq_len, self.model_params.rope_theta, self.model_params.use_scaled_rope)
            kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim).to(device)

            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
            next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, generator=self.generator)

            metrics = calculate_metrics(logits, scores)
            for key in metrics_data.keys():
                if key in metrics:
                    metrics_data[key].append(metrics[key].item())
            sampler_states.append(sampler_state)

            gen_tokens = next_token
            output = self.tokenizer.decode([next_token.item()])
            cur_pos = seqlen
            stop = torch.tensor([0, 2], device=device, dtype=torch.int32)

            while cur_pos < max_tokens:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
                next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, generator=self.generator)

                metrics = calculate_metrics(logits, scores)
                for key in metrics_data.keys():
                    if key in metrics:
                        metrics_data[key].append(metrics[key].item())
                sampler_states.append(sampler_state)
                metrics_data['attention_entropy'].append(metrics['attn_entropy'].item())
                metrics_data['attention_varentropy'].append(metrics['attn_varentropy'].item())
                generated_tokens.append(next_token.item())
                gen_tokens = torch.cat((gen_tokens.to(device), next_token.to(device)), dim=1)
                output += self.tokenizer.decode(next_token.tolist()[0])
                if next_token.item() == torch.tensor([999999], device=device, dtype=torch.int32):
                    printc(status, "yellow")
                    if summarise:
                      status = 'generating'
                    else:
                      status = 'summarise'
                    break
                if torch.isin(next_token, stop).any():
                    print(status, "1")
                    status = 'done'
                    break

        if debug:
            #self.debug_visualize_metrics(metrics_data)
            self.visualize_sampler_metrics(metrics_data['logits_entropy'], metrics_data['logits_varentropy'], sampler_states)
            fig = self.visualize_token_entropy_varentropy(metrics_data, generated_tokens)
            fig.show()
        return output, status

    def debug_visualize_metrics(self, metrics_data):
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Debug Visualization of Sampler Metrics', fontsize=16)

        for idx, (key, values) in enumerate(metrics_data.items()):
            if values:  # Only plot if we have data for this metric
                row = idx // 2
                col = idx % 2
                axs[row, col].plot(values)
                axs[row, col].set_title(key)
                axs[row, col].set_xlabel('Generation Step')
                axs[row, col].set_ylabel('Value')
                axs[row, col].grid(True)

        # Add entropy_attention visualization if we have both metrics
        if metrics_data['logits_entropy'] and metrics_data['attention_entropy']:
            axs[2, 0].scatter(metrics_data['logits_entropy'], metrics_data['attention_entropy'])
            axs[2, 0].set_title('entropy_attention')
            axs[2, 0].set_xlabel('Logits Entropy')
            axs[2, 0].set_ylabel('Attention Entropy')
            axs[2, 0].grid(True)

        # Add entropy_interaction_strength visualization if we have both metrics
        if metrics_data['logits_entropy'] and metrics_data['interaction_strength']:
            axs[2, 1].scatter(metrics_data['logits_entropy'], metrics_data['interaction_strength'])
            axs[2, 1].set_title('entropy_interaction_strength')
            axs[2, 1].set_xlabel('Logits Entropy')
            axs[2, 1].set_ylabel('Interaction Strength')
            axs[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_sampler_metrics(self, entropies, varentropies, sampler_states):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), height_ratios=[4, 1], sharex=True)

        # Plot entropy and varentropy
        x = range(len(entropies))
        ax1.plot(x, entropies, label='Entropy', color='blue')
        ax1.plot(x, varentropies, label='Varentropy', color='red')
        ax1.set_ylabel('Value')
        ax1.set_title('Entropy and Varentropy over Generation Steps')
        ax1.legend()
        ax1.grid(True)

        # Define colors in the same order as SamplerState
        colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'purple']
        cmap = ListedColormap(colors)

        # Explicitly map each SamplerState to its corresponding index
        state_to_num = {
            SamplerState.FLOWING: 0,
            SamplerState.TREADING: 1,
            SamplerState.EXPLORING: 2,
            SamplerState.RESAMPLING: 3,
            SamplerState.ADAPTIVE: 4
        }

        # Map sampler states to numerical values
        numeric_states = [state_to_num[state] for state in sampler_states]

        # Define normalization to map each integer to a color without interpolation
        norm = BoundaryNorm(boundaries=[-0.5 + i for i in range(len(colors)+1)],
                          ncolors=cmap.N,
                          clip=True)

        # Plot color-coded sampler states
        im = ax2.imshow([numeric_states], cmap=cmap, norm=norm, aspect='auto',
                      extent=[0, len(numeric_states), 0, 1])
        ax2.set_yticks([])
        ax2.set_title('Sampler State over Generation Steps')

        mapped_colors = [colors[state_to_num[state]] for state in sampler_states]

        # Create a custom legend for sampler states
        legend_elements = [Patch(facecolor=colors[state_to_num[state]], edgecolor='black', label=state.value)
                          for state in SamplerState]
        ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.show()

# Function to initialize the model (to be run once)
def initialize_model():
    global entropix_model
    entropix_model = EntropixModel()
    print("Model initialized and ready to use!")

def generate_text(prompt):
    printc(prompt, "green")
    if 'entropix_model' not in globals():
        print("Model not initialized. Please run initialize_model() first.")
        return
    resp = "\nSystem: "+prompt+"\n"
    
    while True:
        response, status = entropix_model.generate(prompt, debug=False, summarise=False)
        printc((response, status))
        resp+="\nAgent: "+response+"\n"
        if status == 'summarise':

 
            summary_prompt = f"Give a brief summary of this text:\n{response}"
            resp+="\System: "+summary_prompt+"\n"
            summary, _ = entropix_model.generate(summary_prompt, debug=False, summarise=True)
            resp+="\nAgent: "+summary+"\n"

            prompt = f"{prompt}\n\nBased on this summary below, continue the reasoning:\n{summary}"
        elif status == 'done':
            break
        else:
            break

    return resp



torch.cuda.empty_cache()

initialize_model()





from openai import OpenAI
import json


def evaluate_state(agent_state:str,question:str, goal_state:str, model="gpt-4o-mini", base_url:str=None):
    client = OpenAI() if base_url is None else OpenAI(base_url=base_url)
    score = 0
    response = client.chat.completions.create(
        model=model,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that helps evaluate an agent at problem solving. Given a question and a goal state, evaluate the agent's current state",
            },
            {
                "role": "user",
                "content": str(
                    "QUESTION/START STATE:"
                    + question
                    + "\nAGENT CURRENT STATE: "
                    + agent_state
                    + "\n:CORRECT GOAL STATE: "
                    + goal_state
                ),
            },
        ],
        functions=[
            {
                "name": "evaluate_state",
                "description": "Evaluates the agent's answer compared to the answer key",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "description": "the score of the agent's answer",
                        },
                        "feedback": {
                            "type": "string",
                            "description": "in a few short words, provide feedback to the agent on their answer",
                        },

                    },
                },
            }
        ],
    )

    
    data = json.loads(response.choices[0].message.function_call.arguments)
    

    return {'reward': data.get("score"), 'feedback': data.get("feedback","No feedback")}






#some examples

examples = json.load(open("example_data.json","r"))


val = generate_text(examples[0]["start_state"])
print(val)



grade = evaluate_state(val,examples[0]["start_state"],examples[0]["solution"])
print(grade)