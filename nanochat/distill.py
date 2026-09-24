"""Helpers for on-policy distillation with an external teacher model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _content_to_text(content: Any) -> str:
    """Flatten nanochat message content into text for a chat template."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content)
    raise TypeError(f"Unsupported message content type: {type(content)}")


def conversation_to_teacher_prompt(conversation: dict) -> list[dict[str, str]]:
    """Convert a nanochat conversation into teacher chat-template messages.

    The final assistant message is the reference answer used by SFT. It is
    removed because on-policy distillation replaces it with the student's
    rollout. System instructions are merged into the first user turn to match
    nanochat's own rendering.
    """
    messages = copy.deepcopy(conversation["messages"])
    if not messages:
        raise ValueError("Conversation has no messages")

    if messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            raise ValueError("A system message must be followed by a user message")
        messages[1]["content"] = (
            _content_to_text(messages[0]["content"])
            + "\n\n"
            + _content_to_text(messages[1]["content"])
        )
        messages = messages[1:]

    if messages[-1]["role"] != "assistant":
        raise ValueError("The final message in a distillation prompt must be from the assistant")
    prompt_messages = messages[:-1]
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        raise ValueError("The prompt must end with a user message")

    return [
        {"role": message["role"], "content": _content_to_text(message["content"])}
        for message in prompt_messages
    ]


def response_token_mask_from_offsets(
    offsets: list[tuple[int, int]],
    prompt_char_length: int,
    full_char_length: int,
    sampled_byte_ranges: list[tuple[int, int]] | None = None,
    response_text: str | None = None,
) -> torch.Tensor:
    """Select tokenizer tokens whose source spans belong to the response."""
    if full_char_length <= prompt_char_length:
        raise ValueError("The response is empty")
    selected = []
    for start, end in offsets:
        in_response = end > prompt_char_length and start < full_char_length
        if not in_response:
            selected.append(False)
            continue
        if sampled_byte_ranges is None:
            selected.append(True)
            continue

        # Convert the character offsets reported by the fast tokenizer into
        # byte offsets within the response. This lets us exclude forced tool
        # output tokens without assuming that the two tokenizers split text
        # the same way.
        clipped_start = max(start, prompt_char_length)
        clipped_end = min(end, full_char_length)
        if response_text is None:
            raise ValueError("response_text is required when filtering sampled byte ranges")
        local_start = clipped_start - prompt_char_length
        local_end = clipped_end - prompt_char_length
        selected.append(
            _byte_range_overlaps(
                len(response_text[:local_start].encode("utf-8")),
                len(response_text[:local_end].encode("utf-8")),
                sampled_byte_ranges,
            )
        )
    mask = torch.tensor(selected, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("Could not align teacher tokens to the student response")
    return mask


def _byte_range_overlaps(
    start: int,
    end: int,
    sampled_byte_ranges: list[tuple[int, int]],
) -> bool:
    return any(start < sampled_end and sampled_start < end for sampled_start, sampled_end in sampled_byte_ranges)


def student_sequence_logprobs(
    token_logprobs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Average student log-probability for each sequence, ignoring masked targets."""
    valid = targets >= 0
    token_logprobs = token_logprobs * valid
    counts = valid.sum(dim=1).clamp_min(1)
    return token_logprobs.sum(dim=1) / counts


def on_policy_advantages(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return centered sequence-level advantages for the on-policy KL gradient.

    For forward KL(p_student || p_teacher), the policy-gradient advantage is
    log p_student(y) - log p_teacher(y). Each sequence log-probability is
    normalized by its own token count so the two tokenizers contribute on a
    comparable per-token scale. Centering is a variance-reducing baseline and
    does not change the expected gradient.
    """
    student_logprobs = student_logprobs.to(dtype=torch.float32)
    teacher_logprobs = teacher_logprobs.to(
        device=student_logprobs.device,
        dtype=torch.float32,
    )
    valid = valid.to(device=student_logprobs.device, dtype=torch.bool)
    advantages = student_logprobs - teacher_logprobs
    if bool(valid.any()):
        advantages = advantages - advantages[valid].mean()
    else:
        advantages = torch.zeros_like(advantages)
    return advantages.masked_fill(~valid, 0.0)


def on_policy_kl_objective(
    token_logprobs: torch.Tensor,
    targets: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Return the unnormalized forward-KL policy-gradient objective."""
    valid = targets >= 0
    if not bool(valid.any()):
        return token_logprobs.sum() * 0.0
    per_token_advantages = advantages.to(token_logprobs.device).unsqueeze(1).expand_as(token_logprobs)
    return -(per_token_advantages.detach()[valid] * token_logprobs[valid]).sum()


@dataclass(frozen=True)
class TeacherScore:
    """Normalized teacher log-probability for one response."""

    logprob: float
    num_tokens: int


class QwenTeacher:
    """Score sampled student responses with a HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "sdpa",
        max_seq_len: int = 4096,
        local_files_only: bool = False,
    ):
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose either 4-bit or 8-bit teacher loading, not both")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "On-policy distillation requires the optional distill dependencies. "
                "Install them with `uv sync --extra gpu --extra distill`."
            ) from exc

        self.device = device
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if load_in_4bit or load_in_8bit:
            if device.type != "cuda":
                raise ValueError("Quantized teacher loading requires CUDA")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("BitsAndBytesConfig is unavailable; install a newer transformers") from exc
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = {"": str(device)}
        else:
            model_kwargs["dtype"] = {
                "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype]

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not (load_in_4bit or load_in_8bit):
            self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def score_response(
        self,
        prompt_messages: list[dict[str, str]],
        response: str,
        sampled_byte_ranges: list[tuple[int, int]] | None = None,
    ) -> TeacherScore | None:
        """Return mean teacher log-probability of the exact response text."""
        if not response:
            return None

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + response
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(input_ids) > self.max_seq_len:
            return None

        response_mask = response_token_mask_from_offsets(
            offsets,
            prompt_char_length=len(prompt_text),
            full_char_length=len(full_text),
            sampled_byte_ranges=sampled_byte_ranges,
            response_text=response,
        )
        response_mask = response_mask.to(self.device)
        if response_mask.sum().item() == 0:
            return None

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=ids, use_cache=False)
        logits = outputs.logits[0, :-1, :]
        labels = ids[0, 1:]
        shifted_mask = response_mask[1:]
        if not bool(shifted_mask.any()):
            return None

        token_logprobs = F.log_softmax(logits.float(), dim=-1).gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        response_logprobs = token_logprobs[shifted_mask]
        return TeacherScore(
            logprob=response_logprobs.mean().item(),
            num_tokens=response_logprobs.numel(),
        )
