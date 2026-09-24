"""
On-policy distillation from a HuggingFace teacher into a nanochat model.

The student samples its own responses. The teacher scores those exact decoded
responses, and the student minimizes the forward-KL policy-gradient objective.
Because nanochat and Qwen use different tokenizers, teacher and student
log-probabilities are normalized per token before being compared.

Single GPU:

python -m scripts.chat_distill --teacher-load-in-4bit

Multi-GPU:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_distill -- --run=distill
"""

import argparse
import itertools
import math
import os
import time

import torch
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.distill import (
    QwenTeacher,
    conversation_to_teacher_prompt,
    on_policy_advantages,
    on_policy_kl_objective,
    student_sequence_logprobs,
)
from nanochat.engine import Engine
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


parser = argparse.ArgumentParser(description="On-policy distillation for nanochat")
# Logging and runtime.
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Student checkpoint.
parser.add_argument("--source", type=str, default="sft", choices=["sft", "rl"], help="student checkpoint family")
parser.add_argument("--model-tag", type=str, default=None, help="student model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="student model step to load")
# Teacher.
parser.add_argument(
    "--teacher-model",
    type=str,
    default="Qwen/Qwen3.8-27B",
    help="HuggingFace teacher model",
)
parser.add_argument(
    "--teacher-device",
    type=str,
    default="",
    help="teacher device (empty = same device as the student)",
)
parser.add_argument(
    "--teacher-dtype",
    type=str,
    default="auto",
    choices=["auto", "bfloat16", "float16", "float32"],
    help="teacher compute dtype",
)
parser.add_argument("--teacher-load-in-4bit", action="store_true", help="load the teacher with NF4 quantization")
parser.add_argument("--teacher-load-in-8bit", action="store_true", help="load the teacher with int8 quantization")
parser.add_argument(
    "--teacher-attn-implementation",
    type=str,
    default="sdpa",
    choices=["sdpa", "flash_attention_2", "eager"],
    help="teacher attention implementation",
)
parser.add_argument(
    "--teacher-max-seq-len",
    type=int,
    default=4096,
    help="skip samples above this teacher sequence length",
)
parser.add_argument("--teacher-local-files-only", action="store_true", help="do not download teacher files")
# Training horizon and rollout sizes.
parser.add_argument("--num-steps", type=int, default=100, help="number of optimizer steps")
parser.add_argument("--prompts-per-step", type=int, default=1, help="prompts sampled per rank per optimizer step")
parser.add_argument("--samples-per-prompt", type=int, default=8, help="student responses sampled per prompt")
parser.add_argument("--device-batch-size", type=int, default=8, help="generation batch size per forward pass")
# Generation. These defaults keep the data exactly on-policy.
parser.add_argument("--max-new-tokens", type=int, default=256, help="maximum response tokens")
parser.add_argument("--temperature", type=float, default=1.0, help="student sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="student top-k sampling (0 = full distribution)")
# Optimization.
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters")
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="weight decay for embedding/unembedding parameters",
)
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as a fraction of the base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.03, help="fraction of steps used for linear LR warmup")
# Data mixture.
parser.add_argument("--mmlu-epochs", type=int, default=1, help="MMLU copies in the prompt mixture")
parser.add_argument("--gsm8k-epochs", type=int, default=1, help="GSM8K copies in the prompt mixture")
# Checkpointing.
parser.add_argument("--save-every", type=int, default=50, help="save a checkpoint every N steps")
parser.add_argument("--seed", type=int, default=42, help="sampling seed")
args = parser.parse_args()
user_config = vars(args).copy()

if args.num_steps <= 0:
    parser.error("--num-steps must be positive")
if args.prompts_per_step <= 0:
    parser.error("--prompts-per-step must be positive")
if args.samples_per_prompt <= 1:
    parser.error("--samples-per-prompt must be at least 2 to estimate a within-prompt baseline")
if args.device_batch_size <= 0:
    parser.error("--device-batch-size must be positive")
if args.max_new_tokens <= 0:
    parser.error("--max-new-tokens must be positive")
if args.teacher_max_seq_len <= 0:
    parser.error("--teacher-max-seq-len must be positive")
if args.save_every <= 0:
    parser.error("--save-every must be positive")
if not 0.0 <= args.warmup_ratio < 1.0:
    parser.error("--warmup-ratio must be in [0, 1)")
if args.temperature != 1.0 or args.top_k != 0:
    print0("WARNING: temperature=1 and top-k=0 are required for strictly on-policy rollouts.")

# -----------------------------------------------------------------------------
# Compute, student, tokenizer, and teacher setup.

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-distill",
    name=args.run,
    config=user_config,
)

student, tokenizer, _ = load_model(
    args.source,
    device,
    phase="train",
    model_tag=args.model_tag,
    step=args.model_step,
)
engine = Engine(student, tokenizer)

teacher_device = device if args.teacher_device == "" else torch.device(args.teacher_device)
teacher = QwenTeacher(
    model_name=args.teacher_model,
    device=teacher_device,
    dtype=args.teacher_dtype,
    load_in_4bit=args.teacher_load_in_4bit,
    load_in_8bit=args.teacher_load_in_8bit,
    attn_implementation=args.teacher_attn_implementation,
    max_seq_len=args.teacher_max_seq_len,
    local_files_only=args.teacher_local_files_only,
)
print0(
    f"Loaded teacher {args.teacher_model} on {teacher_device} "
    f"(4bit={args.teacher_load_in_4bit}, 8bit={args.teacher_load_in_8bit})"
)

# -----------------------------------------------------------------------------
# Prompt mixture. The final assistant target is used only to identify the turn
# to replace; the student's own response is what the teacher scores.

train_tasks = [
    SmolTalk(split="train"),
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
]
train_dataset = TaskMixture(train_tasks)
print0(f"Prompt mixture: {len(train_dataset):,} conversations")


def prompt_iterator():
    indices = itertools.cycle(range(ddp_rank, len(train_dataset), ddp_world_size))
    for index in indices:
        yield index, train_dataset[index]


prompt_iter = prompt_iterator()


def next_seed(step: int, prompt_idx: int, sampling_idx: int) -> int:
    return (
        args.seed
        + step * 1_000_003
        + prompt_idx * 10_007
        + sampling_idx * 101
        + ddp_rank * 1_009
    ) % (2**31 - 1)


def sampled_response_byte_ranges(
    response_ids: list[int],
    sampled_mask: list[int],
) -> list[tuple[int, int]]:
    """Return byte spans produced by policy actions, excluding forced tool output."""
    offset = 0
    ranges = []
    for token_id, is_sampled in zip(response_ids, sampled_mask):
        token_bytes = tokenizer.decode_single_token_bytes(token_id)
        if is_sampled:
            ranges.append((offset, offset + len(token_bytes)))
        offset += len(token_bytes)
    return ranges


@torch.no_grad()
def sample_and_score(conversation: dict, step: int, prompt_idx: int):
    """Sample student responses and score their exact text with the teacher."""
    prompt_ids = tokenizer.render_for_completion(conversation)
    available_tokens = student.config.sequence_len - len(prompt_ids)
    if available_tokens <= 0:
        raise ValueError(
            f"Prompt length {len(prompt_ids)} leaves no room for generation "
            f"(student sequence length {student.config.sequence_len})"
        )
    max_new_tokens = min(args.max_new_tokens, available_tokens)
    teacher_prompt = conversation_to_teacher_prompt(conversation)

    sequences = []
    sampled_masks = []
    teacher_scores = []
    student.eval()
    num_sampling_passes = math.ceil(args.samples_per_prompt / args.device_batch_size)
    for sampling_idx in range(num_sampling_passes):
        batch_size = min(
            args.device_batch_size,
            args.samples_per_prompt - sampling_idx * args.device_batch_size,
        )
        generated, masks = engine.generate_batch(
            prompt_ids,
            num_samples=batch_size,
            max_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=next_seed(step, prompt_idx, sampling_idx),
        )
        sequences.extend(generated)
        sampled_masks.extend(masks)

    for sequence, sampled_mask in zip(sequences, sampled_masks):
        response_ids = sequence[len(prompt_ids):]
        response_mask = sampled_mask[len(prompt_ids):]
        response = tokenizer.decode(response_ids)
        sampled_ranges = sampled_response_byte_ranges(response_ids, response_mask)
        if not sampled_ranges:
            teacher_scores.append(None)
        else:
            teacher_scores.append(
                teacher.score_response(
                    teacher_prompt,
                    response,
                    sampled_byte_ranges=sampled_ranges,
                )
            )

    return sequences, sampled_masks, teacher_scores


def build_student_batch(
    sequences: list[list[int]],
    sampled_masks: list[list[int]],
    teacher_scores: list,
):
    """Pad rollouts and build teacher/student sequence-level scores."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    max_length = max(len(sequence) for sequence in sequences)
    padded_sequences = [
        sequence + [assistant_end] * (max_length - len(sequence))
        for sequence in sequences
    ]
    padded_masks = [
        mask + [0] * (max_length - len(mask))
        for mask in sampled_masks
    ]
    ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    masks = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[masks[:, 1:] == 0] = -1

    teacher_valid = torch.tensor(
        [score is not None for score in teacher_scores],
        dtype=torch.bool,
        device=device,
    )
    teacher_logprobs = torch.tensor(
        [score.logprob if score is not None else 0.0 for score in teacher_scores],
        dtype=torch.float32,
        device=device,
    )
    return inputs, targets, teacher_logprobs, teacher_valid


# -----------------------------------------------------------------------------
# Optimizer and schedule.

optimizer = student.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

warmup_steps = max(1, int(args.num_steps * args.warmup_ratio))


def get_lr_multiplier(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, args.num_steps - warmup_steps)
    return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)


def zero_loss(model: torch.nn.Module) -> torch.Tensor:
    """Create a zero loss with a gradient path to every trainable parameter."""
    loss = None
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        term = parameter.sum() * 0.0
        loss = term if loss is None else loss + term
    if loss is None:
        raise RuntimeError("Student has no trainable parameters")
    return loss


# -----------------------------------------------------------------------------
# Training loop.

base_dir = get_base_dir()
depth = student.config.n_layer
output_dirname = args.model_tag if args.model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "chatdistill_checkpoints", output_dirname)
total_training_time = 0.0
teacher_logprob_sum = 0.0
student_logprob_sum = 0.0
valid_sequence_count = 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

for step in range(args.num_steps):
    synchronize()
    step_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    prompt_rewards = []
    prompt_losses = []
    prompt_response_lengths = []
    for prompt_idx in range(args.prompts_per_step):
        _, conversation = next(prompt_iter)
        sequences, sampled_masks, teacher_scores = sample_and_score(
            conversation,
            step,
            prompt_idx,
        )
        inputs, targets, teacher_logprobs, teacher_valid = build_student_batch(
            sequences,
            sampled_masks,
            teacher_scores,
        )
        # Invalid teacher rows must not contribute to the student objective.
        targets = targets.masked_fill(~teacher_valid.unsqueeze(1), -1)

        student.train()
        student_token_logprobs = -student(
            inputs,
            targets,
            loss_reduction="none",
        ).view_as(inputs)
        student_logprobs = student_sequence_logprobs(student_token_logprobs, targets)
        advantages = on_policy_advantages(
            teacher_logprobs,
            student_logprobs.detach(),
            teacher_valid,
        )

        num_valid_tokens = (targets >= 0).sum().clamp_min(1)
        num_valid_sequences = teacher_valid.sum()
        if bool(teacher_valid.any()):
            prompt_teacher_logprob = teacher_logprobs[teacher_valid].mean().item()
            prompt_student_logprob = student_logprobs.detach()[teacher_valid].mean().item()
            prompt_advantages = advantages[teacher_valid]
            teacher_logprob_sum += prompt_teacher_logprob * int(num_valid_sequences.item())
            student_logprob_sum += prompt_student_logprob * int(num_valid_sequences.item())
            valid_sequence_count += int(num_valid_sequences.item())
            prompt_rewards.append(prompt_advantages.mean().item())

        # Token log-probs and sequence-level advantages can be large. The
        # global token normalizer keeps updates comparable across response lengths.
        loss = on_policy_kl_objective(student_token_logprobs, targets, advantages)
        loss = loss / num_valid_tokens / args.prompts_per_step
        if bool(teacher_valid.any()):
            loss.backward()
            prompt_losses.append(loss.detach().item())
        else:
            zero_loss(student).backward()

        prompt_response_lengths.extend(len(sequence) for sequence in sequences)

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    step_time = time.time() - step_start
    total_training_time += step_time

    mean_teacher_logprob = teacher_logprob_sum / max(1, valid_sequence_count)
    mean_student_logprob = student_logprob_sum / max(1, valid_sequence_count)
    mean_reward = sum(prompt_rewards) / max(1, len(prompt_rewards))
    mean_loss = sum(prompt_losses) / max(1, len(prompt_losses))
    mean_response_length = sum(prompt_response_lengths) / max(1, len(prompt_response_lengths))
    print0(
        f"step {step + 1:05d}/{args.num_steps} | loss: {mean_loss:.5f} | "
        f"teacher_logp: {mean_teacher_logprob:.4f} | student_logp: {mean_student_logprob:.4f} | "
        f"advantage: {mean_reward:.5f} | response_len: {mean_response_length:.1f} | "
        f"lrm: {lrm:.3f} | dt: {step_time * 1000:.0f}ms"
    )
    wandb_run.log({
        "step": step + 1,
        "train/loss": mean_loss,
        "train/teacher_logprob": mean_teacher_logprob,
        "train/student_logprob": mean_student_logprob,
        "train/advantage": mean_reward,
        "train/response_length": mean_response_length,
        "train/lrm": lrm,
        "train/dt": step_time,
    })

    # Reset accumulators after each optimizer step.
    teacher_logprob_sum = 0.0
    student_logprob_sum = 0.0
    valid_sequence_count = 0

    if (step > 0 and step % args.save_every == 0) or step == args.num_steps - 1:
        save_checkpoint(
            checkpoint_dir,
            step,
            student.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "model_config": student.config.__dict__,
                "teacher_model": args.teacher_model,
                "user_config": user_config,
            },
            rank=ddp_rank,
        )
        print0(f"Saved checkpoint to {checkpoint_dir}")

print0(f"Total training time: {total_training_time / 60:.2f}m")
wandb_run.finish()
compute_cleanup()
