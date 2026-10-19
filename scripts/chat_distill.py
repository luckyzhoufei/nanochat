"""
On-policy distillation from a HuggingFace teacher into a nanochat model.

The student samples its own responses. The teacher scores those exact decoded
responses, and the student minimizes the forward-KL policy-gradient objective.
Because nanochat and Qwen use different tokenizers, teacher and student
log-probabilities are normalized per token before being compared.

Single GPU:

python -m scripts.chat_distill --teacher-load-in-4bit

Two GPUs (recommended entry point):

bash runs/distill_2gpu.sh

Any other GPU count:

torchrun --standalone --nproc_per_node=N -m scripts.chat_distill -- --run=distill
"""

import argparse
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
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
    completion_messages_to_teacher_prompt,
    fit_prompt_to_context,
    on_policy_advantages,
    on_policy_kl_objective,
    quality_adjusted_scores,
    reward_baseline_advantages,
    response_repetition_rate,
    task_correctness_reward,
    select_teacher_ranked_samples,
    student_sequence_logprobs,
    teacher_reward_advantages,
)
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


parser = argparse.ArgumentParser(description="On-policy distillation for nanochat")
# Logging and runtime.
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--nccl-timeout-minutes", type=int, default=30, help="distributed collective timeout")
# Student checkpoint.
parser.add_argument("--source", type=str, default="sft", choices=["sft", "rl"], help="student checkpoint family")
parser.add_argument("--model-tag", type=str, default=None, help="student model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="student model step to load")
parser.add_argument("--output-tag", type=str, default=None, help="checkpoint tag to save; defaults to --model-tag")
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
parser.add_argument(
    "--teacher-load-in-4bit",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="load the teacher with NF4 quantization (default: enabled)",
)
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
parser.add_argument("--loss-batch-size", type=int, default=2, help="rollouts per student forward/backward chunk")
parser.add_argument(
    "--objective",
    choices=["rejection", "pg", "teacher_reward"],
    default="rejection",
    help="teacher-ranked SFT, forward-KL policy gradient, or teacher-likelihood reward",
)
parser.add_argument("--keep-top-k", type=int, default=2, help="teacher-ranked samples kept for rejection objective")
parser.add_argument(
    "--selection-mode",
    choices=["teacher", "task_first", "task_all", "correct_only", "correct_all"],
    default="task_first",
    help="task_first keeps all correct, otherwise teacher top-k among reward-positive samples",
)
parser.add_argument("--length-penalty", type=float, default=0.0002, help="quality penalty per generated token")
parser.add_argument("--repetition-penalty", type=float, default=2.0, help="quality penalty multiplier for repeated n-grams")
parser.add_argument("--truncation-penalty", type=float, default=5.0, help="quality penalty for unfinished responses")
parser.add_argument(
    "--max-repeat-ngram-rate",
    type=float,
    default=0.8,
    help="reject responses whose repeated 3-gram fraction exceeds this value",
)
parser.add_argument("--gsm8k-reward-weight", type=float, default=1.0, help="reward for the correct GSM8K answer")
parser.add_argument(
    "--gsm8k-format-reward",
    type=float,
    default=0.1,
    help="reward for a GSM8K response containing #### but an incorrect answer",
)
# Generation. These defaults keep the data exactly on-policy.
parser.add_argument("--max-new-tokens", type=int, default=256, help="maximum response tokens")
parser.add_argument("--temperature", type=float, default=1.0, help="student sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="student top-k sampling (0 = full distribution)")
parser.add_argument(
    "--eos-on-max-tokens",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="train an assistant_end target even when generation hits max-new-tokens",
)
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
# Checkpointing.
parser.add_argument("--save-every", type=int, default=50, help="save a checkpoint every N steps")
parser.add_argument("--log-samples-every", type=int, default=50, help="print generated samples every N steps (0 = disable)")
parser.add_argument("--log-samples-count", type=int, default=2, help="number of student responses to print")
parser.add_argument("--log-samples-chars", type=int, default=1000, help="maximum characters per printed prompt/response")
parser.add_argument("--seed", type=int, default=42, help="sampling seed")
args = parser.parse_args()

if args.num_steps <= 0:
    parser.error("--num-steps must be positive")
if args.nccl_timeout_minutes <= 0:
    parser.error("--nccl-timeout-minutes must be positive")
if args.prompts_per_step <= 0:
    parser.error("--prompts-per-step must be positive")
if args.samples_per_prompt <= 1:
    parser.error("--samples-per-prompt must be at least 2 to estimate a within-prompt baseline")
if args.device_batch_size <= 0:
    parser.error("--device-batch-size must be positive")
if args.loss_batch_size <= 0:
    parser.error("--loss-batch-size must be positive")
if args.keep_top_k <= 0:
    parser.error("--keep-top-k must be positive")
if args.length_penalty < 0 or args.repetition_penalty < 0 or args.truncation_penalty < 0:
    parser.error("quality penalties must be non-negative")
if not 0.0 <= args.max_repeat_ngram_rate <= 1.0:
    parser.error("--max-repeat-ngram-rate must be in [0, 1]")
if args.gsm8k_reward_weight < 0:
    parser.error("task reward and logit loss weights must be non-negative")
if not 0.0 <= args.gsm8k_format_reward <= 1.0:
    parser.error("--gsm8k-format-reward must be in [0, 1]")
if args.max_new_tokens <= 0:
    parser.error("--max-new-tokens must be positive")
if args.teacher_max_seq_len <= 0:
    parser.error("--teacher-max-seq-len must be positive")
if args.save_every <= 0:
    parser.error("--save-every must be positive")
if args.log_samples_every < 0:
    parser.error("--log-samples-every must be non-negative")
if args.log_samples_count <= 0:
    parser.error("--log-samples-count must be positive")
if args.log_samples_chars <= 0:
    parser.error("--log-samples-chars must be positive")
if not 0.0 <= args.warmup_ratio < 1.0:
    parser.error("--warmup-ratio must be in [0, 1)")
if args.temperature != 1.0 or args.top_k != 0:
    print0("WARNING: temperature=1 and top-k=0 are required for strictly on-policy rollouts.")
if args.teacher_load_in_4bit is False and args.teacher_load_in_8bit is False:
    print0(
        "WARNING: the teacher is loaded in full precision on every rank. "
        "Use --teacher-load-in-4bit for a 27B teacher."
    )

# -----------------------------------------------------------------------------
# Compute, student, tokenizer, and teacher setup.

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
    device_type,
    timeout_minutes=args.nccl_timeout_minutes,
)
master_process = ddp_rank == 0
if args.teacher_load_in_8bit:
    args.teacher_load_in_4bit = False
if device_type != "cuda" and args.teacher_load_in_4bit:
    print0("Disabling default 4-bit teacher loading because CUDA is not available.")
    args.teacher_load_in_4bit = False
user_config = vars(args).copy()

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
teacher = None
for load_rank in range(ddp_world_size):
    if ddp_rank == load_rank:
        print(
            f"[rank {ddp_rank}/{ddp_world_size}] loading teacher "
            f"{args.teacher_model} on {teacher_device}",
            flush=True,
        )
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
        print(f"[rank {ddp_rank}/{ddp_world_size}] teacher loaded", flush=True)
    # Loading two 27B copies concurrently can exhaust host RAM and make one
    # rank arrive at the first collective long before the other.
    if ddp:
        dist.barrier()
assert teacher is not None

# -----------------------------------------------------------------------------
# Prompt mixture. The final assistant target is used only to identify the turn
# to replace; the student's own response is what the teacher scores.

train_dataset = GSM8K(subset="main", split="train")
print0(f"Prompt dataset: GSM8K {len(train_dataset):,} conversations")


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
    prompt_ids, completion_messages = tokenizer.render_for_completion(
        conversation,
        return_messages=True,
    )
    prompt_ids, max_new_tokens = fit_prompt_to_context(
        prompt_ids,
        sequence_len=student.config.sequence_len,
        max_new_tokens=args.max_new_tokens,
    )
    teacher_prompt = completion_messages_to_teacher_prompt(completion_messages)

    sequences = []
    sampled_masks = []
    termination_tokens = []
    teacher_scores = []
    responses = []
    student.eval()
    num_sampling_passes = math.ceil(args.samples_per_prompt / args.device_batch_size)
    for sampling_idx in range(num_sampling_passes):
        batch_size = min(
            args.device_batch_size,
            args.samples_per_prompt - sampling_idx * args.device_batch_size,
        )
        generated, masks, terminations = engine.generate_batch(
            prompt_ids,
            num_samples=batch_size,
            max_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=next_seed(step, prompt_idx, sampling_idx),
            return_termination=True,
        )
        sequences.extend(generated)
        sampled_masks.extend(masks)
        termination_tokens.extend(terminations)

    for sequence, sampled_mask in zip(sequences, sampled_masks):
        response_ids = sequence[len(prompt_ids):]
        response_mask = sampled_mask[len(prompt_ids):]
        response = tokenizer.decode(response_ids)
        responses.append(response)
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

    if device_type == "cuda":
        torch.cuda.empty_cache()
    prompt_text = tokenizer.decode(prompt_ids)
    return (
        sequences,
        sampled_masks,
        termination_tokens,
        teacher_scores,
        responses,
        prompt_text,
        len(prompt_ids),
    )


def build_student_batch(
    sequences: list[list[int]],
    sampled_masks: list[list[int]],
    termination_tokens: list[int | None],
    teacher_scores: list,
    eos_on_max_tokens: bool,
):
    """Pad rollouts and build teacher/student sequence-level scores."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    sequences = [sequence.copy() for sequence in sequences]
    sampled_masks = [mask.copy() for mask in sampled_masks]
    for i, terminal_token in enumerate(termination_tokens):
        if terminal_token is not None:
            sequences[i].append(terminal_token)
            sampled_masks[i].append(1)
        elif eos_on_max_tokens:
            # A truncated trajectory has no observed stop token. Supervise the
            # stop position explicitly so the student learns to terminate.
            sequences[i].append(assistant_end)
            sampled_masks[i].append(1)
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

    raw_teacher_valid = torch.tensor(
        [score is not None for score in teacher_scores],
        dtype=torch.bool,
        device=device,
    )
    teacher_logprobs = torch.tensor(
        [score.logprob if score is not None else 0.0 for score in teacher_scores],
        dtype=torch.float32,
        device=device,
    )
    # A teacher score is only useful if the same response also has at least
    # one student target after applying Engine's sampled/forced-token mask.
    student_trainable = (targets >= 0).any(dim=1)
    teacher_valid = raw_teacher_valid & student_trainable
    return inputs, targets, teacher_logprobs, teacher_valid, raw_teacher_valid


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


def reduce_sum(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def reduce_mean(total: float, count: float) -> float:
    total = reduce_sum(total)
    count = reduce_sum(count)
    return total / count if count > 0 else float("nan")


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} chars omitted]"


# -----------------------------------------------------------------------------
# Training loop.

base_dir = get_base_dir()
depth = student.config.n_layer
output_dirname = args.output_tag or args.model_tag or f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "chatdistill_checkpoints", output_dirname)
total_training_time = 0.0
teacher_logprob_sum = 0.0
student_logprob_sum = 0.0
valid_sequence_count = 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# Do not let the first rank enter the optimizer collectives while another rank
# is still loading the model, tokenizer, teacher, or prompt datasets.
if ddp:
    dist.barrier()

for step in range(args.num_steps):
    synchronize()
    step_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    prompt_rewards = []
    prompt_advantage_stds = []
    prompt_teacher_score_stds = []
    prompt_losses = []
    prompt_response_lengths = []
    step_responses = []
    step_selected_responses = []
    step_candidate_count = 0
    step_teacher_valid_count = 0
    step_raw_teacher_valid_count = 0
    step_quality_valid_count = 0
    step_metric_count = 0
    step_metric_token_count = 0
    step_terminated_count = 0
    step_skipped_prompt_count = 0
    step_task_reward_sum = 0.0
    step_task_reward_count = 0
    for prompt_idx in range(args.prompts_per_step):
        # Both ranks consume the same prompt but sample different responses.
        # This keeps teacher sequence lengths and per-step work closer together.
        dataset_index = (step * args.prompts_per_step + prompt_idx) % len(train_dataset)
        conversation = train_dataset[dataset_index]
        (
            sequences,
            sampled_masks,
            termination_tokens,
            teacher_scores,
            responses,
            prompt_text,
            prompt_token_length,
        ) = sample_and_score(
            conversation,
            step,
            prompt_idx,
        )
        step_responses.extend(responses)
        step_terminated_count += sum(token is not None for token in termination_tokens)
        response_lengths = [
            len(sequence) - prompt_token_length
            for sequence in sequences
        ]
        task_kind = "other"
        raw_task_rewards = []
        task_rewards = []
        for response in responses:
            task_kind, reward = task_correctness_reward(
                conversation,
                response,
                gsm8k_format_reward=args.gsm8k_format_reward,
            )
            raw_task_rewards.append(reward)
            if task_kind == "gsm8k":
                reward *= args.gsm8k_reward_weight
            task_rewards.append(reward)
        raw_task_rewards = torch.tensor(raw_task_rewards, dtype=torch.float32, device=device)
        task_rewards = torch.tensor(task_rewards, dtype=torch.float32, device=device)
        repetition_rates = torch.tensor(
            [
                response_repetition_rate(sequence[prompt_token_length:])
                for sequence in sequences
            ],
            dtype=torch.float32,
            device=device,
        )
        truncated = torch.tensor(
            [token is None for token in termination_tokens],
            dtype=torch.float32,
            device=device,
        )
        inputs, targets, teacher_logprobs, teacher_valid, raw_teacher_valid = build_student_batch(
            sequences,
            sampled_masks,
            termination_tokens,
            teacher_scores,
            args.eos_on_max_tokens,
        )
        combined_teacher_logprobs = teacher_logprobs + task_rewards
        quality_scores = quality_adjusted_scores(
            combined_teacher_logprobs,
            torch.tensor(response_lengths, dtype=torch.float32, device=device),
            repetition_rates,
            truncated,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            truncation_penalty=args.truncation_penalty,
        )
        quality_valid = (
            (teacher_valid | (task_rewards > 0))
            & (repetition_rates <= args.max_repeat_ngram_rate)
        )
        if args.objective == "rejection":
            selected = select_teacher_ranked_samples(
                quality_scores,
                quality_valid,
                args.keep_top_k,
                task_rewards=raw_task_rewards,
                task_mode=args.selection_mode,
                task_kind=task_kind,
            )
            metric_mask = quality_valid & selected
        else:
            selected = quality_valid
            metric_mask = quality_valid

        should_log_samples = (
            args.log_samples_every > 0
            and step % args.log_samples_every == 0
        )
        if should_log_samples:
            print0(
                f"\n===== distillation sample | step {step + 1} | "
                f"prompt {prompt_idx + 1}/{args.prompts_per_step} ====="
            )
            print0(f"PROMPT:\n{clip_text(prompt_text, args.log_samples_chars)}")
            for response_idx in range(min(args.log_samples_count, len(responses))):
                teacher_score = teacher_scores[response_idx]
                score_text = "none" if teacher_score is None else f"{teacher_score.logprob:.5f}"
                selected_text = bool(metric_mask[response_idx].item())
                response_tokens = len(sequences[response_idx]) - prompt_token_length
                print0(
                    f"RESPONSE {response_idx + 1} | teacher_logp: {score_text} | "
                    f"quality_score: {quality_scores[response_idx].item():.5f} | "
                    f"task_reward: {task_rewards[response_idx].item():.1f} | "
                    f"repeat_rate: {repetition_rates[response_idx].item():.3f} | "
                    f"selected: {selected_text} | response_tokens: {response_tokens} | "
                    f"terminated: {termination_tokens[response_idx] is not None}"
                )
                print0(clip_text(responses[response_idx], args.log_samples_chars))


        # Invalid or unselected rows must not contribute to the student objective.
        targets = targets.masked_fill(~metric_mask.unsqueeze(1), -1)

        # The first pass is only needed to estimate the sequence baseline.
        # Running it without autograd keeps all rollout activations out of VRAM.
        student.eval()
        student_logprob_chunks = []
        for b0 in range(0, inputs.size(0), args.loss_batch_size):
            b1 = min(b0 + args.loss_batch_size, inputs.size(0))
            with torch.no_grad():
                token_logprobs = -student(
                    inputs[b0:b1],
                    targets[b0:b1],
                    loss_reduction="none",
                ).view_as(inputs[b0:b1])
                student_logprob_chunks.append(student_sequence_logprobs(token_logprobs, targets[b0:b1]))
        student_logprobs = torch.cat(student_logprob_chunks)
        if args.objective == "pg":
            advantages = on_policy_advantages(
                combined_teacher_logprobs,
                student_logprobs,
                metric_mask,
            )
        elif args.objective == "teacher_reward":
            advantages = teacher_reward_advantages(
                combined_teacher_logprobs,
                metric_mask,
            )
        else:
            advantages = reward_baseline_advantages(
                rewards=quality_scores.detach(),
                selection_mask=metric_mask,
                baseline_mask=quality_valid,
            )

        num_valid_tokens_raw = (targets >= 0).sum()
        num_valid_tokens = num_valid_tokens_raw.clamp_min(1)
        num_valid_sequences = metric_mask.sum()
        step_candidate_count += len(teacher_scores)
        step_teacher_valid_count += int(teacher_valid.sum().item())
        step_raw_teacher_valid_count += int(raw_teacher_valid.sum().item())
        step_quality_valid_count += int(quality_valid.sum().item())
        step_metric_count += int(metric_mask.sum().item())
        step_metric_token_count += int(num_valid_tokens_raw.item())
        step_task_reward_sum += task_rewards.sum().item()
        step_task_reward_count += task_rewards.numel()
        selected_indices = metric_mask.nonzero(as_tuple=False).squeeze(1)
        step_selected_responses.extend(responses[i] for i in selected_indices.tolist())
        valid_teacher_scores = teacher_logprobs[raw_teacher_valid]
        if valid_teacher_scores.numel() > 1:
            prompt_teacher_score_stds.append(valid_teacher_scores.std(unbiased=False).item())
        if bool(metric_mask.any()):
            prompt_teacher_logprob = teacher_logprobs[metric_mask].mean().item()
            prompt_objective_score = combined_teacher_logprobs[metric_mask].mean().item()
            prompt_student_logprob = student_logprobs[metric_mask].mean().item()
            teacher_logprob_sum += prompt_teacher_logprob * int(num_valid_sequences.item())
            student_logprob_sum += prompt_student_logprob * int(num_valid_sequences.item())
            valid_sequence_count += int(num_valid_sequences.item())
            if args.objective == "pg":
                prompt_score = prompt_student_logprob - prompt_objective_score
            else:
                # Advantages are centered within each prompt, so their mean is
                # zero by construction. Log the uncentered objective score.
                prompt_score = prompt_objective_score
            prompt_rewards.append(prompt_score)
            prompt_advantage_stds.append(advantages[metric_mask].std(unbiased=False).item())

        if bool(metric_mask.any()):
            # A second chunked pass accumulates gradients. This avoids holding
            # full-batch logits and activations for all sampled responses.
            student.train()
            prompt_loss = 0.0
            for b0 in range(0, inputs.size(0), args.loss_batch_size):
                b1 = min(b0 + args.loss_batch_size, inputs.size(0))
                token_logprobs = -student(
                    inputs[b0:b1],
                    targets[b0:b1],
                    loss_reduction="none",
                ).view_as(inputs[b0:b1])
                loss = on_policy_kl_objective(token_logprobs, targets[b0:b1], advantages[b0:b1])
                loss = loss / num_valid_tokens / args.prompts_per_step
                loss.backward()
                prompt_loss += loss.detach().item()
                del token_logprobs, loss
            prompt_losses.append(prompt_loss)
        else:
            step_skipped_prompt_count += 1
            zero_loss(student).backward()

        prompt_response_lengths.extend(response_lengths)

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    step_time = time.time() - step_start
    total_training_time += step_time

    mean_teacher_logprob = reduce_mean(teacher_logprob_sum, valid_sequence_count)
    mean_student_logprob = reduce_mean(student_logprob_sum, valid_sequence_count)
    mean_reward = reduce_mean(sum(prompt_rewards), len(prompt_rewards))
    mean_advantage_std = reduce_mean(
        sum(prompt_advantage_stds),
        len(prompt_advantage_stds),
    )
    mean_teacher_score_std = reduce_mean(
        sum(prompt_teacher_score_stds),
        len(prompt_teacher_score_stds),
    )
    mean_loss = reduce_mean(sum(prompt_losses), len(prompt_losses))
    step_candidate_count = int(reduce_sum(step_candidate_count))
    step_teacher_valid_count = int(reduce_sum(step_teacher_valid_count))
    step_raw_teacher_valid_count = int(reduce_sum(step_raw_teacher_valid_count))
    step_quality_valid_count = int(reduce_sum(step_quality_valid_count))
    step_metric_count = int(reduce_sum(step_metric_count))
    step_metric_token_count = int(reduce_sum(step_metric_token_count))
    step_terminated_count = int(reduce_sum(step_terminated_count))
    step_skipped_prompt_count = int(reduce_sum(step_skipped_prompt_count))
    mean_task_reward = reduce_mean(step_task_reward_sum, step_task_reward_count)
    mean_response_length = reduce_mean(
        sum(prompt_response_lengths),
        len(prompt_response_lengths),
    )

    if ddp:
        gathered_responses = [None] * ddp_world_size
        dist.all_gather_object(
            gathered_responses,
            (step_responses, step_selected_responses),
        )
        all_responses = [
            response
            for responses, _ in gathered_responses
            for response in responses
        ]
        all_selected_responses = [
            response
            for _, responses in gathered_responses
            for response in responses
        ]
    else:
        all_responses = step_responses
        all_selected_responses = step_selected_responses
    unique_responses = len(set(all_responses))
    unique_responses_ratio = unique_responses / max(1, len(all_responses))
    unique_selected_responses = len(set(all_selected_responses))
    unique_selected_ratio = unique_selected_responses / max(1, len(all_selected_responses))

    if len(all_responses) > 1 and unique_responses == 1:
        print0("WARNING: mode collapse detected: all sampled responses are identical")
    if mean_advantage_std == 0.0 and valid_sequence_count > 0:
        print0("WARNING: zero advantage variance; this step has no learning signal")
    score_label = {
        "pg": "kl_advantage",
        "teacher_reward": "objective_reward",
        "rejection": "objective_score",
    }[args.objective]
    print0(
        f"step {step + 1:05d}/{args.num_steps} | loss: {mean_loss:.5f} | "
        f"teacher_logp: {mean_teacher_logprob:.4f} | student_logp: {mean_student_logprob:.6e} | "
        f"{score_label}: {mean_reward:.5f} | adv_std: {mean_advantage_std:.5f} | "
        f"teacher_std: {mean_teacher_score_std:.5f} | "
        f"valid_seq: {step_metric_count}/{step_quality_valid_count}/{step_teacher_valid_count}/"
        f"{step_raw_teacher_valid_count}/{step_candidate_count} | "
        f"valid_tokens: {step_metric_token_count} | "
        f"skipped_prompts: {step_skipped_prompt_count} | "
        f"task_reward: {mean_task_reward:.5f} | "
        f"terminated: {step_terminated_count}/{step_candidate_count} | "
        f"unique: {unique_responses}/{len(all_responses)} ({unique_responses_ratio:.2f}) | "
        f"selected_unique: {unique_selected_responses}/{len(all_selected_responses)} | "
        f"response_len: {mean_response_length:.1f} | "
        f"lrm: {lrm:.3f} | dt: {step_time * 1000:.0f}ms"
    )
    wandb_run.log({
        "step": step + 1,
        "train/loss": mean_loss,
        "train/teacher_logprob": mean_teacher_logprob,
        "train/student_logprob": mean_student_logprob,
        "train/objective_score": mean_reward,
        "train/advantage_std": mean_advantage_std,
        "train/teacher_score_std": mean_teacher_score_std,
        "train/candidate_sequences": step_candidate_count,
        "train/teacher_valid_sequences": step_teacher_valid_count,
        "train/raw_teacher_valid_sequences": step_raw_teacher_valid_count,
        "train/quality_valid_sequences": step_quality_valid_count,
        "train/selected_sequences": step_metric_count,
        "train/valid_tokens": step_metric_token_count,
        "train/terminated_sequences": step_terminated_count,
        "train/skipped_prompts": step_skipped_prompt_count,
        "train/task_reward": mean_task_reward,
        "train/unique_responses": unique_responses,
        "train/unique_responses_ratio": unique_responses_ratio,
        "train/unique_selected_responses": unique_selected_responses,
        "train/unique_selected_ratio": unique_selected_ratio,
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
