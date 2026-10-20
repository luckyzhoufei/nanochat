"""Tests for cross-tokenizer on-policy distillation helpers."""

import math

import torch

from nanochat.distill import (
    QwenTeacher,
    conversation_to_teacher_prompt,
    fit_prompt_to_context,
    on_policy_advantages,
    on_policy_kl_objective,
    quality_adjusted_scores,
    gsm8k_correctness_reward,
    mmlu_correctness_reward,
    response_token_mask_from_offsets,
    response_repetition_rate,
    reward_baseline_advantages,
    select_teacher_ranked_samples,
    student_sequence_logprobs,
    teacher_reward_advantages,
)


class FakeTeacherTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "PROMPT"

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert text == "PROMPTabc"
        return {
            "input_ids": [1, 2, 3, 4],
            "offset_mapping": [(0, 6), (6, 7), (7, 8), (8, 9)],
        }


class FakeTeacherModel:
    def __call__(self, input_ids, use_cache):
        assert input_ids.tolist() == [[1, 2, 3, 4]]
        assert use_cache is False

        class Output:
            logits = torch.zeros(1, 4, 5)

        return Output()


def test_conversation_to_teacher_prompt_replaces_reference_answer():
    conversation = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    prompt = conversation_to_teacher_prompt(conversation)
    assert prompt == [{"role": "user", "content": "Be concise.\n\nWhat is 2+2?"}]


def test_conversation_to_teacher_prompt_accepts_user_ending():
    conversation = {
        "messages": [
            {"role": "user", "content": "Tell me a joke"},
        ]
    }
    prompt = conversation_to_teacher_prompt(conversation)
    assert prompt == [{"role": "user", "content": "Tell me a joke"}]


def test_qwen_teacher_scores_only_response_tokens():
    teacher = QwenTeacher.__new__(QwenTeacher)
    teacher.device = torch.device("cpu")
    teacher.max_seq_len = 16
    teacher.tokenizer = FakeTeacherTokenizer()
    teacher.model = FakeTeacherModel()
    score = teacher.score_response([{"role": "user", "content": "x"}], "abc")
    assert score is not None
    assert score.num_tokens == 3
    assert torch.allclose(torch.tensor(score.logprob), torch.tensor(-math.log(5.0)))


def test_fit_prompt_to_context_reserves_generation_budget():
    prompt_ids = list(range(2049))
    fitted, max_new_tokens = fit_prompt_to_context(
        prompt_ids,
        sequence_len=2048,
        max_new_tokens=256,
    )
    assert len(fitted) == 1792
    assert fitted[-1] == 2048
    assert max_new_tokens == 256


def test_fit_prompt_to_context_keeps_short_prompt():
    prompt_ids = list(range(10))
    fitted, max_new_tokens = fit_prompt_to_context(
        prompt_ids,
        sequence_len=2048,
        max_new_tokens=256,
    )
    assert fitted == prompt_ids
    assert max_new_tokens == 256


def test_response_token_mask_from_offsets():
    offsets = [(0, 10), (10, 11), (11, 15), (15, 20)]
    mask = response_token_mask_from_offsets(
        offsets,
        prompt_char_length=10,
        full_char_length=20,
    )
    assert mask.tolist() == [False, True, True, True]


def test_response_token_mask_excludes_forced_byte_spans():
    response = "答案4"
    offsets = [(0, 10), (10, 12), (12, 13)]
    # Select only the two-byte token containing "4", not the Chinese character.
    mask = response_token_mask_from_offsets(
        offsets,
        prompt_char_length=10,
        full_char_length=13,
    sampled_byte_ranges=[(6, 7)],
        response_text=response,
    )
    assert mask.tolist() == [False, False, True]


def test_student_sequence_logprobs_ignore_masked_targets():
    token_logprobs = torch.tensor([
        [-1.0, -2.0, -3.0, -9.0],
        [-4.0, -5.0, -99.0, -99.0],
    ])
    targets = torch.tensor([
        [1, 1, 1, -1],
        [1, 1, -1, -1],
    ])
    result = student_sequence_logprobs(token_logprobs, targets)
    assert torch.allclose(result, torch.tensor([-2.0, -4.5]))


def test_on_policy_advantages_use_centered_student_minus_teacher():
    teacher = torch.tensor([-2.0, -1.0, -4.0])
    student = torch.tensor([-3.0, -2.0, -2.0])
    valid = torch.tensor([True, True, False])
    advantages = on_policy_advantages(teacher, student, valid)
    # Raw advantages are [-1, -1], centered to [0, 0]. Invalid rows are zeroed.
    assert torch.allclose(advantages, torch.zeros_like(advantages))

    teacher = torch.tensor([-2.0, -1.0, -4.0])
    student = torch.tensor([-4.0, -1.0, -2.0])
    advantages = on_policy_advantages(teacher, student, valid)
    # Raw advantages are [-2, 0], centered to [-1, 1].
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, 0.0]))


def test_teacher_reward_advantages_are_centered_and_masked():
    teacher = torch.tensor([-2.0, -1.0, -4.0])
    valid = torch.tensor([True, True, False])
    advantages = teacher_reward_advantages(teacher, valid)
    assert torch.allclose(advantages, torch.tensor([-0.5, 0.5, 0.0]))


def test_reward_baseline_advantages_use_all_valid_samples_as_baseline():
    rewards = torch.tensor([1.0, 0.5, -0.5, 5.0])
    selected = torch.tensor([True, True, False, False])
    baseline_valid = torch.tensor([True, True, True, False])
    advantages = reward_baseline_advantages(rewards, selected, baseline_valid)
    expected = torch.tensor([2.0 / 3.0, 1.0 / 6.0, 0.0, 0.0])
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_select_teacher_ranked_samples_keeps_highest_valid_scores():
    teacher_logprobs = torch.tensor([-3.0, -1.0, -2.0, -0.5])
    valid = torch.tensor([True, True, False, True])
    selected = select_teacher_ranked_samples(teacher_logprobs, valid, keep_top_k=2)
    assert selected.tolist() == [False, True, False, True]


def test_select_teacher_ranked_samples_handles_fewer_valid_than_k():
    teacher_logprobs = torch.tensor([-3.0, -1.0])
    valid = torch.tensor([False, True])
    selected = select_teacher_ranked_samples(teacher_logprobs, valid, keep_top_k=2)
    assert selected.tolist() == [False, True]


def test_task_first_selection_prioritizes_correct_answers():
    quality_scores = torch.tensor([0.5, -1.0, -0.5])
    valid = torch.tensor([True, True, True])
    task_rewards = torch.tensor([0.0, 1.0, 1.0])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="task_first",
    )
    assert selected.tolist() == [False, True, True]


def test_task_first_prefers_higher_partial_reward_tier():
    quality_scores = torch.tensor([0.5, -1.0, -0.5])
    valid = torch.tensor([True, True, True])
    task_rewards = torch.tensor([0.0, 0.1, 0.1])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="task_first",
    )
    assert selected.tolist() == [False, False, True]


def test_task_all_keeps_every_correct_answer():
    quality_scores = torch.tensor([5.0, -1.0, -2.0])
    valid = torch.tensor([True, True, True])
    task_rewards = torch.tensor([0.0, 1.0, 1.0])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="task_all",
        task_kind="mmlu",
    )
    assert selected.tolist() == [False, True, True]

def test_task_first_skips_mmlu_when_no_correct_answer_exists():
    quality_scores = torch.tensor([0.5, 0.4])
    valid = torch.tensor([True, True])
    task_rewards = torch.tensor([0.0, 0.0])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="task_first",
        task_kind="mmlu",
    )
    assert selected.tolist() == [False, False]


def test_task_first_skips_when_no_positive_reward_exists():
    quality_scores = torch.tensor([0.5, 0.4])
    valid = torch.tensor([True, True])
    task_rewards = torch.tensor([0.0, 0.0])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="task_first",
        task_kind="other",
    )
    assert selected.tolist() == [False, False]


def test_correct_only_skips_when_no_correct_answer_exists():
    quality_scores = torch.tensor([0.5, 0.4])
    valid = torch.tensor([True, True])
    task_rewards = torch.tensor([0.0, 0.1])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=2,
        task_rewards=task_rewards,
        task_mode="correct_only",
    )
    assert selected.tolist() == [False, False]


def test_response_repetition_rate_detects_collapse():
    assert response_repetition_rate([1] * 100) > 0.95
    assert response_repetition_rate([1, 2, 3, 4, 5]) == 0.0


def test_quality_adjusted_scores_penalize_repetition_and_truncation():
    teacher = torch.tensor([-0.1, -1.0])
    lengths = torch.tensor([512.0, 100.0])
    repetition = torch.tensor([1.0, 0.0])
    truncated = torch.tensor([1.0, 0.0])
    scores = quality_adjusted_scores(
        teacher,
        lengths,
        repetition,
        truncated,
        length_penalty=0.0002,
        repetition_penalty=2.0,
        truncation_penalty=5.0,
    )
    assert torch.allclose(scores, torch.tensor([-7.2024, -1.02]))


def test_mmlu_correctness_reward_uses_last_standalone_letter():
    conversation = {
        "letters": ("A", "B", "C", "D"),
        "messages": [{"role": "assistant", "content": "C"}],
    }
    assert mmlu_correctness_reward(conversation, "The answer is C.") == 1.0
    assert mmlu_correctness_reward(conversation, "A is tempting, but C") == 1.0
    assert mmlu_correctness_reward(conversation, "The answer is A.") == 0.0


def test_gsm8k_correctness_reward_compares_final_number():
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "work\n#### 42"}],
            }
        ]
    }
    assert gsm8k_correctness_reward(conversation, "work\n#### 42") == 1.0
    assert gsm8k_correctness_reward(conversation, "work\n#### 42\n") == 1.0
    assert gsm8k_correctness_reward(conversation, "work\n#### 41") == 0.1
    assert gsm8k_correctness_reward(conversation, "The answer is 42") == 0.0
    assert gsm8k_correctness_reward(conversation, "work\n#### 42\nThe answer is: 42") == 0.0


def test_on_policy_kl_objective_has_expected_gradient():
    token_logprobs = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]],
        requires_grad=True,
    )
    targets = torch.ones_like(token_logprobs, dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0])
    objective = on_policy_kl_objective(token_logprobs, targets, advantages)
    assert torch.allclose(objective, torch.tensor(-4.0))
    objective.backward()
    assert torch.allclose(
        token_logprobs.grad,
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )


def test_chunked_kl_objective_matches_full_batch():
    token_logprobs = torch.tensor([
        [-1.0, -2.0],
        [-3.0, -4.0],
        [-5.0, -6.0],
    ])
    targets = torch.ones_like(token_logprobs, dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0, 0.5])
    full = on_policy_kl_objective(token_logprobs, targets, advantages)
    chunked = (
        on_policy_kl_objective(token_logprobs[:1], targets[:1], advantages[:1])
        + on_policy_kl_objective(token_logprobs[1:], targets[1:], advantages[1:])
    )
    assert torch.allclose(full, chunked)


def test_correct_all_keeps_every_fully_correct_answer():
    quality_scores = torch.tensor([5.0, -1.0, -2.0])
    valid = torch.tensor([True, True, True])
    task_rewards = torch.tensor([0.0, 1.0, 1.0])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="correct_all",
        task_kind="gsm8k",
    )
    assert selected.tolist() == [False, True, True]


def test_correct_all_skips_format_only_answers():
    quality_scores = torch.tensor([5.0, 4.0])
    valid = torch.tensor([True, True])
    task_rewards = torch.tensor([0.1, 0.1])
    selected = select_teacher_ranked_samples(
        quality_scores,
        valid,
        keep_top_k=1,
        task_rewards=task_rewards,
        task_mode="correct_all",
        task_kind="gsm8k",
    )
    assert selected.tolist() == [False, False]
