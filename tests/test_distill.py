"""Tests for cross-tokenizer on-policy distillation helpers."""

import math

import torch

from nanochat.distill import (
    QwenTeacher,
    conversation_to_teacher_prompt,
    on_policy_advantages,
    on_policy_kl_objective,
    response_token_mask_from_offsets,
    student_sequence_logprobs,
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
