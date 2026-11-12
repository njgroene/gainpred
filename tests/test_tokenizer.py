"""Tests for the character vocabulary and batch building."""

import pytest
import torch

from gainpred.formats import Addition, Unreadable, example_text, prompt_text, solve
from gainpred.tokenizer import (
    CHARACTERS,
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    decode,
    encode,
    make_prompts,
    make_training_batch,
    read_generated,
)


def test_vocabulary_is_frozen():
    """Trained models depend on these exact ids. If this test fails, you have
    changed the vocabulary: every saved model and cached dataset is now invalid.
    Only update the expected values deliberately, and record it in P10."""
    assert CHARACTERS == "0123456789runleftighQA: +=?()|\n"
    assert (PAD_ID, EOS_ID, VOCAB_SIZE) == (31, 32, 33)


def test_encode_decode_round_trip():
    text = example_text(Addition(6133, 8894), 15027)
    assert decode(encode(text)) == text


def test_every_character_of_the_format_is_in_the_vocabulary():
    """encode raises on unknown characters, so this covers the whole layout."""
    encode(example_text(Addition(1234567890, 987654321), 2222222211))


def test_unknown_character_raises():
    with pytest.raises(KeyError, match="'x'"):
        encode("Q: 1 x 2")


def test_training_batch_layout():
    """Rows are prompt + answer + EOS, right-padded; the mask covers the answer + EOS."""
    examples = [(Addition(1, 2), 3), (Addition(6133, 8894), 15027)]
    tokens, loss_mask = make_training_batch(examples)

    assert tokens.dtype == torch.int64 and loss_mask.dtype == torch.bool
    assert tokens.shape == loss_mask.shape == (2, len(example_text(*examples[1])) + 1)

    for row, mask, (problem, total) in zip(tokens.tolist(), loss_mask.tolist(), examples):
        n_prompt, n_answer = len(prompt_text(problem)), len(str(total))
        # Graded: exactly the answer characters and the EOS after them.
        assert mask == [False] * n_prompt + [True] * (n_answer + 1) + [False] * (len(row) - n_prompt - n_answer - 1)
        real = row[: n_prompt + n_answer]
        assert decode(real) == example_text(problem, total)
        assert row[n_prompt + n_answer] == EOS_ID
        assert all(t == PAD_ID for t in row[n_prompt + n_answer + 1 :])


def test_prompts_match_the_start_of_training_rows():
    """What the model sees at test time must be exactly how training rows begin.
    Any mismatch (e.g. a missing space after "A:") would make the model answer
    a prompt it never saw in training."""
    problems = [Addition(1, 2), Addition(6133, 8894)]
    tokens, _ = make_training_batch([(p, solve(p)) for p in problems])
    for prompt, row in zip(make_prompts(problems), tokens.tolist()):
        assert row[: len(prompt)] == prompt


@pytest.mark.parametrize(
    "ids, expected",
    [
        (encode("15027") + [EOS_ID], 15027),
        (encode("0") + [EOS_ID], 0),
    ],
)
def test_read_generated_readable(ids, expected):
    assert read_generated(ids) == expected


@pytest.mark.parametrize(
    "ids, raw",
    [
        (encode("15027"), "15027"),  # no EOS: cut off at the length cap
        ([], ""),  # nothing generated
        ([EOS_ID], "<eos>"),  # stopped immediately: empty answer
        (encode("015") + [EOS_ID], "015<eos>"),  # leading zero
        (encode("1") + [PAD_ID] + encode("5") + [EOS_ID], "1<pad>5<eos>"),
        (encode("12\nA") + [EOS_ID], "12\nA<eos>"),  # kept writing past the answer
    ],
)
def test_read_generated_unreadable(ids, raw):
    answer = read_generated(ids)
    assert isinstance(answer, Unreadable)
    assert answer.raw == raw
