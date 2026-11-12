"""Tests for the P1 model.

Each test checks one property the rest of the study silently relies on. Run with

    uv run pytest                  # everything (memorisation takes ~1 minute)
    uv run pytest -m "not slow"    # skip the memorisation test
    uv run pytest -s               # also show printed output (parameter table, timings)

Inputs use the real vocabulary and P2 layout from gainpred.tokenizer.
"""

import random
import time

import pytest
import torch

from gainpred.config import ModelConfig
from gainpred.formats import Addition, max_answer_chars, solve
from gainpred.model import Transformer, pick_device
from gainpred.tokenizer import (
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    make_prompts,
    make_training_batch,
    read_generated,
)

def make_model(**overrides) -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, **overrides))


def small_model() -> Transformer:
    """A 2-layer, 64-wide model: same code paths, runs in milliseconds."""
    return make_model(n_layers=2, d_model=64, n_heads=4, d_ff=256)


# ----------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------


def test_parameter_count_matches_p1():
    """P9 acceptance test: "about 14.2M parameters".

    This is also the evidence for the Llama-style block (see config.d_ff): a
    GPT-2-style block of the same width would give ~10.6M.
    """
    model = make_model()
    n = model.num_params()
    print(f"\nparameters: {n:,}")
    for part, count in model.param_breakdown().items():
        print(f"  {part:<12} {count:>12,}")
    assert 14.1e6 < n < 14.3e6


def test_output_shapes():
    """Logits have one score per vocabulary entry at every position."""
    model = small_model()
    tokens = torch.randint(0, PAD_ID, (3, 10))
    logits, loss = model(tokens)
    assert logits.shape == (3, 10, VOCAB_SIZE)
    assert loss is None


# ----------------------------------------------------------------------------
# Masking: the properties that make training and generation correct
# ----------------------------------------------------------------------------


def test_causal_no_peeking_at_the_future():
    """Changing a later character must not change predictions at earlier positions.

    If this failed, training would be "cheating" (reading the answer it is
    meant to predict) and generation would behave differently from training.
    """
    model = small_model().eval()
    a = torch.randint(0, PAD_ID, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % PAD_ID  # change only positions 8 onwards
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[0, :8], logits_b[0, :8])
    assert not torch.allclose(logits_a[0, 8:], logits_b[0, 8:])


def test_left_padding_does_not_change_the_output():
    """A sequence alone and the same sequence after 5 padding tokens must agree.

    generate() relies on this to batch prompts of different lengths. It holds
    because padding is masked out and, with no positional encoding, shifting
    a sequence to the right is invisible to the model.
    """
    model = small_model().eval()
    seq = torch.randint(0, PAD_ID, (1, 9))
    padded = torch.cat([torch.full((1, 5), PAD_ID), seq], dim=1)
    logits_alone, _ = model(seq)
    logits_padded, _ = model(padded)
    torch.testing.assert_close(logits_alone, logits_padded[:, 5:], atol=1e-5, rtol=1e-5)


def test_loss_grades_exactly_the_masked_non_padding_targets():
    """The loss must be the average cross-entropy over exactly the characters
    marked in loss_mask, skipping padding even where the mask says True.

    We recompute it by hand: position t predicts token t+1, so the score for
    target tokens[b, t+1] lives in logits[b, t].
    """
    model = small_model()
    tokens = torch.randint(0, PAD_ID, (2, 10))
    tokens[1, 7:] = PAD_ID  # second row is shorter: 3 padding positions
    loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
    loss_mask[:, 5:] = True  # "answer" = positions 5..9, which includes the padding

    logits, loss = model(tokens, loss_mask)

    by_hand = []
    for b in range(2):
        for t in range(1, 10):  # t = position of the target character
            if loss_mask[b, t] and tokens[b, t] != PAD_ID:
                log_probs = torch.log_softmax(logits[b, t - 1], dim=-1)
                by_hand.append(-log_probs[tokens[b, t]])
    assert len(by_hand) == 5 + 2  # row 0: positions 5-9; row 1: positions 5-6
    torch.testing.assert_close(loss, torch.stack(by_hand).mean())


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------


PROMPT_PROBLEMS = [Addition(1, 2), Addition(123, 4567), Addition(9, 90)]


def test_generate_batched_equals_one_by_one():
    """Greedy answers must not depend on what else is in the batch.

    The three prompts have different lengths, so the batch needs left padding.
    """
    model = small_model()
    prompts = make_prompts(PROMPT_PROBLEMS)
    batched = model.generate(prompts, eos_id=EOS_ID, max_new_tokens=8)
    alone = [model.generate([p], eos_id=EOS_ID, max_new_tokens=8)[0] for p in prompts]
    assert batched == alone
    for out in batched:
        assert len(out) <= 8 and PAD_ID not in out
        # EOS, if present, is the last token: generation stops right after it.
        assert EOS_ID not in out[:-1]


def test_sampling_is_reproducible_with_a_seed():
    """Same seed, same samples (needed for P9's reproducibility)."""
    model = small_model()
    prompts = make_prompts([Addition(1, 2)] * 4)
    a = model.generate(prompts, EOS_ID, 8, temperature=1.0, generator=torch.Generator().manual_seed(1))
    b = model.generate(prompts, EOS_ID, 8, temperature=1.0, generator=torch.Generator().manual_seed(1))
    assert a == b


# ----------------------------------------------------------------------------
# P9 acceptance test: memorise 64 problems
# ----------------------------------------------------------------------------


def make_toy_problems(n: int, seed: int) -> list[Addition]:
    """n random additions of 1-4 digit numbers."""
    rng = random.Random(seed)
    return [Addition(rng.randint(1, 9999), rng.randint(1, 9999)) for _ in range(n)]


@pytest.mark.slow
def test_memorises_64_problems():
    """P9: the full-size model memorises 64 problems to zero error.

    The point is not that this is hard -- a 14M-parameter model can memorise 64
    strings easily. The point is that if *anything* is wired wrong (loss lined
    up with the wrong position, mask hiding the question, optimiser not
    stepping, decoding reading the wrong column), this fails. It exercises the
    whole path from P2 text to tokens to loss to weight update to greedy answer
    and back to a number.

    It also runs the training step exactly as the real loop will: float32
    weights, bfloat16 autocast, P1's AdamW settings.
    """
    device = pick_device()
    model = make_model().to(device)
    optimizer = model.configure_optimizer()
    problems = make_toy_problems(64, seed=0)
    tokens, loss_mask = make_training_batch([(p, solve(p)) for p in problems], device)

    start = time.perf_counter()
    for step in range(1, 2001):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(tokens, loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 and loss.item() < 1e-3:
            break
    elapsed = time.perf_counter() - start
    print(f"\n{device}: {step} steps in {elapsed:.1f}s ({step / elapsed:.1f} steps/s), loss {loss.item():.2e}")

    # +1 leaves room for EOS after the longest possible answer.
    limit = max(max_answer_chars(p) for p in problems) + 1
    outputs = model.generate(make_prompts(problems), eos_id=EOS_ID, max_new_tokens=limit)
    answers = [read_generated(out) for out in outputs]
    wrong = [(p, got) for p, got in zip(problems, answers) if got != solve(p)]
    assert not wrong, f"{len(wrong)} of 64 wrong, e.g. {wrong[:3]}"
