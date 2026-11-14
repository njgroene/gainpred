"""Characters <-> token ids, and building batches for training and generation.

What a tokenizer is
===================
A neural network works on numbers, not characters. The tokenizer gives every
character a fixed integer id; the model's first layer (the embedding) turns
each id into a learned vector. Large language models use "subword" tokenizers
where one token may be a whole word; this study is **character-level** (P1),
so one token is exactly one character. That matters for arithmetic: each digit
gets its own position, so "15027" can never be chopped into odd chunks such as
"150" + "27".

The vocabulary is fixed below and must never be reordered: a trained model's
embedding rows are tied to these ids. It already includes the characters for
the longest-run questions (P2), so adding those later will not change the
vocabulary size or invalidate trained models.

Special tokens
==============
* PAD fills short sequences up to the batch width; the model masks it out.
* EOS ("end of sequence") is written by the model after the answer, so it
  knows when to stop and we know the answer is complete.

Neither is a real character, so they sit at the end of the vocabulary and
``encode`` cannot produce them from text.
"""

from __future__ import annotations

import torch

from collections.abc import Callable

from gainpred.formats import (
    Addition,
    Answer,
    Run,
    RunAnswer,
    RunQuestion,
    Unreadable,
    answer_text,
    parse_answer,
    prompt_text,
)

# ----------------------------------------------------------------------------
# The vocabulary: 31 characters + 2 special tokens = 33 ids.
# ----------------------------------------------------------------------------
DIGITS = "0123456789"
LETTERS = "runleftigh"  # enough to spell runlen, leftrun, rightrun (P2)
SYMBOLS = "QA: +=?()|\n"  # "\n" is the question/answer separator (formats.SEPARATOR)
CHARACTERS = DIGITS + LETTERS + SYMBOLS

PAD_ID = len(CHARACTERS)  # 31
EOS_ID = len(CHARACTERS) + 1  # 32
VOCAB_SIZE = len(CHARACTERS) + 2  # 33

_CHAR_TO_ID = {char: i for i, char in enumerate(CHARACTERS)}
assert len(_CHAR_TO_ID) == len(CHARACTERS), "duplicate character in vocabulary"


# ----------------------------------------------------------------------------
# Text <-> ids
# ----------------------------------------------------------------------------


def encode(text: str) -> list[int]:
    """``"Q: 1"`` -> ``[20, 22, 23, 1]``. Raises KeyError on unknown characters.

    Failing loudly is deliberate: silently mapping an unknown character to some
    "unknown" token would hide formatting bugs.
    """
    try:
        return [_CHAR_TO_ID[char] for char in text]
    except KeyError as error:
        raise KeyError(f"character {error.args[0]!r} is not in the vocabulary") from None


def decode(ids: list[int]) -> str:
    """Inverse of encode, for real characters only (PAD/EOS raise IndexError)."""
    return "".join(CHARACTERS[i] for i in ids)


# ----------------------------------------------------------------------------
# Batches
# ----------------------------------------------------------------------------


def make_prompts(problems: list[Addition] | list[RunQuestion]) -> list[list[int]]:
    """Token ids of each prompt (question + "\\nA: "), ready for model.generate.

    Works for both tasks: ``formats.prompt_text`` writes an addition or a
    longest-run question as needed.
    """
    return [encode(prompt_text(problem)) for problem in problems]


def make_training_batch(
    examples: list[tuple[Addition, int]] | list[tuple[RunQuestion, Run]], device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn (problem, answer to train on) pairs into model inputs.

    Either task: ``(Addition, int)`` or ``(RunQuestion, Run)`` pairs, since
    ``formats.prompt_text`` and ``formats.answer_text`` handle both.

    Each row is ``prompt + answer + EOS``, right-padded with PAD to the longest
    row in the batch. Right padding is fine for training: thanks to the causal
    mask, real characters never look at the padding after them.

    Returns:
        tokens:    (B, T) int64 token ids.
        loss_mask: (B, T) bool, True exactly at the answer characters and EOS --
                   the characters the model is graded on (P1). The prompt is
                   read but not graded; padding is neither.

    Example (one row, ``.`` = PAD)::

        text      Q: 1 + 2 = ?\\nA: 3<EOS>..
        loss_mask 0000000000000000 1  1   00
    """
    rows, masks = [], []
    for problem, total in examples:
        prompt = encode(prompt_text(problem))
        answer = encode(answer_text(total)) + [EOS_ID]
        rows.append(prompt + answer)
        masks.append([False] * len(prompt) + [True] * len(answer))

    width = max(len(row) for row in rows)
    tokens = torch.tensor([row + [PAD_ID] * (width - len(row)) for row in rows], device=device)
    loss_mask = torch.tensor(
        [mask + [False] * (width - len(mask)) for mask in masks], device=device
    )
    return tokens, loss_mask


def read_generated(ids: list[int], parse: Callable[[str], Answer | RunAnswer] = parse_answer) -> Answer | RunAnswer:
    """Turn what model.generate produced into an answer (P2).

    ``parse`` reads the answer text; the default reads an addition answer.
    For longest-run questions pass ``formats.parse_run_answer``. (The model's
    output alone does not say which task it answered, so the caller must.)

    The output is readable only if the model *stopped by itself*, i.e. the ids
    end with EOS. Without that check, a model that runs on writing digits until
    the length cap would have its cut-off output read as a valid number.
    Anything else odd -- PAD or EOS in the middle, a malformed number -- gives
    Unreadable. Never raises.

    An Unreadable's ``raw`` is always the *whole* output with special tokens
    shown, e.g. ``"015<eos>"`` or ``"15027"`` (no ``<eos>``: cut off), so logs
    show both what the model wrote and whether it stopped.
    """
    unreadable = Unreadable(_debug_text(ids))
    if not ids or ids[-1] != EOS_ID:
        return unreadable
    body = ids[:-1]
    if any(i >= len(CHARACTERS) for i in body):
        return unreadable
    answer = parse(decode(body))
    return unreadable if isinstance(answer, Unreadable) else answer


def _debug_text(ids: list[int]) -> str:
    """Readable rendering of any ids, special tokens included, for logs."""
    names = {PAD_ID: "<pad>", EOS_ID: "<eos>"}
    return "".join(names.get(i, CHARACTERS[i] if i < len(CHARACTERS) else f"<{i}>") for i in ids)
