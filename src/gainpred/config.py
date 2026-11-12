"""All settings that define the model and its optimiser (protocol P1).

Why a separate config object?
-----------------------------
Every experiment in the study must use *exactly* the same model. Putting every
number in one frozen dataclass means:

* there is one place to read (and check against P1) instead of numbers being
  scattered through the code;
* the config can be saved next to each checkpoint and logged with each run, so
  months later you can tell exactly what was trained;
* ``frozen=True`` makes it impossible to change a setting by accident halfway
  through a run.

Values marked "P1" come from Shin et al. (Section A.1, A.2, Table 4).
Values marked "P10 assumption" are our choices where the paper is silent; they
are listed in docs/protocol.md P10 so they can be checked against their code.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    # ------------------------------------------------------------------
    # Vocabulary. These depend on the character set chosen in P2, so they
    # have no defaults: whoever builds the tokenizer passes them in.
    # ------------------------------------------------------------------
    vocab_size: int
    """Number of distinct characters (tokens) the model can read and write.
    About 30 for this study: digits, letters of "runlen" etc., punctuation,
    plus special tokens such as padding and end-of-sequence."""

    pad_id: int
    """Token id used to fill short sequences up to the batch length. The model
    is told to ignore these positions completely (see model.Attention)."""

    # ------------------------------------------------------------------
    # Shape of the network (P1).
    # ------------------------------------------------------------------
    n_layers: int = 6
    """P1. Number of Transformer blocks stacked on top of each other."""

    d_model: int = 384
    """P1 ("hidden size"). Length of the vector that represents each character
    inside the model. Every block reads and writes vectors of this size."""

    n_heads: int = 6
    """P1. Attention is split into this many independent "heads", each working
    on d_model / n_heads = 64 dimensions. Different heads can learn to look for
    different things (e.g. "the digit in the same column of the other number"
    versus "the previous carry")."""

    d_ff: int = 1536
    """Inner width of the feed-forward (SwiGLU) layer.

    Not stated in P1 directly; it is *implied* by the parameter count. A
    Llama-style block with d_ff = 4 * d_model = 1536 has
        attention  4 * d_model**2           =   589,824
        SwiGLU     3 * d_model * d_ff       = 1,769,472
        total per block                     = 2,359,296
    times 6 blocks = 14.16M, matching P1's "about 14.2M". A GPT-2-style block
    would give only 10.6M. (P10 assumption: Llama-style block.)"""

    max_len: int = 128
    """Longest sequence (question + answer, in characters) the model will ever
    see. Only used for sanity checks: with no positional encoding the weights
    do not depend on it. A 15-digit addition is about 50 characters."""

    # ------------------------------------------------------------------
    # Initialisation (P10 assumption: GPT-2 / Llama convention).
    # ------------------------------------------------------------------
    init_std: float = 0.02
    """Standard deviation of the normal distribution used for initial weights."""

    norm_eps: float = 1e-5
    """Small constant inside RMSNorm that prevents division by zero."""

    # ------------------------------------------------------------------
    # Optimiser (P1). Kept here so model + optimiser are logged together;
    # used by Transformer.configure_optimizer().
    # ------------------------------------------------------------------
    lr: float = 5e-4
    """P1. Learning rate; constant, no warm-up (P10 assumption)."""

    betas: tuple[float, float] = (0.9, 0.99)
    """P1. AdamW's decay rates for its running averages of the gradient (0.9)
    and of the squared gradient (0.99)."""

    eps: float = 1e-12
    """P1. AdamW's epsilon. Unusually small (the PyTorch default is 1e-8). It
    only has an effect if optimiser state is kept in float32: see the
    precision note in model.py."""

    weight_decay: float = 0.1
    """P1. AdamW weight decay. P10 assumption: applied to weight matrices only,
    not to norm gains (the usual convention; see configure_optimizer)."""

    def __post_init__(self) -> None:
        # Fail early, with a clear message, rather than deep inside a reshape.
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be a valid token id")

    @property
    def head_dim(self) -> int:
        """Dimensions per attention head: 384 / 6 = 64."""
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        """Plain dictionary, handy for saving next to a checkpoint or logging."""
        return asdict(self)
