"""The character-level Transformer of protocol P1.

Big picture
===========
The model reads a string one character at a time and, at every position,
predicts the *next* character. That is all it ever does. Answering
``Q: 6133 + 8894 = ?A:`` means predicting ``1``, appending it, predicting
``5``, appending it, and so on until it predicts the end-of-sequence token.
This is a "decoder-only" or "causal language model", the same family as GPT
and Llama, just tiny (14.2M parameters versus billions) and with a vocabulary
of about 30 characters instead of ~100,000 word pieces.

Data flow through ``Transformer.forward``::

    token ids            (B, T)            B = batch size, T = sequence length
      │  embedding: look up a learned 384-number vector per character
      ▼
    vectors              (B, T, 384)
      │  6 x Block:  x = x + Attention(RMSNorm(x))   <- characters exchange information
      │              x = x + SwiGLU(RMSNorm(x))      <- each position "thinks" on its own
      ▼
    vectors              (B, T, 384)
      │  final RMSNorm, then a linear layer to one score per vocabulary entry
      ▼
    logits               (B, T, vocab)     scores for "which character comes next"

The architecture is "Llama-style" (P10 assumption, implied by the 14.2M
parameter count; see config.ModelConfig.d_ff):

* RMSNorm instead of LayerNorm,
* "pre-norm": normalise *before* each sub-layer, add the result back,
* SwiGLU feed-forward layer instead of a plain MLP with GELU,
* no bias terms in any linear layer,
* separate (untied) input embedding and output layer,
* and, unusually, **no positional encoding at all** (P1). See Attention.

Precision note (bfloat16, P1)
=============================
P1 says "bfloat16". We keep the *weights* and AdamW's state in float32 and run
only the heavy matrix multiplications in bfloat16, via ``torch.autocast``
(standard "mixed precision"). bfloat16 has only ~3 significant decimal digits,
which is fine for a forward pass but too coarse to accumulate many small weight
updates. It is also why P1's AdamW epsilon of 1e-12 means anything: in pure
bfloat16 the squared-gradient average it is added to would round it away.
The training loop is responsible for the autocast; this file is precision-agnostic.

Shape conventions in comments: B batch, T sequence length, D d_model (384),
H heads (6), Dh head dim (64), F feed-forward width (1536), V vocab size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gainpred.config import ModelConfig


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalisation.

    What it does
        Rescales each position's 384-number vector so its root-mean-square is 1,
        then multiplies element-wise by a learned "gain" vector (initially all 1s):

            y = x / sqrt(mean(x**2) + eps) * gain

    Why
        As vectors pass through many layers their scale drifts; normalising
        keeps every layer's input in a predictable range, which makes training
        stable. LayerNorm (used by GPT-2) also subtracts the mean and adds a
        learned bias; RMSNorm (Llama) skips both. It is slightly cheaper and works
        just as well in practice.

    Shapes: (..., D) -> (..., D)
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 even under bfloat16 autocast: squaring and averaging
        # are where low precision hurts most. Then cast back to the input dtype.
        x_float = x.float()
        rms_inverse = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms_inverse).type_as(x) * self.gain


class Attention(nn.Module):
    """Causal multi-head self-attention, with no positional encoding.

    What it does
        This is the only place where different characters exchange information.
        Each position produces three vectors from its current state:

        * a **query**  -- "what am I looking for?"
        * a **key**    -- "what do I contain?"
        * a **value**  -- "what will I hand over if someone attends to me?"

        A position compares its query with every visible key (dot product,
        scaled by 1/sqrt(Dh)), turns the scores into weights that sum to 1
        (softmax), and takes the weighted average of the values. With 6 heads
        this happens 6 times in parallel on 64-dimensional slices; the results
        are concatenated and mixed by one more linear layer (``out``).

    Causal mask
        A position may only look at itself and *earlier* positions. Without
        this the model could "predict" the next character by peeking at it
        during training, and would be useless when generating, where the future
        does not exist yet.

    Padding mask
        Sequences in a batch have different lengths and are filled out with a
        padding token. Padding positions are hidden from every query, so they
        cannot influence the result. One subtlety: a padding position's own
        query would then see *nothing*, and a softmax over nothing is 0/0 = NaN.
        So every position is always allowed to see itself. Outputs at padding
        positions are meaningless but harmless: they are never read.

    No positional encoding (P1, "NoPE")
        GPT-2 adds a learned position vector to each character; Llama rotates
        queries and keys by a position-dependent angle (RoPE). Here there is
        neither, so attention on its own cannot tell *where* a character is.
        Position information still emerges from the causal mask: position t
        averages over exactly t+1 items, so e.g. a head that attends uniformly
        to everything visible produces a signal that changes with t, and later
        layers can decode it. Shin et al. use NoPE because it tends to
        generalise to longer inputs than seen in training -- which is precisely
        the easy-to-hard question of this study.

        A handy side effect: inserting padding on the *left* of a sequence
        (as generate() does) changes nothing for the real characters, since
        padding is masked and there are no position numbers to shift.

    Shapes: x (B, T, D), mask (B, 1, T, T) bool -> (B, T, D)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        # Four D x D matrices, no biases: 4 * 384**2 = 589,824 parameters.
        self.query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh): heads become a batch
            # dimension so all 6 are computed in one matrix multiplication.
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query(x))  # (B, H, T, Dh)
        k = split_heads(self.key(x))  # (B, H, T, Dh)
        v = split_heads(self.value(x))  # (B, H, T, Dh)

        # softmax(q @ k^T / sqrt(Dh), masked) @ v, in one fused, memory-efficient
        # PyTorch call. mask is True where attending is allowed; it broadcasts
        # over the head dimension.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, Dh)

        # Undo split_heads: (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class SwiGLU(nn.Module):
    """Gated feed-forward layer (Llama's MLP).

    What it does
        Works on each position independently (no information moves between
        characters here). It widens the vector from D=384 to F=1536, applies a
        non-linearity, and projects back down to 384:

            out = down( silu(gate(x)) * up(x) )

        ``silu(z) = z * sigmoid(z)`` is a smooth version of ReLU. The element-wise
        product with ``up(x)`` is the "gate": one projection decides how much of
        the other passes through. A plain GPT-2 MLP is ``down(gelu(up(x)))``;
        the gated form has a third matrix and in practice trains a bit better.

    Why have it at all
        Attention only takes weighted averages, which is essentially linear.
        This layer is where the non-linear "computation" happens -- e.g. turning
        "digits 7 and 8 meet here" into "write 5, carry 1".

    Parameters: 3 matrices of 384 x 1536 = 1,769,472 (75% of each block).
    Shapes: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))  # (B,T,F) -> (B,T,D)


class Block(nn.Module):
    """One Transformer layer: attention, then feed-forward, each "pre-norm residual".

        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Residual connection ("x = x + ...")
        Each sub-layer computes a *change* to add to x rather than replacing it.
        The vector x is a shared workspace (the "residual stream") that every
        layer reads from and writes into. This gives gradients a direct path
        from the loss back to early layers, which is what makes deep stacks
        trainable at all.

    Pre-norm
        Normalisation happens on the way *into* each sub-layer, leaving the
        residual stream itself untouched. The original 2017 Transformer
        normalised after adding ("post-norm"), which is harder to train
        without a learning-rate warm-up -- relevant, since P1 uses none.

    Shapes: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ----------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------


class Transformer(nn.Module):
    """Embedding -> 6 Blocks -> RMSNorm -> output layer.

    Parameter count (vocab V ~ 30)
        embedding         V x 384          ~      11.5k
        6 blocks          6 x 2,360,064    = 14,160,384   (incl. 2 x 384 norm gains each)
        final norm        384
        output layer      384 x V          ~      11.5k
        total                              ~ 14.18M      (P1: "about 14.2M")

    Untied embeddings
        The input embedding (character -> vector) and the output layer
        (vector -> score per character) are separate matrices, as in Llama.
        GPT-2 shares ("ties") them. With 30 characters this choice is worth
        only ~12k parameters either way.

    Initialisation (P10 assumption)
        Every weight matrix starts as random normal numbers with std 0.02
        (GPT-2 / Llama convention); norm gains start at 1. Small random values
        break the symmetry between units while keeping early outputs near zero.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        # RMSNorm gains are created as ones and need no extra init.

    # ------------------------------------------------------------------
    # Forward pass and loss
    # ------------------------------------------------------------------

    def forward(
        self, tokens: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Score every possible next character at every position.

        Args:
            tokens: (B, T) integer token ids, padded with ``cfg.pad_id``.
                Padding may be on the right (training) or left (generation).
            loss_mask: optional (B, T) bool, True at the characters the model
                should be *graded on* -- the answer characters and the
                end-of-sequence token after them (P1 assumption: loss on answer
                characters only). If given, the loss is computed.

        Returns:
            logits: (B, T, V) float32. ``logits[b, t]`` scores the character
                that should come at position t+1.
            loss: average cross-entropy over masked target characters, or None.

        How the loss lines up
            Position t predicts token t+1. So predictions ``logits[:, :-1]`` are
            compared with targets ``tokens[:, 1:]``, and a target counts if
            ``loss_mask[:, 1:]`` is True there. The question characters are still
            *read* (the model needs them to answer) -- they just are not graded,
            since predicting random question digits is impossible and would only
            add noise to the loss.
        """
        B, T = tokens.shape
        mask = self._attention_mask(tokens)  # (B, 1, T, T)

        x = self.embedding(tokens)  # (B, T, D)
        for block in self.blocks:
            x = block(x, mask)  # (B, T, D)
        x = self.final_norm(x)  # (B, T, D)
        # float() so the softmax inside cross-entropy runs in full precision.
        logits = self.output(x).float()  # (B, T, V)

        loss = None
        if loss_mask is not None:
            predictions = logits[:, :-1]  # (B, T-1, V)
            targets = tokens[:, 1:]  # (B, T-1)
            graded = loss_mask[:, 1:] & (targets != self.cfg.pad_id)  # (B, T-1)
            loss = F.cross_entropy(predictions[graded], targets[graded])
        return logits, loss

    def _attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, 1, T, T) bool: may query position i attend to key position j?

        Yes iff j <= i (causal) and key j is a real character (not padding),
        or j == i (always see yourself; avoids an all-masked softmax row).
        """
        T = tokens.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=tokens.device).tril()  # (T, T)
        key_is_real = (tokens != self.cfg.pad_id)[:, None, None, :]  # (B, 1, 1, T)
        diagonal = torch.eye(T, dtype=torch.bool, device=tokens.device)  # (T, T)
        return (causal & key_is_real) | diagonal  # (B, 1, T, T)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[int]],
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Continue each prompt until it emits ``eos_id`` (or hits the limit).

        Args:
            prompts: token-id lists, e.g. the encoding of ``Q: 61 + 88 = ?A:``.
                They may differ in length.
            eos_id: end-of-sequence token; generation stops once emitted.
            max_new_tokens: hard limit, so an untrained model cannot loop forever.
            temperature: 0 means **greedy** decoding -- always take the highest-
                scoring character (P1: "greedy decoding for all built answers").
                Above 0 means sample from softmax(logits / temperature); P5's
                "sample and vote" teacher uses temperature 1.0.
            generator: optional seeded random generator, for reproducible sampling.

        Returns:
            The newly generated ids for each prompt, without the prompt and
            without the final eos. Decoding them into a string, and parsing that
            into an answer (P2, "unreadable answers"), is the caller's job.

        How batching works
            Prompts are *left*-padded so they all end at the same column, and
            the next character for every row can be read from the last column.
            Because padding is masked and the model has no positional encoding,
            the padding does not change any answer (tested in tests/test_model.py).

        No key/value cache
            Large-model generators cache each layer's keys and values so each new
            character costs one position of work instead of re-running the whole
            sequence. Our sequences are ~50 characters, so we simply re-run the
            full forward pass each step: slower in principle, much simpler, and
            still fast in practice.
        """
        device = next(self.parameters()).device
        pad_id = self.cfg.pad_id
        width = max(len(p) for p in prompts)
        if width + max_new_tokens > self.cfg.max_len:
            raise ValueError("prompt + max_new_tokens exceeds cfg.max_len")

        # Left-pad into one (B, width) tensor.
        tokens = torch.tensor(
            [[pad_id] * (width - len(p)) + list(p) for p in prompts], device=device
        )
        B = tokens.shape[0]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated: list[torch.Tensor] = []

        was_training = self.training
        self.eval()  # no dropout etc. exists here, but this is the convention
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            next_logits = logits[:, -1, :]  # (B, V): scores for the next character
            next_logits[:, pad_id] = float("-inf")  # never "generate" padding

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1)  # (B,)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator).squeeze(1)

            # Rows that already finished just receive padding from now on.
            next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            finished |= next_token == eos_id
            if finished.all():
                break
        self.train(was_training)

        out = torch.stack(generated, dim=1).tolist() if generated else [[] for _ in prompts]
        results = []
        for row in out:
            row = row[: row.index(eos_id)] if eos_id in row else row
            results.append([t for t in row if t != pad_id])
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Total number of learned numbers in the model (P9 test: ~14.2M)."""
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Parameter count per top-level part; useful for checking against P1."""
        return {
            name: sum(p.numel() for p in child.parameters())
            for name, child in self.named_children()
        }

    def configure_optimizer(self) -> torch.optim.AdamW:
        """AdamW with the P1 settings.

        Weight decay pulls weights gently towards zero at every step, a mild
        brake on memorising. P10 assumption: it applies to every weight matrix
        (2-D parameters: linear layers and the embedding) but not to the RMSNorm
        gains (1-D), whose natural value is 1, not 0. This is the usual
        convention in GPT-2 / Llama training code.
        """
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.cfg.lr, betas=self.cfg.betas, eps=self.cfg.eps)


def pick_device() -> torch.device:
    """Best available device: an NVIDIA GPU (cloud), else Apple GPU (M2), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
