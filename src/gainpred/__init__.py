"""gainpred: self-improvement experiments on easy-to-hard generalisation.

The study design lives in ``docs/protocol.md``. Code comments refer to its
sections as P1, P2, ... so you can jump between the two.

Modules so far
--------------
config    -- every number that defines the model (protocol P1), in one place.
model     -- the Transformer itself, plus greedy decoding.
formats   -- how problems and answers are written as text (protocol P2).
tokenizer -- characters <-> token ids; training batches and prompts.
cuts      -- the P4 cut table: splitting hard problems into easy pieces.
data      -- the addition datasets (P1, P7); run ``python -m gainpred.data``.
"""

from gainpred.config import ModelConfig
from gainpred.model import Transformer

__all__ = ["ModelConfig", "Transformer"]
