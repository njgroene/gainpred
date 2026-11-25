# Pilot report

## Stage A: answer key, floor 2, rounds 1-2 (repeats 0 and 1)

* **Gate A: passed** (>= 90% at every length 16-19 in both repeats).
  * `solid-key-r0`: lowest at 16-19 after round 2: 100.0%.
  * `solid-key-r1`: lowest at 16-19 after round 2: 100.0%.

*3-7 and floor 1 (8-15)*

| Run | When | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-key-r0` | after round 2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `solid-key-r1` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-key-r1` | after round 2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

*floor 2 (16-23)*

| Run | When | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | after round 2 | 100.0 | 100.0 | 100.0 | 100.0 | – | – | – | – |
| `solid-key-r1` | after round 2 | 100.0 | 100.0 | 100.0 | 100.0 | – | – | – | – |

## Stage B: agreement rule, floor 2 (solid floor 1)

**Gate B: passed** (floor 2 learned).

* **Verdict: learned** (3 repeats finished round 4; learned needs 2 of 3 repeats at >= 90% at every length 16-23).
* Floor 1 eroded (> 3 points below the repeat's own level before the floor's first round, in 2 of 3 repeats): **no**.
  * `solid-agr-r0`: learned yes (lowest on the floor 100.0%); floor 1 eroded no; largest drop at any length 3-15: 0.1 points at 13 digits.
  * `solid-agr-r1`: learned yes (lowest on the floor 99.9%); floor 1 eroded no; largest drop at any length 3-15: 0.0 points at 13 digits.
  * `solid-agr-r2`: learned yes (lowest on the floor 100.0%); floor 1 eroded no; largest drop at any length 3-15: 0.1 points at 13 digits.

### Accuracy per length and repeat (percent)

*3-7 and floor 1 (8-15)*

| Run | When | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-agr-r0` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-agr-r0` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 99.9 |
| `solid-agr-r1` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-agr-r1` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `solid-agr-r2` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-agr-r2` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 |

*floor 2 (16-23)*

| Run | When | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-agr-r0` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `solid-agr-r1` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 |
| `solid-agr-r2` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Checks at the new lengths, each round (the condition's own teacher)

| Run | Round | Digits | Kept share | Novelty | Confirmed share | Kept right (audit) | Self-taught pieces replaced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-agr-r0` | 1 | 16 | 35.2 | 0.4 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 1 | 17 | 35.1 | 77.4 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 2 | 18 | 34.1 | 1.0 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 2 | 19 | 35.2 | 61.9 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 3 | 20 | 33.9 | 0.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 3 | 21 | 36.0 | 14.0 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 4 | 22 | 33.8 | 0.0 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r0` | 4 | 23 | 34.5 | 22.2 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 1 | 16 | 34.1 | 0.9 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 1 | 17 | 33.6 | 77.7 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 2 | 18 | 36.4 | 0.5 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 2 | 19 | 33.6 | 89.6 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 3 | 20 | 33.8 | 3.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 3 | 21 | 33.8 | 99.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 4 | 22 | 34.2 | 0.1 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r1` | 4 | 23 | 36.3 | 20.9 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 1 | 16 | 33.0 | 0.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 1 | 17 | 34.9 | 77.4 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 2 | 18 | 33.8 | 0.4 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 2 | 19 | 33.7 | 30.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 3 | 20 | 35.0 | 0.0 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 3 | 21 | 35.7 | 27.3 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 4 | 22 | 33.1 | 1.1 | 100.0 | 100.00 | 0.00 |
| `solid-agr-r2` | 4 | 23 | 34.9 | 75.5 | 100.0 | 100.00 | 0.00 |

### Floor-1 novelty each round (trusted-piece teacher; problems with no carry at cut 1 only; percent)

| Run | Round | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-agr-r0` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r0` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r0` | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r0` | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r1` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r1` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r1` | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r1` | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r2` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r2` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r2` | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-agr-r2` | 4 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Stage B: answer key, floor 2, rounds 3-4 continued (control)

* **Verdict: learned** (2 repeats finished round 4; learned needs all repeats at >= 90% at every length 16-23).
* Floor 1 eroded (> 3 points below the repeat's own level before the floor's first round, in all repeats): **no**.
  * `solid-key-r0`: learned yes (lowest on the floor 100.0%); floor 1 eroded no; largest drop at any length 3-15: 0.0 points at 13 digits.
  * `solid-key-r1`: learned yes (lowest on the floor 100.0%); floor 1 eroded no; largest drop at any length 3-15: 0.0 points at 14 digits.

### Accuracy per length and repeat (percent)

*3-7 and floor 1 (8-15)*

| Run | When | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-key-r0` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `solid-key-r1` | before | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| `solid-key-r1` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

*floor 2 (16-23)*

| Run | When | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `solid-key-r1` | after round 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Checks at the new lengths, each round (the condition's own teacher)

| Run | Round | Digits | Kept share | Novelty | Confirmed share | Kept right (audit) | Self-taught pieces replaced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | 1 | 16 | 100.0 | 0.7 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 1 | 17 | 100.0 | 78.8 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 2 | 18 | 100.0 | 0.0 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 2 | 19 | 100.0 | 10.5 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 3 | 20 | 100.0 | 0.2 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 3 | 21 | 100.0 | 20.8 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 4 | 22 | 100.0 | 0.9 | not applicable | 100.00 | 0.00 |
| `solid-key-r0` | 4 | 23 | 100.0 | 80.8 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 1 | 16 | 100.0 | 1.2 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 1 | 17 | 100.0 | 78.1 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 2 | 18 | 100.0 | 0.2 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 2 | 19 | 100.0 | 50.2 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 3 | 20 | 100.0 | 0.1 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 3 | 21 | 100.0 | 14.3 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 4 | 22 | 100.0 | 0.1 | not applicable | 100.00 | 0.00 |
| `solid-key-r1` | 4 | 23 | 100.0 | 62.4 | not applicable | 100.00 | 0.00 |

### Floor-1 novelty each round (trusted-piece teacher; problems with no carry at cut 1 only; percent)

| Run | Round | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r0` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 |
| `solid-key-r0` | 3 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r0` | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r1` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r1` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r1` | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid-key-r1` | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Stage B2: agreement rule, floor 3

**Gate B2: not passed** (floor 3 not run).

Not run (or no repeat finished round 4).

### Accuracy per length and repeat (percent)

*3-7 and floor 1 (8-15)*

| Run | When | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

*floor 2 (16-23)*

| Run | When | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

*floor 3 (24-31)*

| Run | When | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Floor 2 before floor 3's first round (logged only, no gate)

No-carry rule on split 1, current model, 2,000 fresh problems per length; percent.

| Run | 16: novelty / confirmed | 17: novelty / confirmed | 18: novelty / confirmed | 19: novelty / confirmed | 20: novelty / confirmed | 21: novelty / confirmed | 22: novelty / confirmed | 23: novelty / confirmed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Checks at the new lengths, each round (the condition's own teacher)

| Run | Round | Digits | Kept share | Novelty | Confirmed share | Kept right (audit) | Self-taught pieces replaced |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Floor-1 novelty each round (trusted-piece teacher; problems with no carry at cut 1 only; percent)

| Run | Round | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Answer-key control, floor 3

Not run.

## Stage B3: agreement rule, floor 4

**Gate B3: not passed** (floor 4 not run).

Not run (or no repeat finished round 4).

### Accuracy per length and repeat (percent)

*3-7 and floor 1 (8-15)*

| Run | When | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

*floor 2 (16-23)*

| Run | When | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

*floor 3 (24-31)*

| Run | When | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

*floor 4 (32-39)*

| Run | When | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Floor 3 before floor 4's first round (logged only, no gate)

No-carry rule on split 1, current model, 2,000 fresh problems per length; percent.

| Run | 24: novelty / confirmed | 25: novelty / confirmed | 26: novelty / confirmed | 27: novelty / confirmed | 28: novelty / confirmed | 29: novelty / confirmed | 30: novelty / confirmed | 31: novelty / confirmed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Checks at the new lengths, each round (the condition's own teacher)

| Run | Round | Digits | Kept share | Novelty | Confirmed share | Kept right (audit) | Self-taught pieces replaced |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Floor-1 novelty each round (trusted-piece teacher; problems with no carry at cut 1 only; percent)

| Run | Round | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Answer-key control, floor 4

Not run.

## Stage C (iii): under-trained floor 1

Not run.
## Stage C: solid / no-carry, floor 2

Not run.

## Stage C: corrupted / agreement, floor 2

Not run.

## Stage C: corrupted / no-carry, floor 2

Not run.

## Stage C: under-trained / agreement, floor 2

Not run.

## Stage C: under-trained / no-carry, floor 2

Not run.

## Stage C: under-trained / answer key, floor 2

Not run.

## Seen coming

The two numbers, computed with the no-carry rule on the floor-1 model before any floor-2 training (2,000 fresh problems per length; with corrupted pieces the corruption is applied to the confirmed share; floor-1 novelty uses pieces of 3-7 digits, which are never corrupted, and covers only problems with no carry at cut 1). Percent.

| Version | Run | Confirmed 16 | Confirmed 17 | Novelty 8 | Novelty 9 | Novelty 10 | Novelty 11 | Novelty 12 | Novelty 13 | Novelty 14 | Novelty 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solid | `solid-agr-r0` | 62.7 | 64.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| solid | `solid-agr-r1` | 65.5 | 63.4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| solid | `solid-agr-r2` | 63.3 | 64.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


*Strict* (the spec's rule): every repeat of the condition lies beyond every solid repeat (lower confirmed share; for novelty, higher at one or more floor-1 lengths). *Loose*: the condition's mean lies beyond the solid mean. "Seen coming" applies only to a condition where floor 2 fails.

## Examples and updates per round

| Run | Floor | Round | Examples | Updates | Kept per new length | Kept right (%) | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `solid-key-r0` | 2 | 1 | 125,000 | 3,500 | 16: 10,000 of 10,000, 17: 10,000 of 10,000 | 100.00, 100.00 | 7.1 |
| `solid-key-r0` | 2 | 2 | 145,000 | 4,000 | 18: 10,000 of 10,000, 19: 10,000 of 10,000 | 100.00, 100.00 | 8.3 |
| `solid-key-r0` | 2 | 3 | 165,000 | 4,600 | 20: 10,000 of 10,000, 21: 10,000 of 10,000 | 100.00, 100.00 | 10.0 |
| `solid-key-r0` | 2 | 4 | 185,000 | 5,100 | 22: 10,000 of 10,000, 23: 10,000 of 10,000 | 100.00, 100.00 | 11.7 |
| `solid-key-r1` | 2 | 1 | 125,000 | 3,500 | 16: 10,000 of 10,000, 17: 10,000 of 10,000 | 100.00, 100.00 | 6.7 |
| `solid-key-r1` | 2 | 2 | 145,000 | 4,000 | 18: 10,000 of 10,000, 19: 10,000 of 10,000 | 100.00, 100.00 | 8.0 |
| `solid-key-r1` | 2 | 3 | 165,000 | 4,600 | 20: 10,000 of 10,000, 21: 10,000 of 10,000 | 100.00, 100.00 | 9.5 |
| `solid-key-r1` | 2 | 4 | 185,000 | 5,100 | 22: 10,000 of 10,000, 23: 10,000 of 10,000 | 100.00, 100.00 | 11.0 |
| `solid-agr-r0` | 2 | 1 | 125,000 | 3,500 | 16: 10,000 of 29,222, 17: 10,000 of 28,879 | 100.00, 100.00 | 10.1 |
| `solid-agr-r0` | 2 | 2 | 145,000 | 4,000 | 18: 10,000 of 28,816, 19: 10,000 of 29,108 | 99.99, 99.96 | 11.8 |
| `solid-agr-r0` | 2 | 3 | 165,000 | 4,600 | 20: 10,000 of 29,244, 21: 10,000 of 28,942 | 100.00, 99.99 | 14.0 |
| `solid-agr-r0` | 2 | 4 | 185,000 | 5,100 | 22: 10,000 of 28,731, 23: 10,000 of 29,301 | 99.98, 99.98 | 16.3 |
| `solid-agr-r1` | 2 | 1 | 125,000 | 3,500 | 16: 10,000 of 28,905, 17: 10,000 of 28,950 | 100.00, 100.00 | 10.1 |
| `solid-agr-r1` | 2 | 2 | 145,000 | 4,000 | 18: 10,000 of 29,250, 19: 10,000 of 28,921 | 100.00, 99.94 | 11.8 |
| `solid-agr-r1` | 2 | 3 | 165,000 | 4,600 | 20: 10,000 of 29,070, 21: 10,000 of 28,883 | 99.96, 99.92 | 13.9 |
| `solid-agr-r1` | 2 | 4 | 185,000 | 5,100 | 22: 10,000 of 28,871, 23: 10,000 of 29,125 | 100.00, 99.96 | 16.2 |
| `solid-agr-r2` | 2 | 1 | 125,000 | 3,500 | 16: 10,000 of 28,964, 17: 10,000 of 29,312 | 100.00, 99.99 | 9.6 |
| `solid-agr-r2` | 2 | 2 | 145,000 | 4,000 | 18: 10,000 of 28,956, 19: 10,000 of 28,926 | 99.95, 99.94 | 11.3 |
| `solid-agr-r2` | 2 | 3 | 165,000 | 4,600 | 20: 10,000 of 28,956, 21: 10,000 of 29,188 | 99.99, 99.99 | 13.3 |
| `solid-agr-r2` | 2 | 4 | 185,000 | 5,100 | 22: 10,000 of 29,161, 23: 10,000 of 29,196 | 99.98, 99.96 | 15.4 |

## Dollars (logged GPU wall seconds x $1.10/h)

| Stage | Runs | Logged GPU-hours | Dollars |
| --- | --- | --- | --- |
| Timing and rehearsal set | 2 | 0.07 | 0.07 |
| A (answer key, floor 2 rounds 1-2) | 2 | 0.52 | 0.57 |
| B (agreement floor 2; answer key rounds 3-4) | 5 | 3.39 | 3.73 |
| B2 (agreement floor 3) | 3 | 0.00 | 0.00 |
| B3 (agreement floor 4) | 3 | 0.00 | 0.00 |
| **Pilot total** | | 3.98 | **4.37** |

## Rehearsal set audit

Built once from `/vol/runs/si-nocarry-seed2-authors/round_4/checkpoints/step_003000.pt` with the no-carry rule (pieces of 3-7 digits); 80,000 rows, sha256 `7becd3c558576b97...`. Not filtered: **6 of 80,000 answers are wrong**.

| Digits | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kept | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 |
| Drawn | 25,978 | 25,894 | 26,098 | 26,010 | 25,871 | 26,381 | 26,028 | 66,975 |
| Wrong | 0 | 0 | 0 | 1 | 1 | 2 | 2 | 0 |

