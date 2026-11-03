# LLM and Inference Refresher

A reference sheet for the concepts behind this project: definitions, formulas, worked examples and short interview answers. It grows as each lesson is completed.

Exercises and progress are in [`learning-log.md`](learning-log.md).

## Contents

0. [The 30-second overview](#0-the-30-second-overview)
1. [Tensors and shapes](#1-tensors-and-shapes)
2. [Matrix multiplication](#2-matrix-multiplication)
3. [Linear layers](#3-linear-layers)
4. [FLOPs and compute cost](#4-flops-and-compute-cost)
5. [Interview quick answers](#5-interview-quick-answers)
6. [Common pitfalls](#6-common-pitfalls)

_Coming next: derivatives and autograd, softmax and cross-entropy, the training loop, language models, attention._

---

## 0. The 30-second overview

> A **language model** is a function: given a sequence of tokens, it outputs a probability for every possible next token.
>
> - **Training** adjusts the function's parameters so it gives higher probability to the next token that actually appears in real text.
> - **Generation** calls the function repeatedly: pick a token, append it, repeat.
> - **Inference engineering** makes those repeated calls fast and cheap. It comes down to how many bytes move through GPU memory and how many arithmetic operations are done.

---

## 1. Tensors and shapes

### 1.1 Definition

A **tensor** is an array of numbers with a **shape**: the number of dimensions and the length of each.

| Shape | Name | Example |
|---|---|---|
| `()` | scalar | a loss value: `2.31` |
| `(3,)` | vector | one token's embedding |
| `(2, 3)` | matrix | a weight matrix |
| `(B, T, C)` | 3-D tensor | a batch of token sequences |

### 1.2 Reading a shape from brackets

```
C = [[7, 8, 9, 1],
     [2, 3, 4, 5]]        shape (2, 4)
```

1. **Leading brackets = number of dimensions.** `[` vector, `[[` matrix, `[[[` 3-D.
2. **Outer list length = first dimension.** Two inner lists → 2 rows.
3. **Inner list length = last dimension.** Four numbers per row → 4 columns.

Shape is **(rows, columns)**. Each extra level of nesting adds a dimension at the **front**.
Sanity checks: all rows are the same length, and total elements = product of the shape (2 × 4 = 8).

### 1.3 The `(B, T, C)` convention

The standard shape of activations inside a GPT:

| Letter | Meaning |
|---|---|
| `B` | batch size: sequences processed together |
| `T` | time: sequence length in tokens |
| `C` | channels: numbers per token (the model width, also called `d_model`) |

A `(B, T, C)` tensor is a list of sequences → each a list of tokens → each a list of `C` numbers.

### 1.4 GPT-2 small reference sizes

| Quantity | Value |
|---|---|
| Model width `C` (`d_model`) | 768 |
| MLP hidden width | 3072 (4 × 768) |
| Layers | 12 |
| Attention heads | 12 (64 dims each) |
| Vocabulary | 50,257 tokens |
| Max context `T` | 1024 tokens |
| Parameters | ~124M |

### 1.5 Indexing and reductions

For `x` of shape `(B, T, C)`:

| Expression | Result shape | Meaning |
|---|---|---|
| `x[0]` | `(T, C)` | first sequence in the batch |
| `x[:, -1, :]` | `(B, C)` | last token of every sequence (used in generation) |
| `x.sum(dim=1)` | `(B, C)` | sum over tokens; that dimension disappears |
| `x.sum(dim=1, keepdim=True)` | `(B, 1, C)` | same, but the dimension is kept with size 1 |
| `x.shape` | — | the shape |

**Rule:** integer indexing removes a dimension; slicing (`:`) keeps it. Reducing (sum, mean, max) over a dimension removes it unless `keepdim=True`.

---

## 2. Matrix multiplication

### 2.1 The rule

```
(m, k) @ (k, n) → (m, n)

out[i, j] = Σₚ left[i, p] · right[p, j]
```

- The **inner** sizes must match, and they disappear.
- Each output element is a **dot product** of one row of the left matrix and one column of the right.

### 2.2 Worked example

```
A = [[1, 2, 3],      B = [[1, 0],      A @ B = [[ 4,  5],
     [4, 5, 6]]           [0, 1],               [10, 11]]
                          [1, 1]]
  (2, 3)               (3, 2)               (2, 2)
```

```
out[1, 0] = [4, 5, 6] · [1, 0, 1] = 4·1 + 5·0 + 6·1 = 10
```

### 2.3 Order matters

Matrix multiplication is **not commutative**: `A @ B ≠ B @ A` in general.

- `A @ B`: `(2, 3) @ (3, 2) → (2, 2)`
- `B @ A`: `(3, 2) @ (2, 3) → (3, 3)`

Even when both orders give the same shape, the numbers usually differ.

**Transpose** `Aᵀ` swaps rows and columns: `(m, k)ᵀ → (k, m)`. It's how you make shapes line up, for example `(T, d) @ (T, d)ᵀ = (T, d) @ (d, T) → (T, T)`. This is exactly the attention-score computation `Q @ Kᵀ`.

### 2.4 Leading (batch) dimensions

When the left side has extra leading dimensions, the matmul happens over the **last** dimension only, and the leading dimensions pass through unchanged:

```
(B, T, C) @ (C, D) → (B, T, D)
```

**Worked example:** `(2, 3, 2) @ (2, 4) → (2, 3, 4)`

```
x = [                                   W = [[1, 0, 1, 2],
  [[1, 2], [3, 4], [5, 6]],   # seq 0        [0, 1, 1, 3]]
  [[7, 8], [9, 0], [1, 1]],   # seq 1
]
```

Each token is multiplied by the same `W` independently:

```
seq 0:  [1,2] → [1, 2, 3, 8]     seq 1:  [7,8] → [7, 8, 15, 38]
        [3,4] → [3, 4, 7, 18]            [9,0] → [9, 0, 9, 18]
        [5,6] → [5, 6, 11, 28]           [1,1] → [1, 1, 2, 5]
```

```
(2, 3, 2) @ (2, 4)  →  (2, 3, 4)
 B  T  C     C  D       B  T  D
```

**Flattening trick:** `(B, T, C) @ (C, D)` is the same computation as `(B·T, C) @ (C, D)`, reshaped back afterwards. GPUs run it as one large 2-D matmul. Use this to count FLOPs.

---

## 3. Linear layers

### 3.1 Definition

```
y = x @ W + b

x: (B, T, C_in)    W: (C_in, C_out)    b: (C_out,)    y: (B, T, C_out)
```

One weight matrix, applied to every token independently. It maps each token from `C_in` numbers to `C_out` numbers.

### 3.2 Parameter count

```
params = C_in · C_out + C_out      (weights + bias)
```

Example: a `768 → 3072` layer has `768 · 3072 + 3072 = 2,362,368` parameters.

### 3.3 Where it appears in GPT-2

Each transformer block's **MLP** is: `768 → 3072` (expand 4×), a nonlinearity (GELU), then `3072 → 768` (project back).

> **PyTorch note:** `nn.Linear` stores its weight as `(C_out, C_in)` and computes `x @ W.T + b`. Same maths, transposed storage. This matters when loading GPT-2 weights.

---

## 4. FLOPs and compute cost

### 4.1 Definitions

**FLOP** = **floating-point operation**: one add or one multiply on a floating-point (decimal) number.

| Term | Meaning | Example |
|---|---|---|
| **FLOPs** | a **count** of work | "this layer takes 5 billion FLOPs" |
| **FLOP/s** | a **rate**: work per second | "this GPU does ~100 trillion FLOP/s" |

**Floating-point formats** differ in bytes per number: fp32 = 4 bytes, fp16/bf16 = 2 bytes, int8 = 1 byte.

### 4.2 FLOPs of a matmul

```
FLOPs = 2 · m · n · k
```

**Derivation:** there are `m · n` output elements. Each is a dot product of length `k`: `k` multiplies and about `k` adds (strictly `k − 1`, but hardware pairs them as a **multiply-add**, so we count `2k`).

For 3-D inputs, flatten first: `(B, T, C) @ (C, D)` costs `2 · (B·T) · D · C`.

### 4.3 Rule of thumb: forward pass ≈ 2 × parameters per token

Each weight in a linear layer is used in exactly **one multiply-add per token**: 2 FLOPs. So for a model whose parameters are mostly in matmuls (true for GPTs):

```
forward FLOPs per token ≈ 2 · N_params
```

GPT-2 small: ≈ 2 × 124M ≈ 0.25 GFLOPs per token (ignoring attention-score computation, which grows with context length).

### 4.4 Predicting run time

```
time ≈ FLOPs ÷ achievable FLOP/s
```

Predict first, then measure. If the measured time is much slower than predicted, arithmetic isn't the bottleneck; usually it's memory traffic. This comparison is the basis of the **roofline** model (covered later).

---

## 5. Interview quick answers

**What's the shape rule for matrix multiplication?**
`(m, k) @ (k, n) → (m, n)`. Inner dimensions must match and are contracted away.

**How many FLOPs is a matmul?**
`2·m·n·k`: `m·n` outputs, each a length-`k` dot product of `k` multiply-adds.

**Is matrix multiplication commutative?**
No. `A @ B` and `B @ A` may not both be defined, may have different shapes, and usually differ in value even when square.

**What does a linear layer do to a `(B, T, C)` tensor?**
Applies the same `(C, D)` matrix to every token independently: `(B, T, C) → (B, T, D)`. Batch and sequence dimensions pass through.

**How many parameters does a linear layer have?**
`C_in · C_out` weights plus `C_out` biases.

**Roughly how many FLOPs does a forward pass take per token?**
About `2 × parameter count`, because each weight is used in one multiply-add per token.

**FLOPs vs FLOP/s?**
FLOPs is how much work; FLOP/s is how fast hardware does work. Dividing gives a lower bound on time.

---

## 6. Common pitfalls

- **Wrong matmul order.** `K @ Q.T` instead of `Q @ K.T` may still run (when shapes happen to fit) and silently give wrong results. Check shapes on paper.
- **Assuming 3-D @ 2-D is invalid.** It's valid; only the last dimension is contracted.
- **Forgetting `keepdim`.** A reduction drops the dimension, which can break broadcasting later.
- **`nn.Linear` weight layout.** Stored as `(out, in)`, not `(in, out)`.
- **Confusing FLOPs with FLOP/s.**
