# Learning Log

Exercises, answers and mistakes while working through the foundations. The reference material is in [`refresher.md`](refresher.md).

## Module 0: Refresher

| # | Lesson | Why it matters later | Status |
|---|---|---|---|
| 1 | Tensors and matrix multiplication | Almost all of a GPT's computation is matrix multiplication | ✅ Done |
| 2 | Derivatives and the chain rule; build a tiny autograd by hand | Turns `loss.backward()` from magic into something you've built | Next |
| 3 | Softmax, probability, cross-entropy | How a model turns numbers into "next token" predictions, and how we score it | — |
| 4 | A training loop in PyTorch | The same loop trains your GPT | — |
| 5 | A bigram language model | Your first model that generates text; then on to attention | — |

**How it works:** predict first, do exercises by hand before code, then check them in PyTorch under `learn/`.

---

## Lesson 1: Tensors and matrix multiplication

### Exercises

**1.** Compute `A @ B`, and give its shape.

```
A = [[1, 2, 3],        B = [[1, 0],
     [4, 5, 6]]             [0, 1],
                            [1, 1]]
```

Follow-up: is `B @ A` allowed? What shape? Is it the same as `A @ B`?

**2.** Give the output shape, or say "invalid":

- (a) `(4, 8) @ (8, 16)`
- (b) `(8, 4) @ (8, 16)`
- (c) `(32, 10, 768) @ (768, 3072)`

**3.** For each matmul, first label m, k and n (`(m, k) @ (k, n)`), then compute `2 · m · n · k`.

- (a) Warm-up: `A @ B` from exercise 1, `(2, 3) @ (3, 2)`. Also count it directly from the dot products and check both agree.
- (b) `(1024, 768) @ (768, 768)`. 768 appears twice; only one of them is k. Round to "about N billion".

**4.** `x` has shape `(2, 5, 4)`. Predict the shape of:

- (a) `x[0]`
- (b) `x[:, -1, :]`
- (c) `x.sum(dim=1)`

**Check in PyTorch:** `uv add torch numpy`, then write `learn/01_tensors.py` yourself.

### My answers

| Exercise | Answer | Result |
|---|---|---|
| 1 | `[[4, 5], [10, 11]]`, shape `(2, 2)` | ✅ |
| 1 follow-up | allowed, shape `(3, 3)`, so not equal | ✅ |
| 2a | `(4, 16)` | ✅ |
| 2b | invalid | ✅ |
| 2c | first said invalid, then `(32, 10, 3072)` | ✅ after worked example |
| 3a | 24 FLOPs (m=2, k=3, n=2) | ✅ |
| 3b | ≈ 1.2 billion FLOPs (m=1024, k=768, n=768) | ✅ |
| 4a `x[0]` | first said `(2, ?)`, then `(5, 4)` | ✅ after hint |
| 4b `x[:, -1, :]` | `(2, 4)` | ✅ after explanation of the syntax |
| 4c `x.sum(dim=1)` | `(2, 4)` | ✅ |

### Mistakes and notes

- **2c:** I first thought a 3-D tensor couldn't be multiplied by a 2-D matrix. The matmul only happens over the last dimension; the leading `(B, T)` pass through unchanged.
- **FLOPs formula:** `2·m·n·k` was unclear until the letters were tied to the shapes:
  - `m` = rows of the output = first dim of the **left** matrix
  - `k` = the **shared** inner dim = length of each dot product
  - `n` = columns of the output = last dim of the **right** matrix
  - `m · n` outputs × `2k` FLOPs each. In a linear layer: m = tokens (B·T), k = C_in, n = C_out.
  - See refresher §4.2.
- **4a:** I thought `x[0]` kept the 2. Indexing with a single number *picks one item* and removes that dimension, like `names[0]` on a Python list.
- **4b:** The syntax `x[:, -1, :]` was new: one instruction per dimension, comma-separated. `:` keeps a dimension; a number removes it.
- **4b vs 4c:** both give `(2, 4)` but mean different things: the *last* token vs the *sum* of all tokens. Same shape does not mean same values, so shape checks alone can't catch every bug.
- **dtype:** `torch.tensor([1, 2])` is `int64`; `torch.tensor([1.0, 2.0])` is `float32`.

---

## Open questions (answered in Lesson 3)

GPT-2 has a vocabulary of 50,257 tokens, and its output for each position is a list of 50,257 numbers.

- **(a)** What do those numbers represent?
- **(b)** If the model knew nothing and guessed uniformly, what probability would it give the correct next token?
- **(c)** Cross-entropy loss is −log(probability of the correct token). What is the loss for that know-nothing model?
