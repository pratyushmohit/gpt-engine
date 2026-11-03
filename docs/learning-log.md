# Learning Log

Exercises, answers and mistakes while working through the foundations. The reference material is in [`refresher.md`](refresher.md).

## Module 0: Refresher

| # | Lesson | Why it matters later | Status |
|---|---|---|---|
| 1 | Tensors and matrix multiplication | Almost all of a GPT's computation is matrix multiplication | In progress |
| 2 | Derivatives and the chain rule; build a tiny autograd by hand | Turns `loss.backward()` from magic into something you've built | — |
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

**3.** Warm-up: how many FLOPs does `A @ B` from exercise 1 take? Count it from the dot products, then check with `2 · m · n · k`.
Then: how many FLOPs does `(1024, 768) @ (768, 768)` take? Round to "about N billion".

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
| 3 | | |
| 4 | | |

### Mistakes and notes

- **2c:** I first thought a 3-D tensor couldn't be multiplied by a 2-D matrix. The matmul only happens over the last dimension; the leading `(B, T)` pass through unchanged.

---

## Open questions (answered in Lesson 3)

GPT-2 has a vocabulary of 50,257 tokens, and its output for each position is a list of 50,257 numbers.

- **(a)** What do those numbers represent?
- **(b)** If the model knew nothing and guessed uniformly, what probability would it give the correct next token?
- **(c)** Cross-entropy loss is −log(probability of the correct token). What is the loss for that know-nothing model?
