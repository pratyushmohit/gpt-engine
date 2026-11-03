# Module 0: Refresher

Foundations to relearn before starting v0.1. Each lesson is short and ends with exercises: do them by hand first, then check them in PyTorch under `learn/`.

| # | Lesson | Why it matters later | Status |
|---|---|---|---|
| 1 | [Tensors and matrix multiplication](#lesson-1-tensors-and-matrix-multiplication) | Almost all of a GPT's computation is matrix multiplication | In progress |
| 2 | Derivatives and the chain rule; build a tiny autograd by hand | Turns `loss.backward()` from magic into something you've built | — |
| 3 | Softmax, probability, cross-entropy | How a model turns numbers into "next token" predictions, and how we score it | — |
| 4 | A training loop in PyTorch | The same loop trains your GPT | — |
| 5 | A bigram language model | Your first model that generates text; then on to attention | — |

## How these lessons work

- One idea at a time, built only from ideas you already have.
- **Predict first:** write down what you expect before you run or measure anything.
- Exercises are done by hand before code. Answers are added after you attempt them.
- You write all the code in `learn/`.

## The big picture

> A language model is a function that takes a sequence of tokens and outputs a probability for every possible next token. **Training** adjusts the function's numbers (its parameters) so that it gives higher probability to the next token that actually appears in real text. **Generation** calls that function repeatedly, picking a token each time and appending it to the sequence. **Inference engineering** is the craft of making those repeated calls fast and cheap, which comes down to how many bytes you move through the GPU's memory and how many arithmetic operations you do.

Training is v0.1, generation is v0.2, and making it fast is v0.2 through v0.5.

## Open questions (answered in Lesson 3)

GPT-2 has a vocabulary of 50,257 tokens, and its output for each position is a list of 50,257 numbers.

- **(a)** What do those numbers represent?
- **(b)** If the model knew nothing and guessed uniformly, what probability would it give the correct next token?
- **(c)** Cross-entropy loss is −log(probability given to the correct token). What is the loss for that know-nothing model?

Part (c) is a number you'll see again on your first training run.

---

## Lesson 1: Tensors and matrix multiplication

### 1. What a tensor is

A **tensor** is an array of numbers with a **shape**. The shape tells you how many dimensions there are and how long each one is.

| Shape | Name | Example |
|---|---|---|
| `()` | scalar | the loss: `2.31` |
| `(3,)` | vector | one token's embedding (tiny) |
| `(2, 3)` | matrix | a weight matrix |
| `(B, T, C)` | 3-D tensor | **a batch of B sequences, each T tokens long, each token a vector of C numbers** |

**Memorise `(B, T, C)`.** You'll see that shape constantly in the GPT.

- `B`: batch size, how many sequences are processed together
- `T`: time, meaning sequence length in tokens
- `C`: channels, the size of each token's vector. For GPT-2 small, C = 768.

### 2. The matrix multiplication rule

```
(m, k) @ (k, n) → (m, n)
```

- The **inner** sizes (`k`) must match, and they disappear in the result.
- Each output number is a **dot product**: one row of the left matrix times one column of the right, multiplied element by element and summed.

```
out[i, j] = sum over p of  left[i, p] * right[p, j]
```

### 3. The cost of a matmul

Each output number needs `k` multiplies and `k` adds. There are `m × n` outputs, so:

```
FLOPs = 2 · m · n · k
```

Keep this formula. All through the project you'll use it to predict how long something should take on the GPU, and then compare that with the measured time.

### 4. Why matmul is the core of a neural network

A **linear layer** is just:

```
y = x @ W
```

If `x` is `(B, T, 768)` and `W` is `(768, 3072)`, then `y` is `(B, T, 3072)`.

The leading dimensions `(B, T)` just come along unchanged, and the multiplication happens over the **last** dimension. Each token's 768 numbers become 3072 new ones, and every token is transformed by the same `W`.

### 5. Indexing and reducing (the operations you'll use most)

For `x` of shape `(B, T, C)`:

| Expression | Meaning |
|---|---|
| `x[0]` | the first sequence in the batch |
| `x[:, -1, :]` | the last token of every sequence (generation uses exactly this) |
| `x.sum(dim=1)` | sum over the time dimension; that dimension disappears |
| `x.shape` | the shape |

General rule: **reducing over a dimension (sum, mean, max) removes that dimension** unless you pass `keepdim=True`.

### Key takeaways

1. Always know the shape of every tensor. Most bugs in this project will be shape bugs.
2. `(m, k) @ (k, n) → (m, n)`: the inner sizes match and disappear.
3. A matmul costs `2·m·n·k` FLOPs.
4. A linear layer is a matmul over the last dimension; the leading dimensions come along unchanged.

---

### Exercises (by hand first)

**1.** Compute `A @ B`. What shape is the result, and what are the four numbers?

```
A = [[1, 2, 3],        B = [[1, 0],
     [4, 5, 6]]             [0, 1],
                            [1, 1]]
```

**2.** Give the output shape, or say "invalid":

- (a) `(4, 8) @ (8, 16)`
- (b) `(8, 4) @ (8, 16)`
- (c) `(32, 10, 768) @ (768, 3072)`

**3.** How many FLOPs does `(1024, 768) @ (768, 768)` take? Round it to something like "about N billion".

**4.** `x` has shape `(2, 5, 4)`, meaning B=2, T=5, C=4. Predict each shape:

- (a) `x[0]`
- (b) `x[:, -1, :]`
- (c) `x.sum(dim=1)`

### Check in PyTorch

Install the CPU build of PyTorch (enough for Module 0):

```
uv add torch numpy
```

Then create `learn/01_tensors.py` and check each answer with `torch.tensor(...)`, `@` and `.shape`. Write the code yourself.

### Your answers

_Fill in your hand answers here before checking them in PyTorch._

### Notes and mistakes

_Anything that surprised you or that you got wrong, and why._
