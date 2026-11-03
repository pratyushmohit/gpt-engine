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

#### Reading a shape from the brackets

```
C = [[7, 8, 9, 1],
     [2, 3, 4, 5]]
```

1. **Count the opening brackets at the very start.** `[[` is two, so C has 2 dimensions: it's a matrix. One `[` is a vector, and `[[[` is a 3-D tensor.
2. **The outer list holds rows.** Count the inner lists, `[7, 8, 9, 1]` and `[2, 3, 4, 5]`. That's 2 rows.
3. **Each inner list is one row.** Count the numbers in one row: 4. That's the number of columns.

Shape is always written **(rows, columns)**, so C is **(2, 4)**.

**Checks:**
- Every row must be the same length; otherwise it isn't a valid matrix.
- The total count of numbers equals rows × columns: 2 × 4 = 8.

**Higher dimensions:** each extra level of brackets adds a dimension at the **front** of the shape. A `(B, T, C)` tensor is a list of sequences, each sequence is a list of tokens, and each token is a list of numbers.

### 2. The matrix multiplication rule

```
(m, k) @ (k, n) → (m, n)
```

- The **inner** sizes (`k`) must match, and they disappear in the result.
- Each output number is a **dot product**: one row of the left matrix times one column of the right, multiplied element by element and summed.

```
out[i, j] = sum over p of  left[i, p] * right[p, j]
```

#### Worked example (exercise 1)

```
A = [[1, 2, 3],        B = [[1, 0],        A @ B = [[ 4,  5],
     [4, 5, 6]]             [0, 1],                 [10, 11]]
                            [1, 1]]
```

One entry, step by step:

```
out[1, 0] = row 1 of A · column 0 of B
          = [4, 5, 6] · [1, 0, 1]
          = 4·1 + 5·0 + 6·1
          = 10
```

Shape check: `(2, 3) @ (3, 2)`. The inner 3s match and disappear, which leaves `(2, 2)`.

#### Order matters

`B @ A` is also allowed: `(3, 2) @ (2, 3) → (3, 3)`. But `A @ B` is `(2, 2)`, so the two results can't be equal. **Matrix multiplication is not commutative:** even when both orders give the same shape, the numbers are usually different.

This matters in attention, where you'll compute `Q @ Kᵀ` (Kᵀ is K flipped so its rows become columns). Write it in the wrong order and you get either a shape error or, worse, a wrong result that still runs. Checking shapes is how you catch it.

### 3. The cost of a matmul

#### What a FLOP is

**FLOP** stands for **floating-point operation**: one arithmetic step, such as an addition or a multiplication, on a decimal number.

- **"Floating-point"** is how computers store numbers with a decimal point, like `0.37` or `-2.5`. A neural network's weights and activations are all stored this way. Formats like fp32 and bf16 differ in how many bytes each number takes.
- **One FLOP** is one such step. `3.1 × 2.0` is one FLOP, and `6.2 + 1.5` is another.

| Term | Means | Example |
|---|---|---|
| **FLOPs** (lowercase s) | a **count**: how much work a calculation needs | "this matmul takes 24 FLOPs" |
| **FLOP/s** (per second) | a **speed**: how much work hardware does each second | "this GPU does 100 trillion FLOP/s" |

#### Why we count them

If you know both numbers, you can predict how long a calculation should take before running it:

```
time ≈ FLOPs needed ÷ FLOP/s the GPU can do
```

Then you measure the actual time. If it's much slower than predicted, something besides arithmetic is the bottleneck, usually memory. That comparison is the core of the performance analysis in this project, and it's where the roofline plots in v0.3 come from.

#### Counting the FLOPs of a matmul

Each output number is a dot product of length `k`:

```
4·1 + 5·0 + 6·1
```

- `k` multiplications (here 3)
- about `k` additions. Strictly it's `k − 1`, but by convention we count one add per multiply, because hardware does them as a pair called a **multiply-add**.

So each output number costs about `2k` FLOPs. There are `m × n` outputs, so:

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

#### Worked example with small numbers

**x has shape (2, 3, 2):** 2 sentences, each 3 tokens long, and each token is 2 numbers.

```
x = [
  [[1, 2], [3, 4], [5, 6]],    # sentence 0: three tokens
  [[7, 8], [9, 0], [1, 1]],    # sentence 1: three tokens
]
```

**W has shape (2, 4).** It turns each 2-number token into a 4-number token.

```
W = [[1, 0, 1, 2],
     [0, 1, 1, 3]]
```

**Step 1: multiply one token.** Treat `[1, 2]` as a `(1, 2)` matrix: `(1, 2) @ (2, 4) → (1, 4)`.

```
[1, 2] @ W = [1·1 + 2·0,  1·0 + 2·1,  1·1 + 2·1,  1·2 + 2·3]
           = [1, 2, 3, 8]
```

**Step 2: do the same for every token and keep each one in its place.**

```
sentence 0:  [1,2] → [1, 2, 3, 8]
             [3,4] → [3, 4, 7, 18]
             [5,6] → [5, 6, 11, 28]

sentence 1:  [7,8] → [7, 8, 15, 38]
             [9,0] → [9, 0, 9, 18]
             [1,1] → [1, 1, 2, 5]
```

**Step 3: read the shape.** It's still 2 sentences of 3 tokens each, but each token now has 4 numbers.

```
(2, 3, 2) @ (2, 4)  →  (2, 3, 4)
 ↑  ↑  ↑     ↑  ↑       ↑  ↑  ↑
 B  T  C     C  new     B  T  new
```

- `B` and `T` come along unchanged, because no sentence or token is created or removed.
- The last dimension `C` must match W's first dimension. It disappears and is replaced by W's second dimension.
- `W` is the **same** for every token. That's what a linear layer is: one weight matrix applied to every token.

#### Flattening trick

A 3-D tensor of tokens is really a stack of token vectors. `(32, 10, 768) @ (768, 3072)` is the same computation as `(320, 768) @ (768, 3072)`: flatten B and T into one dimension, multiply, and reshape back. GPUs run it that way, as one large 2-D matmul. Use this to count FLOPs for 3-D inputs.

**In GPT-2:** every transformer block's MLP expands each token from 768 to 3072 numbers (4×), then shrinks it back to 768. You'll write this layer in v0.1.

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
3. Order matters: `A @ B` is generally not `B @ A`.
4. A FLOP is one floating-point add or multiply. FLOPs is a count; FLOP/s is a speed.
5. A matmul costs `2·m·n·k` FLOPs.
6. A linear layer is a matmul over the last dimension; the leading dimensions come along unchanged.

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

**3.** Warm-up: how many FLOPs does `A @ B` from exercise 1 take? Count it from the dot products, then check it with `2 · m · n · k`.

Then: how many FLOPs does `(1024, 768) @ (768, 768)` take? Round it to something like "about N billion".

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

| Exercise | My answer | Result |
|---|---|---|
| 1 | `[[4, 5], [10, 11]]`, shape `(2, 2)` | ✅ |
| 1 follow-up: `B @ A` | allowed, shape `(3, 3)` | ✅ |
| 2a | `(4, 16)` | ✅ |
| 2b | invalid | ✅ |
| 2c | first said invalid, then `(32, 10, 3072)` | ✅ after the worked example |
| 3 | | |
| 4 | | |

### Notes and mistakes

- **2c:** I first thought a 3-D tensor couldn't be multiplied by a 2-D matrix. In fact the matmul only happens over the last dimension, and the leading `(B, T)` come along unchanged. The worked example in section 4 made it clear.
