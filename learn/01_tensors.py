"""Lesson 1: tensors and matrix multiplication.

A runnable reference for docs/refresher.md sections 1-4, plus the checks for the
Lesson 1 exercises in docs/learning-log.md.

Run:  uv run learn/01_tensors.py
"""

import torch

torch.manual_seed(0)  # same random numbers every run
print(f"PyTorch version: {torch.__version__}\n")


def check(name, prediction, actual):
    """Compare a predicted shape with the real one."""
    actual = tuple(actual)
    status = "OK" if tuple(prediction) == actual else "WRONG"
    print(f"  {name:<30} actual {str(actual):<16} predicted {str(tuple(prediction)):<16} {status}")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =============================================================================
# 1. Tensors, shapes and dtypes                                (refresher §1)
# =============================================================================
section("1. Tensors, shapes and dtypes")

scalar = torch.tensor(2.31)                    # shape ()      a loss value
vector = torch.tensor([0.1, 0.2, 0.3])         # shape (3,)    one token's embedding
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)  rows, columns
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"  {name:<7} shape {str(tuple(t.shape)):<8} ndim {t.ndim}")

# Reading shape from brackets: [[ = 2 dims, 2 inner lists = 2 rows, 4 numbers each = 4 columns.
C = torch.tensor([[7, 8, 9, 1], [2, 3, 4, 5]])
print(f"  C shape {tuple(C.shape)}, total elements {C.numel()} (= 2 x 4)")

# dtype: whole numbers -> int64 (8 bytes); decimals -> float32 (4 bytes).
# Neural networks use floating point; later we use bfloat16 (2 bytes) to halve memory traffic.
for dt_name, t in [("int", torch.tensor([1, 2])),
                   ("float", torch.tensor([1.0, 2.0])),
                   ("bf16", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))]:
    print(f"  {dt_name:<6} dtype {str(t.dtype):<15} bytes per element {t.element_size()}")

# The (B, T, C) convention: B sequences, T tokens each, C numbers per token.
B_, T_, C_ = 2, 5, 4
x = torch.randn(B_, T_, C_)
print(f"  activations x: shape {tuple(x.shape)}  (B={B_} sequences, T={T_} tokens, C={C_} numbers per token)")


# =============================================================================
# 2. Matrix multiplication                                      (refresher §2)
# =============================================================================
section("2. Matrix multiplication: (m, k) @ (k, n) -> (m, n)")

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])     # (2, 3)
B = torch.tensor([[1, 0],
                  [0, 1],
                  [1, 1]])        # (3, 2)

print("  A @ B =\n", A @ B)
check("A @ B", (2, 2), (A @ B).shape)

# One output element = dot product of a row of A and a column of B.
row1_dot_col0 = (A[1] * B[:, 0]).sum()   # [4,5,6] . [1,0,1] = 10
print(f"  out[1, 0] by hand = {row1_dot_col0.item()}, from A @ B = {(A @ B)[1, 0].item()}")

# Order matters: B @ A has a different shape, so it cannot equal A @ B.
check("B @ A", (3, 3), (B @ A).shape)

# Transpose swaps rows and columns: (m, k) -> (k, m). Used to make shapes line up (Q @ K.T).
Q = torch.randn(5, 4)   # (T, d)
K = torch.randn(5, 4)   # (T, d)
check("Q @ K.T  (T,d)@(d,T)", (5, 5), (Q @ K.T).shape)
try:
    Q @ K               # (5, 4) @ (5, 4): inner sizes 4 and 5 don't match
except RuntimeError as e:
    print(f"  Q @ K   fails: {e}")

# Leading dimensions pass through; only the LAST dim is contracted.
x_small = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]],
                        [[7., 8.], [9., 0.], [1., 1.]]])   # (2, 3, 2)
W_small = torch.tensor([[1., 0., 1., 2.],
                        [0., 1., 1., 3.]])                 # (2, 4)
y_small = x_small @ W_small
check("(2,3,2) @ (2,4)", (2, 3, 4), y_small.shape)
print("  first token [1, 2] @ W =", y_small[0, 0].tolist(), " (expected [1, 2, 3, 8])")

# Flattening trick: (B, T, C) @ (C, D) == reshape to (B*T, C), multiply, reshape back.
W = torch.randn(4, 6)
batched = x @ W                                             # (2, 5, 6)
flat = (x.reshape(B_ * T_, C_) @ W).reshape(B_, T_, 6)      # (10, 4) @ (4, 6) -> (2, 5, 6)
print(f"  flattened == batched: {torch.allclose(batched, flat)}")


# =============================================================================
# 3. Linear layers                                              (refresher §3)
# =============================================================================
section("3. Linear layers: y = x @ W + b")

C_in, C_out = 768, 3072
layer = torch.nn.Linear(C_in, C_out)
n_params = sum(p.numel() for p in layer.parameters())
print(f"  params: {n_params:,}  (= {C_in}*{C_out} + {C_out} = {C_in * C_out + C_out:,})")

# nn.Linear stores weight as (C_out, C_in) and computes x @ W.T + b.
print(f"  layer.weight shape {tuple(layer.weight.shape)}  <- (out, in), NOT (in, out)")
x_tokens = torch.randn(2, 10, C_in)
manual = x_tokens @ layer.weight.T + layer.bias
print(f"  layer(x) == x @ W.T + b: {torch.allclose(layer(x_tokens), manual, atol=1e-5)}")
check("Linear (2,10,768)", (2, 10, 3072), layer(x_tokens).shape)


# =============================================================================
# 4. FLOPs                                                      (refresher §4)
# =============================================================================
section("4. FLOPs = 2 * m * n * k")


def matmul_flops(m, k, n):
    """(m, k) @ (k, n): m*n outputs, each a length-k dot product of k multiply-adds."""
    return 2 * m * n * k


print(f"  (2, 3) @ (3, 2):          {matmul_flops(2, 3, 2):,} FLOPs")
print(f"  (1024, 768) @ (768, 768): {matmul_flops(1024, 768, 768):,} FLOPs (~1.2 billion)")
# 3-D input: flatten first. (32, 10, 768) @ (768, 3072) -> m = 32*10 tokens.
print(f"  (32, 10, 768) @ (768, 3072): {matmul_flops(32 * 10, 768, 3072):,} FLOPs")

# Predicting time: time ~= FLOPs / achievable FLOP/s (assumed ~100 TFLOP/s, to be measured).
assumed_flops_per_s = 100e12
t_us = matmul_flops(1024, 768, 768) / assumed_flops_per_s * 1e6
print(f"  predicted time for the 1.2 GFLOP matmul at 100 TFLOP/s: ~{t_us:.0f} microseconds")


# =============================================================================
# 5. Indexing and reductions                                  (refresher §1.5)
# =============================================================================
section("5. Indexing and reductions on x (2, 5, 4)")

# One instruction per dimension:  ':' keeps a dimension, a single number removes it.
check("x[0]            first sequence", (5, 4), x[0].shape)
check("x[:, -1, :]     last token", (2, 4), x[:, -1, :].shape)
check("x[:, 1:3, :]    tokens 1-2", (2, 2, 4), x[:, 1:3, :].shape)
check("x.sum(dim=1)    sum over tokens", (2, 4), x.sum(dim=1).shape)
check("x.sum(dim=1, keepdim=True)", (2, 1, 4), x.sum(dim=1, keepdim=True).shape)
check("x.mean(dim=-1)  mean over C", (2, 5), x.mean(dim=-1).shape)

# Same shape, different values: last token vs sum of all tokens.
same = torch.allclose(x[:, -1, :], x.sum(dim=1))
print(f"  x[:, -1, :] and x.sum(dim=1) both (2, 4); equal values? {same}")

# Worked example from the refresher.
print("  x_small[:, -1, :]  =", x_small[:, -1, :].tolist())    # [[5, 6], [1, 1]]
print("  x_small.sum(dim=1) =", x_small.sum(dim=1).tolist())   # [[9, 12], [17, 9]]


# =============================================================================
# Exercises (docs/learning-log.md, Lesson 1)
# =============================================================================
section("Exercises")

print(" Exercise 2")
check("(a) (4,8) @ (8,16)", (4, 16), (torch.randn(4, 8) @ torch.randn(8, 16)).shape)
try:
    torch.randn(8, 4) @ torch.randn(8, 16)
    print("  (b) ran?! expected an error")
except RuntimeError as e:
    print(f"  (b) (8,4) @ (8,16) invalid: {e}")
check("(c) (32,10,768) @ (768,3072)", (32, 10, 3072),
      (torch.randn(32, 10, 768) @ torch.randn(768, 3072)).shape)

print(" Exercise 4 (x has shape (2, 5, 4))")
check("(a) x[0]", (5, 4), x[0].shape)
check("(b) x[:, -1, :]", (2, 4), x[:, -1, :].shape)
check("(c) x.sum(dim=1)", (2, 4), x.sum(dim=1).shape)
