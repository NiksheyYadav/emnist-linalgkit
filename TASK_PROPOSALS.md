# Task Proposals from Codebase Review

## 1) Typo task: Fix the Quick Start code block language tag in `README.md`
- **Issue found:** The `Quick Start` fenced block is tagged as `python`, but it includes shell commands (`python train.py ...`).
- **Why this matters:** Syntax highlighting is misleading and copy/paste guidance is less clear for readers.
- **Proposed task:** Split the section into a Python example block and a Bash block (or retag the current block to `bash`).
- **Acceptance criteria:** The Quick Start section clearly separates runnable Python code from CLI commands.

## 2) Bug task: Correct `BinaryCrossEntropyLoss.backward` batch-size scaling for 1D predictions
- **Issue found:** In `nn/losses.py`, `BinaryCrossEntropyLoss.backward` sets `batch_size = 1` whenever predictions are 1D. This over-scales gradients for batch-shaped vectors `(N,)`.
- **Why this matters:** Training behavior depends on tensor shape conventions, and the same data can produce different gradient magnitudes.
- **Proposed task:** Use `self.predictions.shape[0]` for both `(N,)` and `(N, 1)` cases (and keep scalar handling explicit if needed).
- **Acceptance criteria:** Gradients from `(N,)` and `(N,1)` prediction formats are numerically equivalent after shape normalization.

## 3) Documentation discrepancy task: Align README claims about dependencies with actual optional packages
- **Issue found:** `README.md` says dependencies are "NumPy only", but the repository also relies on `LinAlgKit` and optionally uses `requests`/`matplotlib` depending on workflow.
- **Why this matters:** New users can misunderstand install/runtime expectations and hit import errors.
- **Proposed task:** Update the feature/dependency table and installation notes to distinguish required vs optional dependencies.
- **Acceptance criteria:** README dependency statements match actual imports and `requirements.txt`.

## 4) Test improvement task: Replace `test_nn.py` print-based smoke test with assertion-based tests
- **Issue found:** `test_nn.py` prints status messages and always ends with "All tests passed!" without assertions.
- **Why this matters:** Failures can go undetected in CI because success is based on script completion, not validated outcomes.
- **Proposed task:** Convert to `pytest` tests with assertions for output shape, probability normalization, gradient existence, and parameter updates.
- **Acceptance criteria:** Tests fail on regression, pass on correct behavior, and can be run via a single command (e.g., `pytest`).
