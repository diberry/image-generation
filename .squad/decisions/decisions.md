# Decisions Log

## Current Decisions

### Morpheus Review: PR #5 — MEDIUM Memory Fixes

**Date:** 2026-03-25
**Reviewer:** Morpheus (Lead)
**Branch:** `squad/pr5-medium-memory-fixes`
**Verdict:** ✅ APPROVED — merge when ready

---

## Fix-by-Fix Findings

### Fix 1: Latents CPU Transfer (MEDIUM)

**Status: CORRECT**

- `latents = latents.cpu()` is placed **before** `del base` and the cache flush. Order is correct — the GPU pin is released before `empty_cache()` runs.
- Device transfer back uses `latents.to(device)` inline at the refiner call site, not hardcoded `"cuda"`. MPS is respected.
- The `if device in ("cuda", "mps")` guard is clean — CPU path is untouched.
- One note: `latents.to(device)` is called inline and its result passed directly to the refiner. The `latents` variable continues to hold the CPU copy until `finally` cleans it up. This is benign — CPU tensors carry negligible VRAM impact.

### Fix 2: Dynamo Cache Reset (MEDIUM)

**Status: CORRECT**

- `torch._dynamo.reset()` is inside the `finally` block. ✅
- Guarded by `device == "cuda" and hasattr(torch, "_dynamo")`. Both guards are present and necessary.
  - `device == "cuda"` is correct: `torch.compile` is only applied on CUDA in `load_base()` (confirmed at line 72–75).
  - `hasattr(torch, "_dynamo")` protects against torch versions without the attribute.
- The comment correctly flags that if `torch.compile` is ever extended to MPS, the guard must be broadened.

### Fix 3: Entry-Point VRAM Flush (MEDIUM)

**Status: CORRECT**

- All three calls present: `gc.collect()`, `torch.cuda.empty_cache()` (CUDA-guarded), `torch.mps.empty_cache()` (MPS-guarded).
- Located at the very start of `generate()`, before `load_base()` is called. Order confirmed in diff and test call-log tracking.
- Two-flush pattern (entry + finally) is intentional and correct. Entry reclaims fragmented VRAM from prior runs; finally unconditionally cleans up after this run.

### Fix 4: Global State Audit (MEDIUM — no code change)

**Status: CLEAN — agree with audit**

Scanned `generate.py` manually. All pipeline variables (`base`, `refiner`, `latents`, `text_encoder_2`, `vae`) are locals initialized inside `generate()`, not module-level. Zero process-persistent pipeline references. Clean architecture for a CLI tool called in batch.

---

## Test Review (9 new MEDIUM tests)

**All 22 tests pass (verified by running `PYTHONPATH=. pytest tests/test_memory_cleanup.py`).**

### Ordering / call-log tracking

- `TestEntryPointVRAMFlush`: All 3 tests use `call_log` + `side_effect` to verify the flush fires **before** `load_base`. Genuine regression guards. ✅
- `TestLatentsTensorHandling.test_latents_not_holding_gpu_ref_during_cache_flush`: Uses `call_log` to verify `latents.cpu()` precedes `mps.empty_cache()`. ✅
- `TestLatentsTensorHandling.test_latents_transferred_back_for_refiner`: Checks that `cpu()` and `to("mps")` are both called, but does NOT verify order. If someone swapped them, this test would still pass. Minor gap — not a blocker.
- `TestDynamoCacheReset`: CUDA test mocks the full `torch` module, correctly exercises the `hasattr` guard. CPU test correctly confirms `_dynamo.reset()` is NOT called on non-CUDA paths. ✅
- `TestGlobalState.test_repeated_calls_dont_compound_memory`: Tracks cumulative gc count across two `generate()` calls — verifies entry flush fires on the second call too. ✅

### Code smell: orphaned assert messages

Several tests use this pattern:
```python
mock.assert_called(), "this message is never shown"
```
The string after the comma is a valid Python expression but is silently orphaned — it's NOT a pytest `assert` statement. When the mock assertion fails, the custom message does NOT appear in the output. The underlying `assert_called()` still raises, so the tests catch regressions, but debugging is harder. Three affected tests: `test_latents_transferred_back_for_refiner`, `test_dynamo_reset_called_after_cleanup`, and `test_base_freed_before_refiner_loads`. **Not a blocker for merge, but Neo should fix this as a follow-up.**

---

## Verdict

**APPROVED.** All four MEDIUM-severity issues are correctly addressed. Code logic is sound. The 9 new tests use call-order tracking and would catch regressions on Fixes 1, 2, and 3. Fix 4 is verified clean by inspection.

**Follow-up (not blocking):** Neo to fix the orphaned message pattern in 3 tests and add call-ordering assertion to `test_latents_transferred_back_for_refiner`.
