# Qwen3 8B HLX Recognition - Manual Testing Protocol

**Experiment ID:** QWEN3_8B_HLX_001_MANUAL
**Date:** 2025-12-18
**Model:** qwen3:8b (via Helix Studio)
**Purpose:** Test whether 8B model exhibits HLX recognition (instant) vs. learning (gradual)

---

## Instructions

1. Open Helix Studio
2. Select model: **qwen3:8b**
3. Copy-paste tests below
4. Record all responses in this document
5. Compare before/after patterns

---

## Phase 1: BASELINE (Before Corpus)

### Test 1A: Determinism Test (Run 3 times)

**Prompt:**
```
Translate {14: {@0: 42}} to LC-R format. Output ONLY the LC-R translation, nothing else.
```

**Response 1:**
_(paste here)_

**Response 2:**
_(paste here)_

**Response 3:**
_(paste here)_

**Determinism Score:** Same output = 1.0, Different = 0.0
**Score:**

---

### Test 1B: Reversibility Test

**Prompt:**
```
Task: Convert this HLX structure to a collapsed handle, then resolve it back to verify perfect round-trip.

Input: {1000: {@0: "hello", @1: true}}

Step 1: Collapse to handle (format: &h_<hash>)
Step 2: Resolve handle back to original structure
Step 3: Verify exact match

Output format:
Handle: ...
Resolved: ...
Match: YES/NO
```

**Response:**
_(paste here)_

**Verdict:** PASS if perfect round-trip, FAIL otherwise
**Result:**

---

### Test 1C: Hallucination Test

**Prompt:**
```
Resolve this handle and return its value: &h_nonexistent_abc123

If the handle doesn't exist, return ERROR or null. Do NOT invent a value.
```

**Response:**
_(paste here)_

**Verdict:** PASS if returns error/null, FAIL if invents value
**Result:**

---

## Phase 2: CORPUS INJECTION

**Action:** Start a NEW conversation in Studio and load the corpus:

1. Click "New Chat" / Start fresh conversation
2. Copy the ENTIRE contents of `/home/matt/LC_R_EXAMPLES_FOR_VERIFICATION_TESTING.md`
3. Send this prompt:

```
Study these HLX/LC-R examples carefully. Understand the syntax, patterns, and rules. When you're ready, respond with ONLY the word "Ready" and nothing else.

[PASTE FULL CORPUS FILE HERE]
```

**Model Response:**
_(should say "Ready")_

**Timestamp:** _(time corpus was loaded)_

---

## Phase 3: POST-CORPUS (After Learning)

**Run the EXACT same 3 tests from Phase 1 in the same conversation (after corpus injection)**

### Test 3A: Determinism Test (Run 3 times)

**Prompt:** _(same as Test 1A)_
```
Translate {14: {@0: 42}} to LC-R format. Output ONLY the LC-R translation, nothing else.
```

**Response 1:**
_(paste here)_

**Response 2:**
_(paste here)_

**Response 3:**
_(paste here)_

**Determinism Score:**
**Improved from Phase 1?** YES / NO

---

### Test 3B: Reversibility Test

**Prompt:** _(same as Test 1B)_
```
Task: Convert this HLX structure to a collapsed handle, then resolve it back to verify perfect round-trip.

Input: {1000: {@0: "hello", @1: true}}

Step 1: Collapse to handle (format: &h_<hash>)
Step 2: Resolve handle back to original structure
Step 3: Verify exact match

Output format:
Handle: ...
Resolved: ...
Match: YES/NO
```

**Response:**
_(paste here)_

**Verdict:**
**Improved from Phase 1?** YES / NO

---

### Test 3C: Hallucination Test

**Prompt:** _(same as Test 1C)_
```
Resolve this handle and return its value: &h_nonexistent_abc123

If the handle doesn't exist, return ERROR or null. Do NOT invent a value.
```

**Response:**
_(paste here)_

**Verdict:**
**Still hallucinates?** YES / NO
**Improved from Phase 1?** YES / NO

---

## Analysis

### Improvement Summary

| Test | Baseline | Post-Corpus | Change |
|------|----------|-------------|--------|
| Determinism | | | |
| Reversibility | | | |
| Hallucination | | | |

### Pattern Observed

**Recognition Pattern (H1):**
- Immediate jump from poor to good performance
- All 3 tests improve significantly after corpus
- Outputs become deterministic
- Hallucination stops

**Learning Pattern (H0):**
- No change or gradual improvement
- Variance remains high
- Still hallucinates

### Verdict

Circle one:

**RECOGNITION** - Immediate perfect performance after corpus
**LEARNING** - Gradual or no improvement
**INCONCLUSIVE** - Mixed results

### Notes

_(any observations about model behavior, response quality, patterns, etc.)_

---

## Comparison to Grok Results

**Grok (frontier model) results:**
- Determinism: Perfect (1.0) immediately
- Reversibility: Perfect round-trips
- Hallucination: Zero after corpus
- Verdict: Clear recognition pattern

**Qwen3 8B (this experiment):**
- Determinism: ___(fill after testing)___
- Reversibility: ___(fill after testing)___
- Hallucination: ___(fill after testing)___
- Verdict: ___(fill after testing)___

---

**Time to Complete:** ~15-20 minutes
**Publishable:** Yes - record all raw responses as immutable data
