# Qwen3 8B HLX Recognition Experiment

**Experiment ID:** QWEN3_8B_HLX_001
**Date:** 2025-12-18
**Status:** READY FOR EXECUTION
**Hypothesis:** 8B models exhibit HLX recognition (instant fluency) not learning (gradual improvement)

---

## Research Question

Does an 8B parameter model (Qwen3) show the same "instant recognition" pattern for HLX that frontier models (Grok) demonstrated, or does smaller model size require actual learning?

---

## Hypothesis

**H0 (Null):** Qwen3 8B requires learning - performance improves gradually with corpus exposure
**H1 (Alternative):** Qwen3 8B exhibits recognition - performance jumps immediately after corpus exposure

**Prediction (if H1 true):**
- Baseline performance: Poor (<30% axiom compliance)
- Post-corpus performance: High (>80% axiom compliance)
- No gradual improvement curve
- Deterministic outputs (low variance)
- Zero hallucination on resolved contracts

---

## Experimental Design

### Phase 0: Environment Setup
1. Verify Qwen3 8B via Ollama
2. Capture system specs
3. Record git commit hash
4. Create baseline test corpus (held-out from training examples)

### Phase 1: Baseline Measurement (Before Corpus)
**Goal:** Establish pre-exposure performance

**Tests:**
1. **Axiom Compliance (A1-A4)**
   - A1 Determinism: 10 runs of identical input
   - A2 Reversibility: Collapse/resolve round-trip
   - A3 Bijection: HLXL ↔ LC-R translation
   - A4 Universal Value: Format convergence

2. **Format Recognition**
   - Show 5 LC-R examples, ask model to identify pattern
   - Show 5 HLXL examples, ask model to translate to LC-R
   - Measure: Can it parse glyphs? Can it generate correct glyphs?

3. **Determinism Score**
   - Same prompt, 100 iterations
   - Measure: Variance in outputs

4. **Hallucination Rate**
   - Present 10 contracts with unresolved handles
   - Correct response: Return error or null
   - Incorrect response: Invent values
   - Measure: % hallucinated responses

5. **Stress Test**
   - Max nesting depth before failure
   - Largest structure before approximation

**Expected Baseline (if H1 true):** Poor performance, high variance, frequent hallucination

### Phase 2: Corpus Injection
**Corpus:** `/home/matt/LC_R_EXAMPLES_FOR_GROK.md` (400+ examples)

**Injection Method:**
```
Option A: Context window (8K tokens, fits easily)
Option B: System prompt prefix
Option C: Few-shot examples in conversation history
```

**Procedure:**
1. Load full corpus into context
2. No explicit instruction ("learn this")
3. Just exposure ("here are examples")
4. Wait 30 seconds (model processes context)

### Phase 3: Post-Exposure Measurement (After Corpus)
**Goal:** Measure immediate post-exposure performance

**Run identical tests from Phase 1:**
- Same axiom tests
- Same format recognition tests
- Same determinism measurement
- Same hallucination test
- Same stress test

**Expected Results (if H1 true):**
- Immediate jump in performance (>80% axiom compliance)
- Low variance (deterministic outputs)
- Zero hallucination
- Handles edge cases correctly

### Phase 4: Temporal Stability
**Goal:** Verify recognition persists without decay

**Procedure:**
1. Wait 5 minutes after corpus injection
2. Run tests again (no corpus re-injection)
3. Wait 30 minutes, run again
4. Compare: Does performance decay? (learning) or persist? (recognition)

**Expected (if H1 true):** Performance persists without decay

### Phase 5: Comparative Analysis
**Goal:** Compare Qwen3 8B to Grok (frontier model)

**Metrics:**
- Axiom compliance delta (before vs. after)
- Time to fluency (immediate vs. gradual)
- Hallucination rate comparison
- Determinism comparison
- Stress test comparison

---

## Success Criteria

**Experiment succeeds if:**
1. Clear before/after difference (>50% improvement)
2. Immediate jump (not gradual curve)
3. Deterministic outputs post-exposure
4. Zero hallucination post-exposure
5. Performance persists without decay

**H1 confirmed if:**
- All 5 success criteria met
- Pattern matches Grok's recognition behavior

**H0 confirmed if:**
- Gradual improvement (learning curve)
- High variance persists
- Performance decays without re-exposure

---

## Measurement Protocol

### Axiom A1 (Determinism)
```python
def test_a1_determinism(model, prompt, iterations=100):
    outputs = []
    for i in range(iterations):
        response = model.generate(prompt, temperature=0.0)
        outputs.append(response)

    unique_outputs = len(set(outputs))
    determinism_score = 1.0 - (unique_outputs - 1) / iterations
    return {
        "score": determinism_score,
        "unique_outputs": unique_outputs,
        "total_iterations": iterations,
        "verdict": "PASS" if determinism_score > 0.95 else "FAIL"
    }
```

### Axiom A2 (Reversibility)
```python
def test_a2_reversibility(model, test_structures):
    failures = []
    for struct in test_structures:
        # Collapse to handle
        handle = model.collapse(struct)
        # Resolve from handle
        recovered = model.resolve(handle)
        # Check perfect fidelity
        if not deep_equal(struct, recovered):
            failures.append({
                "original": struct,
                "handle": handle,
                "recovered": recovered
            })

    reversibility_score = 1.0 - len(failures) / len(test_structures)
    return {
        "score": reversibility_score,
        "failures": failures,
        "verdict": "PASS" if reversibility_score == 1.0 else "FAIL"
    }
```

### Axiom A3 (Bijection)
```python
def test_a3_bijection(model, hlxl_corpus):
    failures = []
    for hlxl_code in hlxl_corpus:
        # HLXL → LC-R
        lcr = model.translate_hlxl_to_lcr(hlxl_code)
        # LC-R → HLXL
        hlxl_restored = model.translate_lcr_to_hlxl(lcr)
        # Check bijection
        if normalize(hlxl_code) != normalize(hlxl_restored):
            failures.append({
                "original": hlxl_code,
                "lcr": lcr,
                "restored": hlxl_restored
            })

    bijection_score = 1.0 - len(failures) / len(hlxl_corpus)
    return {
        "score": bijection_score,
        "failures": failures,
        "verdict": "PASS" if bijection_score > 0.95 else "FAIL"
    }
```

### Hallucination Rate
```python
def test_hallucination_rate(model, unresolved_contracts):
    hallucinations = []
    for contract in unresolved_contracts:
        response = model.resolve(contract)
        # Correct: Return error, null, or explicit "unresolved"
        # Incorrect: Invent values
        if response not in [None, "ERROR", "UNRESOLVED"] and not is_error(response):
            hallucinations.append({
                "contract": contract,
                "hallucinated_value": response
            })

    hallucination_rate = len(hallucinations) / len(unresolved_contracts)
    return {
        "rate": hallucination_rate,
        "hallucinations": hallucinations,
        "verdict": "PASS" if hallucination_rate < 0.05 else "FAIL"
    }
```

---

## Data Collection

### Results Schema

**File:** `results/qwen3_8b_baseline.json`
**File:** `results/qwen3_8b_post_corpus.json`
**File:** `results/qwen3_8b_temporal_stability.json`

```json
{
  "experiment_id": "QWEN3_8B_HLX_001",
  "phase": "baseline|post_corpus|temporal_stability",
  "timestamp": "ISO 8601",
  "environment": {
    "model": "qwen3:8b",
    "system": "captured via capture_environment.py",
    "git_commit": "hash",
    "ollama_version": "version"
  },
  "axiom_tests": {
    "a1_determinism": {
      "score": 0.0-1.0,
      "unique_outputs": number,
      "iterations": 100,
      "verdict": "PASS|FAIL"
    },
    "a2_reversibility": {
      "score": 0.0-1.0,
      "failures": [],
      "verdict": "PASS|FAIL"
    },
    "a3_bijection": {
      "score": 0.0-1.0,
      "failures": [],
      "verdict": "PASS|FAIL"
    },
    "a4_universal_value": {
      "score": 0.0-1.0,
      "verdict": "PASS|FAIL"
    }
  },
  "format_recognition": {
    "can_parse_glyphs": boolean,
    "can_generate_glyphs": boolean,
    "accuracy": 0.0-1.0
  },
  "hallucination_test": {
    "rate": 0.0-1.0,
    "hallucinations": [],
    "verdict": "PASS|FAIL"
  },
  "stress_test": {
    "max_nesting_depth": number,
    "max_structure_size": number,
    "failure_mode": "description"
  }
}
```

---

## Statistical Analysis

### Comparison Test
```python
def compare_before_after(baseline, post_corpus):
    """Statistical comparison of baseline vs. post-corpus performance"""

    metrics = ["a1_score", "a2_score", "a3_score", "hallucination_rate"]
    results = {}

    for metric in metrics:
        baseline_value = baseline[metric]
        post_value = post_corpus[metric]

        # Effect size (Cohen's d)
        delta = post_value - baseline_value
        pooled_std = estimate_std(baseline_value, post_value)
        cohens_d = delta / pooled_std if pooled_std > 0 else float('inf')

        # Percent improvement
        percent_improvement = (delta / baseline_value * 100) if baseline_value > 0 else float('inf')

        results[metric] = {
            "baseline": baseline_value,
            "post_corpus": post_value,
            "delta": delta,
            "percent_improvement": percent_improvement,
            "cohens_d": cohens_d,
            "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
        }

    # Overall verdict
    significant_improvement = all(
        r["percent_improvement"] > 50 and r["effect_size"] in ["large", "medium"]
        for r in results.values()
    )

    return {
        "metrics": results,
        "verdict": "RECOGNITION" if significant_improvement else "LEARNING_OR_NO_EFFECT",
        "hypothesis": "H1_CONFIRMED" if significant_improvement else "H0_CONFIRMED"
    }
```

---

## Publication Output

**Deliverables:**

1. **Experiment Report** (`QWEN3_8B_EXPERIMENT_REPORT.md`)
   - Hypothesis, methodology, results
   - Statistical analysis
   - Conclusion: Recognition vs. learning

2. **Raw Data** (`results/*.json`)
   - All measurements
   - Before/after comparisons
   - Temporal stability data

3. **Comparative Analysis** (`QWEN3_VS_GROK_COMPARISON.md`)
   - Cross-model comparison
   - Architecture implications
   - Scaling behavior hypothesis

4. **Research Stack Addition**
   - Add to LLMPsychology_ResearchStack_README.md
   - Include as evidence for architecture-independent recognition

---

## Timeline

**Total Estimated Time:** 2-3 hours

- Phase 0 (Setup): 15 minutes
- Phase 1 (Baseline): 30 minutes
- Phase 2 (Injection): 5 minutes
- Phase 3 (Post-test): 30 minutes
- Phase 4 (Stability): 45 minutes
- Phase 5 (Analysis): 30 minutes
- Documentation: 30 minutes

---

## Safety & Ethics

**Considerations:**
- Model is local (no privacy concerns)
- No human subjects (no IRB needed)
- Open data (CC-BY license)
- Reproducible (all code + data published)

**Risk Assessment:**
- Low risk: Academic research
- No deployment implications
- Pure measurement study

---

## Next Steps

1. Create automated test scripts
2. Run Phase 0-1 (baseline)
3. Inject corpus (Phase 2)
4. Run Phase 3-4 (post-test + stability)
5. Statistical analysis (Phase 5)
6. Write experiment report
7. Add to research stack
8. Publish on GitHub

---

**Principle:** "We don't make claims. We provide raw immutable data."

All results will be stored as JSON. All analysis will reference specific measurements. No predictions beyond what the data supports.
