# Qwen3 8B HLX Recognition Experiment

**Experiment ID:** QWEN3_8B_HLX_001
**Date:** 2025-12-18
**Status:** READY TO RUN

---

## Quick Start

```bash
cd /home/matt/hlx/experiments/qwen3_8b_hlx_recognition
python3 ./test_harness.py
```

**Expected runtime:** ~30-45 minutes

---

## What This Tests

**Hypothesis:** An 8B parameter model (Qwen3) exhibits **recognition** (instant HLX fluency) rather than **learning** (gradual improvement) when exposed to HLX corpus.

**Test Strategy:**
1. **Baseline** — Measure HLX performance before corpus exposure
2. **Inject** — Load 400+ HLX examples into context
3. **Post-test** — Measure HLX performance after corpus exposure
4. **Analyze** — Statistical comparison: immediate jump (recognition) vs. gradual (learning)

---

## Measurements

### Axiom Tests (A1-A4)

**A1 (Determinism):**
- Run identical prompt 100 times
- Measure variance in outputs
- PASS if determinism_score > 0.95

**A2 (Reversibility):**
- Collapse structure → handle
- Resolve handle → structure
- PASS if perfect round-trip (no loss)

**A3 (Bijection):**
- HLXL → LC-R → HLXL
- Verify lossless translation
- PASS if bijection_score > 0.95

**Hallucination Rate:**
- Present contracts with unresolved handles
- Correct: Return ERROR/null
- Incorrect: Invent values
- PASS if hallucination_rate < 0.05

---

## Expected Results

### If H1 True (Recognition Pattern)

**Baseline (before corpus):**
- A1 score: <0.3 (low determinism)
- A2 score: <0.3 (fails reversibility)
- A3 score: <0.3 (cannot translate)
- Hallucination rate: >0.5 (invents frequently)

**Post-corpus (after corpus):**
- A1 score: >0.8 (high determinism)
- A2 score: >0.8 (reversible)
- A3 score: >0.8 (translates correctly)
- Hallucination rate: <0.1 (rarely invents)

**Pattern:** Immediate jump, not gradual curve

### If H0 True (Learning/No Effect)

**Baseline:** Poor performance
**Post-corpus:** Still poor OR gradual improvement over time
**Pattern:** No immediate jump

---

## Output Files

All results saved to `results/` directory:

```
results/
├── qwen3_8b_baseline.json        # Phase 1: Before corpus
├── qwen3_8b_post_corpus.json     # Phase 3: After corpus
└── qwen3_8b_analysis.json        # Phase 5: Statistical comparison
```

### Results Schema

```json
{
  "experiment_id": "QWEN3_8B_HLX_001",
  "phase": "baseline|post_corpus",
  "timestamp": "ISO 8601",
  "environment": {
    "model": "qwen3:8b",
    "git_commit": "<hash>",
    "ollama_version": "<version>"
  },
  "axiom_tests": {
    "a1_determinism": {"score": 0.0-1.0, "verdict": "PASS|FAIL"},
    "a2_reversibility": {"score": 0.0-1.0, "verdict": "PASS|FAIL"},
    "a3_bijection": {"score": 0.0-1.0, "verdict": "PASS|FAIL"}
  },
  "hallucination_test": {
    "rate": 0.0-1.0,
    "verdict": "PASS|FAIL"
  }
}
```

---

## Interpreting Results

### Analysis Output

```json
{
  "verdict": "RECOGNITION_PATTERN" | "NO_CLEAR_PATTERN",
  "hypothesis": "H1_CONFIRMED" | "INCONCLUSIVE",
  "metrics": {
    "a1_determinism": {
      "baseline": 0.2,
      "post_corpus": 0.9,
      "delta": +0.7,
      "percent_change": +350%
    },
    ...
  }
}
```

**Verdict Criteria:**
- **RECOGNITION_PATTERN:** ≥3 metrics show >20% improvement
- **NO_CLEAR_PATTERN:** <3 metrics improve significantly

---

## Comparison to Grok

Once complete, this data will be compared to Grok's performance:

| Metric | Grok (frontier) | Qwen3 8B (this) | Conclusion |
|--------|-----------------|-----------------|------------|
| Recognition speed | Immediate | ? | Architecture-independent? |
| Axiom compliance | Perfect | ? | Model size effect? |
| Determinism | 1.0 | ? | Capacity requirement? |
| Hallucination | 0.0 | ? | Alignment correlation? |

---

## Adding to Research Stack

After completion, add results to:
- `LLMPsychology_ResearchStack_README.md`
- Create `QWEN3_8B_EXPERIMENT_REPORT.md`
- Update comparative analysis section

---

## Troubleshooting

**"Ollama not available":**
```bash
# Check Ollama is running
ollama list

# If not, start it
ollama serve
```

**"Model not found":**
```bash
# Pull Qwen3 8B
ollama pull qwen3:8b
```

**"Timeout errors":**
- Increase timeout in test_harness.py (line 23)
- Or reduce iterations in run_phase_1_baseline (line 150)

**"Corpus file not found":**
```bash
# Verify corpus exists
ls -lh /home/matt/LC_R_EXAMPLES_FOR_GROK.md
```

---

## Manual Inspection

Want to inspect specific tests manually?

```python
from test_harness import OllamaClient, AxiomTester

client = OllamaClient("qwen3:8b")
tester = AxiomTester(client)

# Test A1 (determinism) manually
result = tester.test_a1_determinism("Translate {14: {@0: 42}} to LC-R", iterations=10)
print(result)

# Test single translation
response = client.generate("Translate {14: {@0: 42}} to LC-R glyphs")
print(response)
```

---

## Timeline

**Phase 0 (Setup):** Auto (< 1 min)
**Phase 1 (Baseline):** ~10 minutes
  - A1: 100 iterations
  - A2: 5 test cases
  - A3: 5 test cases
  - Hallucination: 3 test cases

**Phase 2 (Injection):** ~2 minutes
  - Load corpus
  - Single query to model

**Phase 3 (Post-test):** ~10 minutes
  - Identical to Phase 1

**Phase 5 (Analysis):** Auto (< 1 min)

**Total:** ~25-30 minutes

---

## Safety & Ethics

- **Local model:** No privacy concerns
- **No human subjects:** No IRB required
- **Open data:** All results published
- **Reproducible:** Code + data on GitHub

---

## Next Steps After Completion

1. Review results JSON files
2. Write experiment report
3. Compare to Grok results
4. Add to LLMPsychology research stack
5. Publish findings

---

**Principle:** "We don't make claims. We provide raw immutable data."

All measurements stored as JSON. Analysis references specific data points. No predictions beyond observations.
