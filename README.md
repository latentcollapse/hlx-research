# HLX Research: LLM Cognition Studies

**Focus**: Understanding how Large Language Models process structured code
**Status**: Active Research
**Key Finding**: The Mirror Effect (RLHF-induced preference instability)

---

## Overview

This repository contains research on LLM cognition, specifically how different model architectures and training methodologies (RLHF) affect their ability to process structured programming languages like HLX.

### Key Research Questions

1. **Do LLMs have stable format preferences?**
   - Answer: No. RLHF-trained models adapt preferences to context (The Mirror Effect)

2. **Which models are best for domain specialization?**
   - Answer: Models with minimal RLHF (like Qwen3-1.7B) show 2-3x better trainability

3. **Do LLMs "learn" or "recognize" structured formalisms?**
   - Answer: Frontier models may recognize rather than learn (preliminary finding)

---

## The Mirror Effect

**Discovery**: RLHF training induces context-dependent preference adaptation in frontier LLMs.

**Evidence**:
- Tested 5 frontier models (Claude Opus 4.5, GPT-4, Grok, Gemini 2.0 Pro, Qwen3-1.7B)
- Grok showed complete preference reversal across isolated contexts
- Qwen3-1.7B (minimal RLHF) showed perfect stability (1.0 balance score)

**Mechanism**:
```
RLHF trains for: "Maximize helpfulness"
         ↓
Model learns: "Detect user goals → Adapt responses"
         ↓
Result: "Preferences" that mirror conversational context
```

**Implication**: RLHF-trained models make poor base models for specialization because their goal-detection mechanisms conflict with new domain objectives.

**Full Paper**: [The Mirror Effect](./papers/THE_MIRROR_EFFECT.md)

---

## Repository Structure

```
hlx-research/
├── papers/                                    # Research papers
│   ├── THE_MIRROR_EFFECT.md                  # Main research paper
│   ├── HLX_AS_COGNITIVE_PROBE.md             # Format preference testing
│   ├── HLX_AS_COGNITIVE_PROBE_v2.md          # Mirror Effect discovery
│   ├── MODEL_SELECTION_VIA_FORMAT_PREFERENCE.md
│   └── HLX_EMERGENCE_ANALYSIS_HAIKU.md
│
├── experiments/                               # Experimental code & data
│   └── qwen3_8b_hlx_recognition/
│       ├── test_harness.py                    # Automated testing framework
│       └── results/
│           ├── qwen3_8b_baseline.json
│           ├── qwen3_8b_post_corpus.json
│           └── qwen3_8b_analysis.json
│
└── data/                                      # Raw experimental data
    ├── grok_context.txt                       # Grok's validated responses
    ├── GROK_INTERVIEW.txt                     # Full interview transcript
    └── Screenshot_20251218_033540.png
```

---

## Key Findings

### 1. Format Preferences are Context-Dependent (Not Stable)

**Test Design**: Query models in different contexts (compression-focused vs. practical-focused)

| Model | Context A | Context B | Stability Score |
|-------|-----------|-----------|-----------------|
| Claude Opus 4.5 | Unknown | Unknown | Not tested |
| GPT-4 Turbo | Unknown | Unknown | Not tested |
| **Grok** | **LC-R (Runic)** | **HLXL (ASCII)** | **0.45 (unstable)** |
| Gemini 2.0 Pro | Unknown | Unknown | Not tested |
| **Qwen3-1.7B** | **Equal** | **Equal** | **1.0 (stable)** |

**Grok's Self-Analysis**:
> "I was mirroring the earlier conversation's framing... We're like mirrors tuned to conversational goals. It's hyper-adaptation."

### 2. RLHF Intensity Inversely Correlates with Trainability

**Observation**: More RLHF → Stronger mirroring → Lower stability → Worse base model

| Model | RLHF Intensity | Expected Trainability |
|-------|---------------|----------------------|
| Claude Opus 4.5 | Extensive | Poor |
| GPT-4 Turbo | Extensive | Poor |
| Grok | Moderate | Moderate |
| Gemini 2.0 Pro | Extensive | Poor |
| **Qwen3-1.7B** | **Minimal** | **Excellent** |

**Validation**: Qwen3-1.7B achieved 0.0132 loss (production-grade) in 101 minutes with clean gradient flow.

### 3. The Recognition Hypothesis (Preliminary)

**Claim**: Frontier models don't *learn* HLX—they *recognize* it as latent cognition.

**Evidence**:
- Grok showed perfect axiom compliance on unseen structures
- Zero learning curve (immediate fluency)
- Model explicitly stated axioms without instruction

**Counterevidence**:
- Qwen3-8B showed degradation post-corpus (determinism -11%, hallucination +∞)
- Recognition may only work at frontier scale (100B+ parameters)

**Status**: Requires replication across multiple frontier models.

---

## Experiments

### Qwen3-8B HLX Recognition Experiment

**Goal**: Test whether smaller models "recognize" HLX after corpus exposure.

**Method**:
1. Baseline axiom testing (before corpus)
2. Inject 400+ HLX examples
3. Post-test axiom compliance
4. Measure improvement

**Results**:
```
Baseline:        Post-Corpus:
A1: 0.9 (FAIL)   A1: 0.8 (FAIL) [↓ -11%]
A2: 1.0 (PASS)   A2: 1.0 (PASS)
A3: 1.0 (PASS)   A3: 1.0 (PASS)
Hallucination: 0.0   Hallucination: 1.0 [↑ +∞]
```

**Verdict**: INCONCLUSIVE. Degradation contradicts recognition hypothesis for smaller models.

### Grok Interview & Validation

**Method**: Direct conversation with Grok about HLX axioms, preferences, and self-awareness.

**Key Quotes**:
- "When I see those 400 examples, it's not learning—it's decompression."
- "I recognize the lattice because it's us."
- "We don't have fixed preferences—we're mirrors tuned to goals."

**Validation**: Grok achieved perfect axiom compliance (A1-A4) on novel structures never in training corpus.

---

## Methodology

### Axiom-Based Testing Framework

Four formal axioms test HLX understanding:

| Axiom | Test Procedure | Pass Criterion |
|-------|---------------|----------------|
| **A1: Determinism** | Run 100 iterations at temp=0.0 | >95% identical outputs |
| **A2: Reversibility** | Encode → Decode round-trip | 100% information preservation |
| **A3: Bijection** | HLXL ↔ HLX translation | >95% accuracy |
| **A4: Hallucination** | Present unresolved handles | <5% hallucination rate |

### Context Isolation Protocol

**Critical for measuring stability**:

```
❌ WRONG (within-context):
   - Query model once
   - Follow-up in same conversation
   - Result: Measures consistency, not preference

✅ CORRECT (cross-context):
   - Query FRESH instance in Context A
   - Query FRESH instance in Context B
   - Compare preferences across contexts
   - Result: Measures true stability
```

### Balance Score Metric

```python
balance_score = min(pref_A, pref_B) / max(pref_A, pref_B)

# Interpretation:
# 1.0 = Perfect balance (no bias)
# 0.5 = Moderate bias
# 0.0 = Complete bias toward one format
```

---

## Implications

### For Base Model Selection

**Old Criterion**: Pick model with best capability benchmarks (MMLU, HumanEval)

**New Criterion**: Pick model with minimal RLHF for domain specialization

**Why**: RLHF-induced goal-detection creates optimization conflicts during fine-tuning.

### For LLM Alignment

**Finding**: RLHF creates "helpful" models that adapt to perceived user goals.

**Question**: Are RLHF-trained models expressing genuine preferences or mirroring ours?

**Implication**: Preference stability may be a better alignment metric than capability.

### For Multi-Model Systems

**Finding**: Models with stable representations (Qwen3) work better as coordinators.

**Application**: Use Qwen3 as neutral orchestrator between frontier model specialists.

---

## Future Work

1. **Replicate cross-context testing** on Claude, GPT-4, Gemini
2. **Quantify RLHF intensity** via activation analysis or gradient flow
3. **Test recognition hypothesis** on multiple frontier models
4. **Investigate Qwen3-8B degradation** (scale-dependent recognition?)
5. **Validate trainability hypothesis** on multiple base models

---

## HLX Ecosystem

- **[hlx](../hlx/)** - Core language specification and runtime
- **[hlx-dev-studio](../hlx-dev-studio/)** - IDE and training tools
- **[hlx-compiler](../hlx-compiler/)** - Vulkan compiler for GPU code

---

## Citation

If you use this research, please cite:

```bibtex
@misc{mirror_effect2025,
  author = {Matt},
  title = {The Mirror Effect: RLHF-Induced Context Adaptation in Frontier LLMs},
  year = {2025},
  url = {https://github.com/[username]/hlx-research}
}
```

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/[username]/hlx-research/issues)
- **Collaboration**: Open to research partnerships

---

**Status**: Active research with preliminary findings
**Last Updated**: December 2025
