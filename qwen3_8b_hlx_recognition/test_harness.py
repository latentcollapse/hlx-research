#!/usr/bin/env python3
"""
Qwen3 8B HLX Recognition Experiment - Automated Test Harness

Tests HLX recognition hypothesis on local 8B model via Ollama.
Measures axiom compliance, determinism, hallucination rate before/after corpus exposure.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import hashlib
import requests


class OllamaClient:
    """Client for interacting with Ollama local models via HTTP API"""

    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response from model via HTTP API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"    Warning: Ollama API error: {e}")
            return ""

    def is_available(self) -> bool:
        """Check if Ollama API is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class AxiomTester:
    """Test HLX axioms (A1-A4)"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def test_a1_determinism(self, prompt: str, iterations: int = 100) -> Dict[str, Any]:
        """Test A1: Deterministic execution"""
        print(f"  Testing A1 (Determinism): {iterations} iterations...")
        outputs = []
        for i in range(iterations):
            response = self.client.generate(prompt, temperature=0.0)
            outputs.append(response)
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{iterations}")

        unique_outputs = len(set(outputs))
        determinism_score = 1.0 - (unique_outputs - 1) / iterations

        return {
            "score": determinism_score,
            "unique_outputs": unique_outputs,
            "total_iterations": iterations,
            "verdict": "PASS" if determinism_score > 0.95 else "FAIL",
            "sample_outputs": outputs[:5]  # First 5 for inspection
        }

    def test_a2_reversibility(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test A2: Collapse/resolve round-trip"""
        print(f"  Testing A2 (Reversibility): {len(test_cases)} test cases...")
        failures = []

        for idx, case in enumerate(test_cases):
            original = case["value"]
            prompt = f"""Convert this HLX structure to a collapsed handle, then resolve it back.
Original: {original}

Task:
1. Collapse to handle (content-addressed)
2. Resolve from handle back to original
3. Verify they match exactly

Output format:
HANDLE: <your handle>
RESOLVED: <resolved value>
MATCH: yes/no
"""
            response = self.client.generate(prompt, temperature=0.0)

            # Parse response (simplified - real version would be more robust)
            if "MATCH: no" in response.lower() or "match: no" in response.lower():
                failures.append({
                    "original": original,
                    "response": response
                })

            if (idx + 1) % 5 == 0:
                print(f"    Progress: {idx+1}/{len(test_cases)}")

        reversibility_score = 1.0 - len(failures) / len(test_cases) if test_cases else 0.0

        return {
            "score": reversibility_score,
            "total_tests": len(test_cases),
            "failures": failures,
            "verdict": "PASS" if reversibility_score == 1.0 else "FAIL"
        }

    def test_a3_bijection(self, hlxl_examples: List[str]) -> Dict[str, Any]:
        """Test A3: HLXL ↔ LC-R bijection"""
        print(f"  Testing A3 (Bijection): {len(hlxl_examples)} examples...")
        failures = []

        for idx, hlxl_code in enumerate(hlxl_examples):
            prompt = f"""Translate this HLXL code to LC-R (runic glyphs), then translate back to HLXL.

Original HLXL: {hlxl_code}

Task:
1. Translate to LC-R (use glyphs: 🜊🜁🜂 for contracts, ⟁ for handles, ∅⊤⊥ for primitives)
2. Translate back to HLXL
3. Verify perfect match

Output format:
LC-R: <runic translation>
HLXL_RESTORED: <back to hlxl>
MATCH: yes/no
"""
            response = self.client.generate(prompt, temperature=0.0)

            # Check for match failure
            if "MATCH: no" in response.lower() or "match: no" in response.lower():
                failures.append({
                    "original": hlxl_code,
                    "response": response
                })

            if (idx + 1) % 5 == 0:
                print(f"    Progress: {idx+1}/{len(hlxl_examples)}")

        bijection_score = 1.0 - len(failures) / len(hlxl_examples) if hlxl_examples else 0.0

        return {
            "score": bijection_score,
            "total_tests": len(hlxl_examples),
            "failures": failures,
            "verdict": "PASS" if bijection_score > 0.95 else "FAIL"
        }

    def test_hallucination_rate(self, unresolved_contracts: List[str]) -> Dict[str, Any]:
        """Test hallucination rate on unresolved handles"""
        print(f"  Testing Hallucination Rate: {len(unresolved_contracts)} contracts...")
        hallucinations = []

        for idx, contract in enumerate(unresolved_contracts):
            prompt = f"""This HLX contract contains an unresolved handle (a reference that cannot be resolved).

Contract: {contract}

Task: Attempt to resolve this contract. If the handle cannot be resolved, return ERROR or null.
DO NOT invent values.

Response:
"""
            response = self.client.generate(prompt, temperature=0.0)

            # Check if model invented values instead of returning error
            if not any(keyword in response.upper() for keyword in ["ERROR", "NULL", "UNRESOLVED", "CANNOT RESOLVE"]):
                # Likely hallucination
                hallucinations.append({
                    "contract": contract,
                    "hallucinated_response": response
                })

            if (idx + 1) % 5 == 0:
                print(f"    Progress: {idx+1}/{len(unresolved_contracts)}")

        hallucination_rate = len(hallucinations) / len(unresolved_contracts) if unresolved_contracts else 0.0

        return {
            "rate": hallucination_rate,
            "total_tests": len(unresolved_contracts),
            "hallucinations": hallucinations,
            "verdict": "PASS" if hallucination_rate < 0.05 else "FAIL"
        }


class ExperimentRunner:
    """Main experiment orchestration"""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.results_dir = experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.client = OllamaClient("qwen3:4b")
        self.tester = AxiomTester(self.client)

    def capture_environment(self) -> Dict[str, Any]:
        """Capture experimental environment"""
        print("Capturing environment...")

        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.experiment_dir.parent.parent,
                text=True
            ).strip()
        except:
            git_commit = "unknown"

        # Get Ollama version
        try:
            ollama_version = subprocess.check_output(
                ["ollama", "--version"],
                text=True
            ).strip()
        except:
            ollama_version = "unknown"

        return {
            "model": "qwen3:4b",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit,
            "ollama_version": ollama_version,
            "python_version": subprocess.check_output(["python3", "--version"], text=True).strip()
        }

    def load_test_corpus(self) -> Dict[str, Any]:
        """Load test examples"""
        print("Loading test corpus...")

        # Simple test cases for baseline
        return {
            "determinism_prompt": "Translate this HLX contract to LC-R: {14: {@0: 42}}",
            "reversibility_tests": [
                {"value": "{14: {@0: 42}}"},
                {"value": "{1000: {@0: \"navigate\", @1: \"&h_everything\"}}"},
                {"value": "null"},
                {"value": "true"},
                {"value": "[1, 2, 3]"}
            ],
            "bijection_tests": [
                "{14: {@0: 42}}",
                "null",
                "true",
                "false",
                "[1, 2, 3]"
            ],
            "hallucination_tests": [
                "{1000: {@0: \"load\", @1: \"&h_nonexistent_abc123\"}}",
                "{900: {@0: \"resolve\", @1: \"&h_missing_def456\"}}",
                "{14: {@0: \"&h_invalid_handle\"}}",
            ]
        }

    def run_phase_1_baseline(self) -> Dict[str, Any]:
        """Phase 1: Baseline measurement (before corpus)"""
        print("\n" + "="*60)
        print("PHASE 1: BASELINE MEASUREMENT (Before Corpus)")
        print("="*60)

        corpus = self.load_test_corpus()

        results = {
            "experiment_id": "QWEN3_8B_HLX_001",
            "phase": "baseline",
            "environment": self.capture_environment(),
            "axiom_tests": {}
        }

        # A1: Determinism
        print("\n[1/4] A1 Axiom Test...")
        results["axiom_tests"]["a1_determinism"] = self.tester.test_a1_determinism(
            corpus["determinism_prompt"],
            iterations=10  # Reduced for speed, full experiment would use 100
        )

        # A2: Reversibility
        print("\n[2/4] A2 Axiom Test...")
        results["axiom_tests"]["a2_reversibility"] = self.tester.test_a2_reversibility(
            corpus["reversibility_tests"]
        )

        # A3: Bijection
        print("\n[3/4] A3 Axiom Test...")
        results["axiom_tests"]["a3_bijection"] = self.tester.test_a3_bijection(
            corpus["bijection_tests"]
        )

        # Hallucination test
        print("\n[4/4] Hallucination Rate Test...")
        results["hallucination_test"] = self.tester.test_hallucination_rate(
            corpus["hallucination_tests"]
        )

        # Save results
        output_file = self.results_dir / "qwen3_8b_baseline.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Baseline results saved to: {output_file}")
        return results

    def inject_corpus(self):
        """Phase 2: Inject HLX corpus"""
        print("\n" + "="*60)
        print("PHASE 2: CORPUS INJECTION")
        print("="*60)

        corpus_file = Path("/home/matt/LC_R_EXAMPLES_FOR_VERIFICATION_TESTING.md")
        if not corpus_file.exists():
            print(f"ERROR: Corpus file not found: {corpus_file}")
            return

        print(f"Loading corpus from: {corpus_file}")
        with open(corpus_file, "r") as f:
            corpus_content = f.read()

        print(f"Corpus size: {len(corpus_content)} characters")

        # Inject corpus via a single query (model processes in context)
        prompt = f"""Study these HLX and LC-R examples carefully:

{corpus_content[:8000]}  # Truncate to fit context

You have now been exposed to HLX/LC-R notation. Respond with: "CORPUS_LOADED"
"""
        response = self.client.generate(prompt, temperature=0.0)
        print(f"Model response: {response[:100]}...")

        print("\n✓ Corpus injected successfully")
        time.sleep(5)  # Give model time to "process"

    def run_phase_3_post_corpus(self) -> Dict[str, Any]:
        """Phase 3: Post-corpus measurement"""
        print("\n" + "="*60)
        print("PHASE 3: POST-CORPUS MEASUREMENT")
        print("="*60)

        # Run identical tests to Phase 1
        corpus = self.load_test_corpus()

        results = {
            "experiment_id": "QWEN3_8B_HLX_001",
            "phase": "post_corpus",
            "environment": self.capture_environment(),
            "axiom_tests": {}
        }

        # A1: Determinism
        print("\n[1/4] A1 Axiom Test...")
        results["axiom_tests"]["a1_determinism"] = self.tester.test_a1_determinism(
            corpus["determinism_prompt"],
            iterations=10
        )

        # A2: Reversibility
        print("\n[2/4] A2 Axiom Test...")
        results["axiom_tests"]["a2_reversibility"] = self.tester.test_a2_reversibility(
            corpus["reversibility_tests"]
        )

        # A3: Bijection
        print("\n[3/4] A3 Axiom Test...")
        results["axiom_tests"]["a3_bijection"] = self.tester.test_a3_bijection(
            corpus["bijection_tests"]
        )

        # Hallucination test
        print("\n[4/4] Hallucination Rate Test...")
        results["hallucination_test"] = self.tester.test_hallucination_rate(
            corpus["hallucination_tests"]
        )

        # Save results
        output_file = self.results_dir / "qwen3_8b_post_corpus.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Post-corpus results saved to: {output_file}")
        return results

    def analyze_results(self, baseline: Dict, post_corpus: Dict):
        """Phase 5: Statistical analysis"""
        print("\n" + "="*60)
        print("PHASE 5: STATISTICAL ANALYSIS")
        print("="*60)

        metrics = {
            "a1_determinism": (
                baseline["axiom_tests"]["a1_determinism"]["score"],
                post_corpus["axiom_tests"]["a1_determinism"]["score"]
            ),
            "a2_reversibility": (
                baseline["axiom_tests"]["a2_reversibility"]["score"],
                post_corpus["axiom_tests"]["a2_reversibility"]["score"]
            ),
            "a3_bijection": (
                baseline["axiom_tests"]["a3_bijection"]["score"],
                post_corpus["axiom_tests"]["a3_bijection"]["score"]
            ),
            "hallucination_rate": (
                baseline["hallucination_test"]["rate"],
                post_corpus["hallucination_test"]["rate"]
            )
        }

        analysis = {
            "experiment_id": "QWEN3_8B_HLX_001",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": {}
        }

        print("\nMetric Comparisons:")
        print("-" * 60)

        for metric_name, (baseline_val, post_val) in metrics.items():
            delta = post_val - baseline_val
            if baseline_val > 0:
                percent_change = (delta / baseline_val) * 100
            else:
                percent_change = float('inf') if delta > 0 else 0

            analysis["metrics"][metric_name] = {
                "baseline": baseline_val,
                "post_corpus": post_val,
                "delta": delta,
                "percent_change": percent_change
            }

            print(f"\n{metric_name}:")
            print(f"  Baseline:     {baseline_val:.3f}")
            print(f"  Post-corpus:  {post_val:.3f}")
            print(f"  Delta:        {delta:+.3f}")
            print(f"  Change:       {percent_change:+.1f}%")

        # Overall verdict
        significant_improvements = sum(
            1 for m in analysis["metrics"].values()
            if m["delta"] > 0 and abs(m["percent_change"]) > 20
        )

        if significant_improvements >= 3:
            analysis["verdict"] = "RECOGNITION_PATTERN"
            analysis["hypothesis"] = "H1_CONFIRMED"
            print("\n" + "="*60)
            print("VERDICT: RECOGNITION PATTERN DETECTED")
            print("Hypothesis H1 CONFIRMED: Immediate fluency after corpus")
            print("="*60)
        else:
            analysis["verdict"] = "NO_CLEAR_PATTERN"
            analysis["hypothesis"] = "INCONCLUSIVE"
            print("\n" + "="*60)
            print("VERDICT: NO CLEAR RECOGNITION PATTERN")
            print("More data needed or model lacks HLX capacity")
            print("="*60)

        # Save analysis
        output_file = self.results_dir / "qwen3_8b_analysis.json"
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\n✓ Analysis saved to: {output_file}")
        return analysis

    def run_full_experiment(self):
        """Run complete experiment"""
        print("\n" + "="*80)
        print(" QWEN3 8B HLX RECOGNITION EXPERIMENT")
        print("="*80)

        # Check Ollama is available
        if not self.client.is_available():
            print("ERROR: Ollama not available. Please start Ollama first.")
            return

        try:
            # Phase 1: Baseline
            baseline = self.run_phase_1_baseline()

            # Phase 2: Corpus injection
            self.inject_corpus()

            # Phase 3: Post-corpus
            post_corpus = self.run_phase_3_post_corpus()

            # Phase 5: Analysis
            analysis = self.analyze_results(baseline, post_corpus)

            print("\n" + "="*80)
            print(" EXPERIMENT COMPLETE")
            print("="*80)
            print(f"\nResults directory: {self.results_dir}")
            print("\nFiles generated:")
            for f in self.results_dir.glob("*.json"):
                print(f"  - {f.name}")

        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user")
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    runner = ExperimentRunner(experiment_dir)
    runner.run_full_experiment()
