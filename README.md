# Visibility-Aware Readability (Human–Machine Systems)

This repository provides reproducible code and artifacts for a study on **text readability feasibility**
under controlled, classroom-like viewing configurations. The central goal is **not** to infer intent or behavior,
but to model whether text is *perceptually accessible* given a viewing setup supporting a feasibility layer for
human–machine systems in educational and assessment contexts.

The experiments emphasize **interpretability** (Logistic Regression, Random Forest) and **deployment-relevant reliability**
including discrimination, probability calibration, robustness stress tests, and uncertainty-aware evaluation.

> **Important scope note.** The dataset includes a **human-rated visibility score** used as a perceptual proxy.
> This introduces *conceptual circularity* with the binary readability label (`can_read`) and should be interpreted
> as **predictive sufficiency within this measurement regime**, not as deployable sufficiency in real classrooms.


---

## Key Results (What this repo reproduces)

This code reproduces the following components reported in the manuscript:

1. **Scenario-level evaluation**
   - repeated stratified cross-validation (OOF predictions)
   - condition-blocked generalization across configuration groups

2. **Interpretable modeling**
   - Logistic Regression (L2-regularized) and Random Forest baselines

3. **Probability calibration**
   - Platt scaling and isotonic regression fitted on training-only predictions

4. **Explainability**
   - impurity-based importance, permutation importance, PDPs

5. **Sufficiency-oriented analyses (predictive, not causal)**
   - ablation: Geometry-only vs Visibility-only vs Full
   - residualization-style stress tests
   - information-theoretic summaries (as implemented)

6. **Robustness diagnostics**
   - structured perturbations (e.g., visibility bias / clipping, pose quantization)

7. **Uncertainty-aware evaluation**
   - split conformal prediction under scenario-level splits (as configured)
