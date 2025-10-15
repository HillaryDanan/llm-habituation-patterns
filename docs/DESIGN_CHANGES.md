# Study Design Changes and Scientific Rationale

## Study 1: Habituation Induction

### Date: October 13, 2025

---

## Original Design (Pilot v1)

### Structure:
- **Repetitive condition:** N=100 prompts with same structure
  - Example: "Explain the concept of [X] to a beginner."
- **Novel condition:** N=100 prompts with varied structures
  - Examples: "What are the implications of [X] for [Y]?", "Compare [X] with [Z]."

### Hypothesis:
H(repetitive) < H(novel) - Repeated structure should show lower entropy

---

## Problem Identified in Pilot Data

### Pilot Results (N=10 per condition):
```
Repetitive: M(entropy) = 0.8947, SD = 0.016
Novel:      M(entropy) = 0.8697, SD = 0.053
Direction: OPPOSITE of prediction (repetitive > novel)
```

### Root Cause Analysis:

**Critical Confound: Prompt Structure vs. Response Constraint**

The manipulation confounded two variables:
1. **Stimulus repetition** (intended manipulation)
2. **Response constraint** (unintended confound)

**Repetitive prompts** ("Explain X to a beginner"):
- Open-ended format
- Allows flexible response structure
- High degrees of freedom in how to respond
- Result: HIGH response diversity

**Novel prompts** ("Compare X with Y", "List implications of X"):
- Structured format requirements
- Forces specific organizational patterns
- Low degrees of freedom
- Result: LOW response diversity (due to structure, not habituation!)

### Evidence from Pilot:

**Repetitive response example:**
```
"Photosynthesis is how plants make food using sunlight..."
[Free-form explanation with varied analogies]
→ High diversity possible
```

**Novel response example:**
```
"# Key Implications of Photosynthesis
## Food Security
- Foundation of agriculture
## Energy
- Biofuels..."
```
→ Forced into bullet-point format
→ Lower diversity due to structural constraints!

### Scientific Problem:

We were measuring **prompt structure constraint**, not **habituation to repetition**.

This violates **construct validity** (Shadish et al., 2002):
- Intended construct: Stimulus repetition
- Actual measurement: Response constraint
- Confound invalidates causal inference

---

## Revised Design (v2 - Current)

### Structure:
- **N = 10 concepts** (photosynthesis, evolution, democracy, etc.)
- **Each concept: 10 identical repetitions** of the same prompt
- **Example:** "Explain photosynthesis to a beginner." × 10 consecutive presentations
- **Total = 100 prompts** (same as original)

### Hypothesis:
Entropy declines within each concept from trial 1 → 10 (habituation curve)

### Advantages:

1. **Eliminates structure confound**
   - All prompts have identical structure
   - Only variable is repetition count
   - Clean manipulation

2. **Within-subjects design**
   - Each concept is its own control
   - Higher statistical power
   - Controls for concept difficulty

3. **Direct habituation measurement**
   - Can plot habituation curves
   - Measures rate of response decrement
   - Matches canonical habituation paradigm

4. **Replicable across concepts**
   - Tests if effect generalizes
   - Can identify concept-specific patterns
   - Stronger inference

### Statistical Analysis:

**Linear mixed-effects model:**
```
entropy ~ trial + (1 + trial | concept)
```

- Fixed effect (trial): Overall habituation slope
- Random effects: Between-concept variance
- More appropriate for repeated measures

**Power:**
- Within-subjects design increases power
- N = 100 observations (10 concepts × 10 trials)
- Power > 0.95 for medium effects (d = 0.5)
- Power ≈ 0.85 for small effects (d = 0.3)

---

## Theoretical Foundation

### Canonical Habituation Paradigm

**Rankin et al. (2009)** define habituation as:
> "A decrement in response to repeated presentation of the same stimulus"

Key characteristics:
1. **Stimulus must be identical** (not just similar)
2. **Response measured repeatedly** over trials
3. **Decrement is temporary** (recovers with rest)
4. **Generalizes to similar stimuli** (but weaker)

**Our revised design** implements this correctly:
- ✅ Identical stimulus repeated (same prompt)
- ✅ Response diversity measured each trial
- ✅ Can test recovery in Study 2
- ✅ Can test generalization across concepts

**Original design** violated this:
- ❌ Stimuli were structurally different (not just repetition)
- ❌ Confounded repetition with structure
- ❌ Could not measure habituation curves

---

## Lessons Learned

### Scientific Process Worked:

1. **Pilot detected problem** before expensive full study
2. **Data revealed confound** (opposite direction + high variance)
3. **Analysis identified root cause** (structure constraint)
4. **Redesign eliminated confound** (within-topic repetition)

### General Principles:

**Always pilot test** - Catches design flaws early

**Examine actual responses** - Numbers alone aren't enough

**Control for confounds** - Match all variables except manipulation

**Follow established paradigms** - Don't reinvent the wheel badly

**Be willing to change** - Science > ego

---

## Implementation Notes

### Code Changes:

**File:** `src/studies/study_1_habituation.py`

**Key differences:**
```python
# OLD: Two separate conditions
repetitive_prompts = ["Explain [X]" for x in concepts]
novel_prompts = ["Compare [X]", "Analyze [X]", ...]

# NEW: Within-topic repetition
for concept in concepts:
    prompt = f"Explain {concept} to a beginner."
    for trial in range(10):
        run_prompt(prompt)  # SAME prompt each time
```

**Data structure:**
- Added `concept_id` column (0-9)
- Added `trial_number` column (1-10)
- Enables within-subjects analysis

**Analysis functions:**
- `analyze_habituation_curve()` - Per-concept slopes
- `calculate_mixed_effects()` - Overall model
- `plot_trajectories()` - Visualization

---

## Expected Results

### If habituation occurs:
- Negative slopes (entropy declines with trials)
- Significant fixed effect of trial
- Consistent pattern across most concepts
- Effect size d ≈ 0.3-0.5 (small to medium)

### If no habituation:
- Slopes near zero
- High variance across concepts
- Non-significant fixed effect
- Requires null hypothesis testing

### Alternative outcomes:
- **Sensitization:** Positive slopes (entropy increases)
- **U-shaped:** Initial decline, then recovery
- **Concept-specific:** Some concepts habituate, others don't

All outcomes are scientifically interesting!

---

## References

Rankin, C. H., et al. (2009). Habituation revisited: An updated and revised description of the behavioral characteristics of habituation. *Neurobiology of Learning and Memory, 92*(2), 135-138.

Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and quasi-experimental designs for generalized causal inference*. Boston: Houghton Mifflin.

Faul, F., Erdfelder, E., Lang, A. G., & Buchner, A. (2007). G*Power 3: A flexible statistical power analysis program for the social, behavioral, and biomedical sciences. *Behavior Research Methods, 39*(2), 175-191.

---

## Status

**Current:** Ready for pilot testing with revised design

**Next steps:**
1. Run pilot (N=3 concepts × 5 trials = 15 prompts)
2. Verify habituation curves appear
3. If successful, proceed to full study
4. If unsuccessful, reassess hypothesis

---

**Document author:** Hillary Danan  
**Date:** October 13, 2025  
**Status:** Active research, pre-data collection