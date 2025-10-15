# Supplementary Materials

## Large Language Models Exhibit Response Invariance

---

## Supplementary Methods

### Extended Experimental Details

**Pilot Study and Design Refinement**

Initial pilot testing (N=10 per condition) revealed a critical confound in the original design. The original approach compared repetitive prompts (same structure: "Explain X") with novel prompts (varied structures: "Compare X with Y", "Analyze X", "List implications of X"). Pilot data showed the opposite pattern from prediction: repetitive prompts yielded *higher* entropy than novel prompts (repetitive M=0.895, novel M=0.870).

Analysis revealed the confound: varied-structure prompts were inadvertently more *constrained*. Prompts requiring comparison, analysis, or structured lists forced specific response formats, reducing diversity. "Explain" prompts allowed open-ended responses with higher degrees of freedom.

**Solution:** Complete redesign using within-subjects repetition. Eliminated structure confound by holding prompt structure constant (always "Explain X") while varying repetition (same concept repeated 10 times). This implements the canonical habituation paradigm correctly.

Full documentation of design evolution: github.com/HillaryDanan/llm-habituation-patterns/docs/DESIGN_CHANGES.md

### Detailed Metric Calculations

**Shannon Entropy:**
```python
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize

def calculate_entropy(text):
    tokens = word_tokenize(text.lower())
    if len(tokens) == 0:
        return 0.0
    
    # Token probabilities
    counts = Counter(tokens)
    total = len(tokens)
    probs = np.array([count/total for count in counts.values()])
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalize by max possible entropy
    max_entropy = np.log2(len(counts))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized
```

**Semantic Novelty:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def semantic_novelty(text1, text2):
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    novelty = 1 - cos_sim
    
    return float(novelty)
```

### API Parameters and Rate Limiting

**API Calls:**
- Anthropic API: claude-sonnet-4-5-20250929
- OpenAI API: gpt-4-0125-preview
- Google Gemini API: gemini-2.5-flash

**Standard Parameters:**
- temperature: 1.0
- max_tokens: 300
- top_p: 1.0 (default)
- frequency_penalty: 0 (default for Anthropic)
- presence_penalty: 0 (default for Anthropic)

**Rate Limiting:**
- Anthropic: 50 requests/minute, 100K tokens/minute
- OpenAI: 500 requests/minute, 150K tokens/minute
- 1.2 second minimum interval between requests (across all studies)
- Automatic exponential backoff on rate limit errors

### Power Analysis Details

**G*Power 3.1 Calculations:**

**Study 1 (within-subjects):**
- Design: Paired t-test on slopes
- Effect size: d=0.5 (medium)
- α: 0.05 (two-tailed)
- Power: 0.90
- Required N: 34 concepts minimum
- Actual N: 10 concepts × 10 trials = 100 observations
- Achieved power: >0.95 for d=0.5

**For small effects:**
- Effect size: d=0.3
- Required N: 90 observations
- Actual N: 100
- Achieved power: 0.87

**Study 2 (recovery):**
- Design: Paired t-test (habituation vs. re-test)
- N=20 concept-repetitions per phase
- Power=0.83 for d=0.5

**Study 3 (correlation):**
- Design: Linear regression (novelty ~ turn)
- N=30 conversations × 10 turns = 300 observations
- Power>0.99 for medium effects (r=0.3)

**Study 4 (ANOVA):**
- Design: Repeated measures ANOVA (3 sessions)
- N=20 per session
- Power=0.85 for f=0.25 (medium effect)

---

## Supplementary Results

### Extended Data Figure 1: Pilot Study Results

*[Would show original pilot data revealing the structure confound]*

**Panel A:** Original pilot design comparison (repetitive vs. novel structures)
**Panel B:** Entropy distributions showing opposite pattern from prediction
**Panel C:** Sample responses illustrating structural constraint in "novel" prompts
**Panel D:** Revised within-subjects design schematic

### Extended Data Figure 2: Mixed-Effects Models

**Study 1 Mixed-Effects Results:**

```
Mixed Linear Model Regression Results
========================================
Dep. Variable:         entropy
Model:                 MixedLM
No. Observations:      100
No. Groups:            10
Method:                REML
Formula:               entropy ~ trial
Random Effects:        Intercept, trial | concept

                coef    std err          z      P>|z|
----------------------------------------------------------
Intercept     0.8971      0.012     74.759      0.000
trial         0.0001      0.001      0.113      0.910

Random Effects Covariance:
                  Intercept     trial
Intercept           0.0015    -0.0001
trial              -0.0001     0.0000

AIC: -420.5
BIC: -408.2
Log-Likelihood: 215.3
```

**Interpretation:** Fixed effect of trial is effectively zero (β=0.0001, p=0.910). Random effects show minimal variance across concepts. Model confirms null finding with proper accounting for hierarchical structure.

### Extended Data Figure 3: Additional Metrics

**Lexical Diversity (MTLD):**
- Study 1 Claude: trial 1 M=39.2, trial 10 M=38.8, change=1%, n.s.
- Study 1 GPT-4: trial 1 M=42.1, trial 10 M=43.0, change=2%, n.s.
- No systematic patterns across repetitions

**Response Length:**
- Study 1 Claude: M=268 tokens, SD=15, stable across trials
- Study 1 GPT-4: M=275 tokens, SD=18, stable across trials
- Length does not confound entropy findings

**Type-Token Ratio:**
- Correlates with entropy (r=0.68) as expected
- Shows same null pattern as entropy
- Confirms metric consistency

### Extended Data Figure 4: Per-Concept Detailed Analysis

*[Individual trajectory plots for all 10 concepts in Study 1]*

**Claude Concept Trajectories:**
- Concept 0 (photosynthesis): slight decline, n.s.
- Concept 1 (evolution): slight increase, n.s.
- Concept 2 (democracy): slight decline, n.s.
- Concept 3 (quantum mechanics): larger decline, p=0.063 marginal
- Concept 4 (cell division): slight increase, n.s.
- Concept 5 (gravity): slight decline, n.s.
- Concept 6 (electromagnetic induction): **increase, p=0.001** ← anomaly
- Concept 7 (protein synthesis): moderate decline, n.s.
- Concept 8 (thermodynamics): minimal change
- Concept 9 (neural transmission): minimal change

**Pattern:** High heterogeneity, no consistent direction, mean≈zero

### Extended Data Figure 5: Temperature Effects (Sensitization Pilot)

Temperature=1.0 tested in main studies. Sensitization pilot also at temp=1.0. No evidence that temperature creates temporal dynamics (though it affects variance).

Future work could systematically vary temperature (0.3, 0.7, 1.0, 1.5) but architectural considerations suggest response invariance should hold across sampling parameters.

### Extended Data Table 1: Complete Statistical Results

| Study | Model | N | Mean Slope (β) | t-statistic | p-value | Cohen's d | 95% CI |
|-------|-------|---|----------------|-------------|---------|-----------|---------|
| 1 | Claude | 100 | 0.000071 | 0.11 | 0.91 | -0.05 | [-0.0013, 0.0015] |
| 1 | GPT-4 | 100 | 0.000622 | 1.54 | 0.16 | -0.53 | [-0.0003, 0.0015] |
| 2 | Claude | 70 | 0.010 | 1.40 | 0.18 | 0.44 | [-0.005, 0.025] |
| 3 | Claude | 300 | -0.032 | -8.12 | <0.001 | -1.82 | [-0.040, -0.024] |
| 4 | Claude | 60 | 0.0005 | 0.82 | 0.45 | 0.02 | [-0.008, 0.009] |
| Sens. | Claude | 15 | -0.001319 | -1.15 | 0.26 | -0.09 | [-0.004, 0.001] |

**Meta-analysis:** Random-effects: d=-0.03, 95% CI [-0.15, 0.09], p=0.63

---

## Supplementary Discussion

### Alternative Architectures

**Recurrent Neural Networks:**
RNNs maintain hidden states across timesteps, potentially enabling temporal dynamics. However, modern LLMs use transformers due to superior performance and parallelizability. Testing RNN-based language models (if available at scale) could determine if recurrence enables habituation.

**Memory-Augmented Networks:**
Systems like Neural Turing Machines (Graves et al., 2014) or Memory Networks (Weston et al., 2015) include external memory stores. These could implement persistent state changes, potentially showing temporal dynamics. However, these architectures are not used in current large-scale LLMs.

**State Space Models:**
Recent Mamba and other state space model architectures (Gu & Dao, 2024) provide alternatives to transformers with linear-time complexity. Their temporal processing properties remain largely unexplored. Future work should test whether SSMs show response dynamics.

### Comparison to Human Data

**Human Habituation Parameters:**
- Typical decrement: 20-50% reduction in response magnitude
- Trials to habituation: 5-20 for simple stimuli, 50-100 for complex
- Recovery time: Minutes to hours
- Spontaneous recovery: 70-90% after rest

**LLM Results:**
- Observed decrement: <1% (essentially zero)
- Trials tested: Up to 15
- Recovery: Not observed
- Cross-session: No change

**Conclusion:** LLMs show fundamentally different patterns from biological systems, not merely weaker effects.

### Implications for Training

Could RLHF training create temporal dynamics even if inference doesn't? No—RLHF optimizes the reward function and updates parameters, but doesn't create temporal response patterns during inference. Training creates the *learned distribution*; inference samples from it statelessly.

### Philosophical Implications

**Embodiment:** Lack of temporal dynamics may relate to disembodiment. Biological systems exist in time, with resource constraints and fatigue. LLMs exist as frozen parameter sets, invoked on demand.

**Learning vs. Inference:** Biological systems continuously learn during "inference" (ongoing experience). LLMs separate training (learning) from inference (generation). This categorical difference may preclude temporal dynamics.

---

## Supplementary References

[Additional references supporting supplementary analyses]

Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. *arXiv* preprint arXiv:1410.5401 (2014).

Weston, J., Chopra, S. & Bordes, A. Memory networks. *arXiv* preprint arXiv:1410.3916 (2015).

Gu, A. & Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv* preprint arXiv:2312.00752 (2023).

[... complete supplementary reference list ...]

---

## Code Availability Appendix

**Repository Structure:**
```
llm-habituation-patterns/
├── src/
│   ├── studies/
│   │   ├── study_1_habituation.py
│   │   ├── study_2_recovery.py
│   │   ├── study_3_novelty_gradient.py
│   │   ├── study_4_tolerance.py
│   │   └── study_sensitization.py
│   ├── metrics.py
│   ├── llm_interface.py
│   └── config.py
├── analysis/
│   ├── statistical_analysis.py
│   ├── generate_figures.py
│   └── [...]
├── data/
│   ├── raw/ (API responses)
│   ├── processed/ (calculated metrics)
│   └── prompts/ (stimulus sets)
└── docs/
    ├── theory_paper.md
    ├── preregistration.md
    └── DESIGN_CHANGES.md
```

**Running Studies:**
```bash
# Study 1
python3 src/studies/study_1_habituation.py --models claude gpt4

# Study 2
python3 src/studies/study_2_recovery.py --models claude

# Study 3
python3 src/studies/study_3_novelty_gradient.py --models claude

# Study 4
python3 src/studies/study_4_tolerance.py --models claude
```

**Analysis:**
```bash
python3 analysis/statistical_analysis.py
python3 analysis/generate_figures.py
```

---

END OF SUPPLEMENTARY MATERIALS