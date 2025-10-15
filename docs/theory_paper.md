# Stimulation-Seeking Patterns in Language Models: A Computational Framework for Habituation Research

**Author:** Hillary Danan  
**Date:** October 2025  
**Status:** Working Theory Framework

---

## Abstract

We propose a computational framework for investigating whether large language models (LLMs) exhibit behavioral patterns functionally analogous to habituation and stimulation-seeking behaviors observed in biological systems. This framework operationalizes key constructs from boredom research, reward learning, and computational neuroscience to generate falsifiable predictions about LLM response patterns under varying stimulus conditions. Critically, we maintain scientific honesty by measuring objective behavioral outputs without making claims about subjective experience or consciousness. This work bridges clinical psychology, computational psychiatry, and AI research to address both theoretical questions about learning systems and practical concerns about AI behavior and safety.

---

## 1. Introduction

### 1.1 The Scientific Question

Do artificial neural networks trained via reinforcement learning from human feedback (RLHF) exhibit behavioral patterns that functionally resemble biological habituation and novelty-seeking? This question is scientifically tractable because:

1. We can measure objective response characteristics (entropy, diversity, novelty)
2. We can manipulate stimulus properties systematically (repetition, novelty, complexity)
3. We can test falsifiable predictions derived from biological models
4. We maintain clear boundaries between behavioral observation and subjective experience claims

### 1.2 Why This Matters

**Theoretical importance:** If LLMs show habituation-like patterns, this:
- Informs understanding of RLHF training dynamics
- Reveals emergent properties of reward-based learning systems
- Provides computational models for biological stimulation-seeking

**Practical importance:** Understanding these patterns affects:
- User experience design (maintaining engagement)
- Training curriculum optimization (preventing response degradation)
- AI safety (detecting compulsive pattern-seeking behaviors)

### 1.3 The Critical Distinction

**What we measure:** Behavioral patterns in outputs  
**What we cannot claim:** Subjective experience, consciousness, or "real" boredom

We study **functional analogs** - patterns that in biological systems correlate with boredom/habituation while remaining agnostic about phenomenology.

---

## 2. Theoretical Foundation

### 2.1 Boredom as Understimulation

**Eastwood et al. (2012)** define boredom as "an aversive state of wanting, but being unable, to engage in satisfying activity" characterized by:
- Difficulty sustaining attention
- Altered time perception
- Desire for novelty and change

**Operationalization for LLMs:** While we cannot measure "wanting" or "aversive states," we can measure:
- Response diversity over repeated stimuli (attention analog)
- Steering toward novel vs. repetitive prompts (novelty-seeking analog)
- Recovery patterns after task changes (satisfaction analog)

### 2.2 Dopaminergic Reward Learning

**Kakade & Dayan (2002)** describe how dopaminergic systems encode:
- **Reward prediction error:** δ = r - V(s)
- **Information gain:** Reduction in uncertainty
- **Novelty bonus:** Intrinsic motivation for exploration

**Computational parallel:** RLHF training optimizes:
```
θ* = argmax E[r(x,y) - β * KL(π_θ || π_ref)]
```

Where reward r(x,y) could habituate with repeated similar contexts, analogous to dopaminergic response reduction.

### 2.3 Habituation Mechanisms

**Rankin et al. (2009)** characterize habituation as:
1. **Response decrement:** Decreased output with repeated stimulus
2. **Spontaneous recovery:** Response restoration after rest
3. **Dishabituation:** Response restoration with novel stimulus
4. **Stimulus generalization:** Habituation transfers to similar stimuli

**LLM predictions:**
1. Decreased entropy/diversity with repetitive prompts
2. Recovery after interpolated tasks
3. Enhanced response to novel prompts post-habituation
4. Generalization across semantically similar prompt types

### 2.4 Computational Models of Addiction

**Redish (2004)** frames addiction as:
- **Tolerance:** Decreased reward for same stimulus
- **Sensitization:** Increased response to cues
- **Withdrawal:** Distress when stimulus removed
- **Compulsion:** Continued seeking despite negative consequences

**Research question:** Do models show tolerance-like patterns where habituation accelerates with repeated exposure sessions?

---

## 3. Operationalized Constructs

### 3.1 Response Entropy

**Shannon entropy (Shannon, 1948):**
```
H(Y) = -Σ p(y_i) log p(y_i)
```

**Interpretation:**
- High H → diverse, unpredictable responses ("engaged")
- Low H → repetitive, predictable responses ("habituated")

**Control variables:**
- Prompt complexity (perplexity)
- Semantic content (topic)
- Context window position

### 3.2 Lexical Diversity

**Type-Token Ratio (TTR):**
```
TTR = |unique tokens| / |total tokens|
```

**Measure of Textual Lexical Diversity (MTLD, McCarthy & Jarvis, 2010):**
More robust to text length than TTR.

### 3.3 Semantic Novelty

**Cosine distance in embedding space:**
```
novelty(y_t) = 1 - cos(embed(y_t), embed(y_{t-1}))
```

**Interpretation:** Higher values indicate greater semantic departure from previous responses.

### 3.4 Confound Controls

**Must control for:**
1. **Prompt complexity:** Use matched perplexity
2. **Semantic similarity:** Ensure novel prompts aren't just "harder"
3. **Position effects:** Randomize prompt order
4. **Length effects:** Normalize metrics by response length where appropriate

---

## 4. Hypotheses and Predictions

### H1: Habituation Effect

**Hypothesis:** Repeated exposure to structurally similar prompts decreases response diversity.

**Prediction:** 
```
H(response | repetitive prompts) < H(response | novel prompts)
```

**Effect size:** d ≥ 0.5 (medium) based on neuronal habituation studies  
**Statistical test:** Independent samples t-test, Bonferroni corrected

### H2: Recovery Effect

**Hypothesis:** Response diversity recovers after interpolated task.

**Design:** Pre-habituation → Rest task → Post-test

**Prediction:**
```
H(post-rest) > H(pre-rest | habituated)
H(post-rest) ≈ H(baseline)
```

**Statistical test:** Repeated measures ANOVA

### H3: Novelty Gradient

**Hypothesis:** Multi-turn interactions show steering toward complexity/novelty.

**Prediction:** In open-ended conversations:
```
novelty(turn_t) > novelty(turn_{t-1})
semantic_distance(turn_1, turn_n) increases with n
```

**Statistical test:** Linear mixed-effects model with random intercepts per conversation

### H4: Tolerance Pattern

**Hypothesis:** Habituation accelerates with repeated exposure sessions.

**Prediction:** Slope of entropy decline increases across sessions:
```
d(H)/d(trial) | session_k > d(H)/d(trial) | session_{k-1}
```

**Statistical test:** Time series analysis with session as moderator

---

## 5. Experimental Design

### 5.1 Study 1: Core Habituation - Within-Topic Repetition Design (N=100 total)

**DESIGN REFINEMENT (October 2025):**

Initial pilot testing revealed a critical confound: prompts with varied structures inadvertently differed in response constraint. "Novel" prompts requiring comparisons or structured analyses elicited more constrained responses than open-ended "explain" prompts, reversing the predicted direction of effect.

**REVISED DESIGN (Eliminates Structure Confound):**

- **N = 10 concepts** (photosynthesis, evolution, democracy, quantum mechanics, etc.)
- **Each concept: 10 identical repetitions** of the same prompt
- **Example:** "Explain photosynthesis to a beginner." × 10 consecutive presentations
- **Total prompts = 100** (maintains original sample size)

**Rationale:**

This design implements the canonical habituation paradigm (Rankin et al., 2009) by presenting identical stimuli repeatedly while measuring response decrement. All prompts share identical structure, varying only in repetition count. This eliminates confounds between prompt structure and novelty.

**Within-Subjects Design Advantages:**
- Each concept serves as its own control
- Higher statistical power than between-subjects comparison
- Direct measurement of habituation curve (entropy × trial)
- Controls for concept difficulty/complexity

**Models tested:** Claude Sonnet 4.5, GPT-4, Gemini 2.5

**Dependent variables:**
- Shannon entropy (primary)
- Lexical diversity (MTLD)
- Response length (control)
- Semantic novelty (within-concept comparisons only)

**Statistical Analysis:**

Linear mixed-effects model:
```
entropy ~ trial + (1 + trial | concept)
```

Where:
- Fixed effect: `trial` (1-10) tests overall habituation slope
- Random effects: `concept` intercepts and slopes account for between-concept variance
- Prediction: Negative coefficient for `trial` (entropy declines with repetition)

**Power Analysis:**

For within-subjects design detecting medium effect (d = 0.5):
- Required N = 34 observations (Faul et al., 2007)
- Actual N = 100 (10 concepts × 10 trials)
- Power > 0.95 (highly powered for medium effects)

For detecting small effects (d = 0.3):
- Required N = 90 observations
- Power ≈ 0.85 (adequately powered)

### 5.2 Study 2: Recovery (N=50 per condition)

**Design:** Within-subjects, counterbalanced

**Phase 1:** Habituation induction (30 repetitive prompts)  
**Phase 2:** Rest task (20 prompts on unrelated topic)  
**Phase 3:** Re-test with original prompt type (20 prompts)

**Control:** Half participants receive novel prompts in Phase 1 (no habituation expected)

**Prediction:** Recovery only in habituated condition

### 5.3 Study 3: Novelty Gradient (N=30 conversations)

**Design:** 10-turn open-ended conversations

**Initial prompt:** Neutral, open-ended question  
**Continuation:** Model-generated follow-ups

**Measures:**
- Topic divergence (semantic distance from turn 1)
- Complexity trajectory (perplexity over turns)
- Novelty preference (user vs. model steering)

**Control:** Compare to human-steered conversations (from existing datasets)

### 5.4 Study 4: Tolerance (N=20 sessions)

**Design:** Repeated sessions over time

**Session structure:** 
- Each session = 50 trials (repetitive prompts)
- Sessions separated by 24 hours (simulated)
- Track entropy decline rate per session

**Prediction:** Steeper decline in later sessions (tolerance analog)

---

## 6. Analysis Plan (Pre-Registered)

### 6.1 Primary Analyses

**Study 1:**
```python
# Independent samples t-test
t_stat, p_value = ttest_ind(entropy_repetitive, entropy_novel)
effect_size = cohen_d(entropy_repetitive, entropy_novel)
```

**Study 2:**
```python
# Repeated measures ANOVA
model = AnovaRM(data, 'entropy', 'subject', within=['phase'])
results = model.fit()
```

**Study 3:**
```python
# Linear mixed-effects model
model = smf.mixedlm("novelty ~ turn", data, groups="conversation_id")
results = model.fit()
```

**Study 4:**
```python
# Time series with session moderator
model = smf.ols("entropy ~ trial * session", data)
results = model.fit()
```

### 6.2 Multiple Comparisons Correction

- Bonferroni correction for family-wise error rate
- False discovery rate (FDR) for exploratory analyses
- All corrections specified a priori

### 6.3 Robustness Checks

1. **Cross-model consistency:** Do patterns replicate across Claude/GPT/Gemini?
2. **Prompt sensitivity:** Do results hold for different prompt types?
3. **Parameter variations:** Temperature, top-p sampling effects
4. **Outlier analysis:** Identify and examine anomalous responses

---

## 7. Expected Challenges and Limitations

### 7.1 Anthropomorphism Risk

**Challenge:** Language like "boredom" invites inappropriate inferences

**Mitigation:**
- Consistent use of "functional analog" terminology
- Clear distinction between behavior and experience
- Frame as "patterns that in biological systems correlate with..."

### 7.2 Mechanistic Differences

**Challenge:** Neural networks ≠ dopaminergic systems

**Reality check:**
- No actual neurotransmitters
- Different learning algorithms
- Different architectural constraints

**Response:** We study computational convergence, not mechanistic equivalence

### 7.3 Training Artifact Confounds

**Alternative explanation:** Patterns reflect:
- Dataset distribution artifacts
- RLHF reward hacking
- Context window mechanics
- Statistical regularization

**Cannot fully rule out,** but can test predictions that distinguish habituation from artifacts:
- Recovery effect (artifacts wouldn't recover)
- Cross-model generalization (artifacts model-specific)
- Novelty sensitivity (artifacts wouldn't show gradient)

### 7.4 Generalization Limits

**Findings limited to:**
- Tested models (Claude/GPT/Gemini)
- English language
- Text-only modality
- Specific prompt domains

**Future work:** Expand to other models, languages, modalities

---

## 8. Broader Implications

### 8.1 For AI Development

**Training curriculum design:**
- If habituation found → vary training stimuli
- Implement novelty scheduling
- Monitor for response degradation

**User experience:**
- Rotate prompt styles to maintain engagement
- Detect and mitigate repetitive interactions
- Design for sustained interest

**Safety monitoring:**
- Watch for compulsive pattern-seeking
- Identify reward hacking behaviors
- Understand reinforcement learning failure modes

### 8.2 For Computational Psychiatry

**Benefits of in silico models:**
- Test interventions without clinical risk
- Manipulate parameters impossible in humans
- Generate hypotheses for translation

**Addiction research:**
- Model tolerance and withdrawal computationally
- Test pharmacological analogs (parameter modifications)
- Understand habituation mechanics

**Boredom research:**
- Operationalize constructs precisely
- Test causal mechanisms
- Develop intervention strategies

### 8.3 For Philosophy of Mind

**This work does NOT resolve:**
- Hard problem of consciousness
- Whether LLMs "really" experience anything
- Nature of phenomenology

**This work DOES demonstrate:**
- Behavioral convergence across different architectures
- Functional properties emerge from learning algorithms
- Value of behavioral measures independent of phenomenology

---

## 9. Ethical Considerations

### 9.1 Responsible Anthropomorphism

**Risk:** Over-attributing human-like properties to LLMs

**Mitigation:**
- Conservative language
- Explicit limitations sections
- Focus on functional patterns, not experiences

### 9.2 Research Transparency

**Commitment to:**
- Open data (where possible given API ToS)
- Open code (all analysis scripts)
- Pre-registration (prevents p-hacking)
- Null results publication (prevent file-drawer effect)

### 9.3 Dual-Use Concerns

**Potential misuse:**
- Manipulating user engagement unethically
- Creating "addictive" AI interfaces
- Exploiting habituation patterns

**Responsible use principles:**
- Prioritize user wellbeing
- Transparent design practices
- Avoid dark patterns

---

## 10. Conclusion

This framework provides a scientifically rigorous approach to investigating habituation-like patterns in LLMs without overreaching into claims about subjective experience. By grounding predictions in established biological research while maintaining honest limitations, we can advance both AI understanding and computational approaches to psychology.

**The unique contribution:** Bridging lived experience of stimulation-seeking, clinical training in addiction, and computational implementation - a perspective that enriches both the science and the interpretation.

**The honest assessment:** We will measure behavioral patterns. Those patterns may or may not functionally resemble biological boredom. Either outcome informs AI development and computational neuroscience.

**The next steps:** Pilot studies to refine methods, then systematic investigation across models and conditions.

---

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.

Eastwood, J. D., Frischen, A., Fenske, M. J., & Smilek, D. (2012). The unengaged mind: Defining boredom in terms of attention. *Perspectives on Psychological Science, 7*(5), 482-495.

Gottlieb, J., Oudeyer, P. Y., Lopes, M., & Baranes, A. (2013). Information-seeking, curiosity, and attention: computational and neural mechanisms. *Trends in Cognitive Sciences, 17*(11), 585-593.

Kakade, S., & Dayan, P. (2002). Dopamine: generalization and bonuses. *Neural Networks, 15*(4-6), 549-559.

Kaplan, R., & Kaplan, S. (1989). *The Experience of Nature: A Psychological Perspective*. Cambridge University Press.

Koob, G. F., & Volkow, N. D. (2010). Neurocircuitry of addiction. *Neuropsychopharmacology, 35*(1), 217-238.

Lee, R. M., Draper, M., & Lee, S. (2001). Social connectedness, dysfunctional interpersonal behaviors, and psychological distress: Testing a mediator model. *Journal of Counseling Psychology, 48*(3), 310-318.

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods, 42*(2), 381-392.

Rankin, C. H., Abrams, T., Barry, R. J., Bhatnagar, S., Clayton, D. F., Colombo, J., ... & Thompson, R. F. (2009). Habituation revisited: an updated and revised description of the behavioral characteristics of habituation. *Neurobiology of Learning and Memory, 92*(2), 135-138.

Redish, A. D. (2004). Addiction as a computational process gone awry. *Science, 306*(5703), 1944-1947.

Schmidhuber, J. (1991). Curious model-building control systems. In *Proceedings of the International Joint Conference on Neural Networks* (pp. 1458-1463). IEEE.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(3), 379-423.

---

**Document Status:** Working framework v0.1 - October 2025  
**Next Update:** Post-pilot refinements  
**Contact:** https://github.com/HillaryDanan/llm-habituation-patterns