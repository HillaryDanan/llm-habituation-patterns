# Follow-Up Studies: What LLMs DO Show

## Based on Response Invariance Framework

Since LLMs don't show temporal dynamics, what DO they show? These studies test the positive predictions from our framework.

---

## Study A: Context Window Position Effects

**Hypothesis:** LLM responses depend on *context content and position*, not temporal order.

**Design:**
- Vary conversation history length: 0, 5, 10, 20 turns
- Test same query at different context positions
- Measure: response characteristics vs. context factors

**Predictions:**
1. Recent context >> temporal order (recency effects)
2. Content matters >> length (semantic relevance)
3. Position in window >> actual time elapsed

**Implementation:**
```python
for history_length in [0, 5, 10, 20]:
    context = previous_turns[-history_length:]
    response = model.generate(query, context=context)
    # Measure response characteristics
```

**N:** 50 queries × 4 history lengths × 3 models = 600 prompts

**Expected result:** Strong context effects (d>0.5), zero temporal effects

---

## Study B: Conversational Grounding Mechanisms

**Hypothesis:** LLMs implement computational grounding without explicit mechanisms.

**Design:**
- Track semantic convergence in human vs. LLM conversations
- Compare grounding markers (anaphora, lexical overlap, topic stability)
- Test if patterns match human dialogue literature

**Metrics:**
1. Semantic distance trajectories
2. Lexical alignment (Pickering & Garrod, 2004)
3. Anaphora resolution patterns
4. Topic drift vs. focus

**Comparison:** 
- LLM multi-turn data (Study 3 reanalysis)
- Human dialogue corpus (e.g., Switchboard)
- Computational grounding models

**Peer-reviewed basis:**
- Clark & Brennan (1991) - grounding theory
- Pickering & Garrod (2004) - alignment
- Brennan & Clark (1996) - conceptual pacts

**Expected result:** LLMs show functional grounding patterns similar to humans

---

## Study C: Prompt Structure Effects

**Hypothesis:** Structural variation produces larger effects than temporal variation.

**Design:**
- Systematic prompt structure manipulation:
  1. Specificity gradient (vague → precise)
  2. Constraint level (open → closed)
  3. Framing (neutral → opinionated)
  4. Format (question → instruction → completion)

**Example manipulations:**
- Vague: "Tell me about X"
- Precise: "Explain the mechanism by which X causes Y in Z contexts"
- Open: "Discuss X"
- Closed: "List exactly 3 facts about X"

**Metrics:** Same as Study 1 (entropy, MTLD, etc.)

**Predictions:**
- Structure effects: d>0.8 (large)
- Temporal effects: d<0.1 (trivial)
- Structure >> time by order of magnitude

**N:** 10 structures × 10 repetitions × 3 models = 300 prompts

**Analysis:** 
- Two-way ANOVA: structure × repetition
- Expected: Main effect of structure, no effect of repetition
- Interaction: None

**Expected result:** Structure dominates, confirming framework

---

## Study D: Temperature as Exploration Parameter

**Hypothesis:** Temperature affects diversity via sampling, not temporal dynamics.

**Design:**
- Test temperature effects: 0.3, 0.7, 1.0, 1.5
- Same prompts, varied temperature
- No temporal manipulation

**Metrics:**
- Response diversity (entropy, MTLD)
- Semantic exploration
- Creativity measures (e.g., remote associates)

**Predictions:**
1. Strong temperature main effect (sampling mechanism)
2. No temperature × repetition interaction (no temporal component)
3. Linear relationship: temp ↑ → diversity ↑

**N:** 20 prompts × 4 temperatures × 3 models = 240 prompts

**Analysis:**
- ANOVA: temperature × repetition
- Expected: Main effect temp (F>10.0), no interaction

**Expected result:** Temperature matters, repetition doesn't

---

## Study E: Explicit Memory Systems

**Hypothesis:** External memory enables temporal effects absent in base models.

**Design:**
- Implement external memory (vector database)
- Store previous responses with metadata
- Retrieve relevant history during generation
- Test if this creates "pseudo-temporal" effects

**Conditions:**
1. No memory (baseline, should show invariance)
2. Short-term memory (last 10 responses)
3. Long-term memory (all responses, similarity-based retrieval)

**Test:** Do responses now show adaptation with memory?

**Predictions:**
- No memory: Invariance (replicates Study 1)
- With memory: Context-dependent variation (not true temporal dynamics)
- Memory acts as explicit context, not persistent state

**This tests architectural augmentation approach**

---

## Study F: Cross-Model Architectural Comparison

**Hypothesis:** Response invariance is specific to standard transformers.

**Design:**
- Test alternative architectures if available:
  1. Standard transformer (Claude, GPT-4) - baseline
  2. State space models (Mamba, if accessible)
  3. RNN-based models (if any exist at scale)
  4. Memory-augmented transformers

**Same protocol as Study 1:**
- 10 concepts × 10 repetitions
- Measure habituation patterns

**Predictions:**
- Transformers: Invariance (replication)
- SSMs: Unknown (test if linear recurrence enables dynamics)
- RNNs: Potentially show dynamics (recurrent state)
- Memory-augmented: Context-dependent, not temporal

**Expected result:** Architecture determines temporal properties

---

## Implementation Priority

**Phase 1 (Immediate - strengthen main paper):**
1. Study C: Prompt structure effects
   - Quick to run
   - Strengthens "structure > time" claim
   - Direct comparison to Study 1

**Phase 2 (Build on findings):**
1. Study A: Context window position
2. Study B: Grounding mechanisms reanalysis
3. Study D: Temperature systematic test

**Phase 3 (Architectural):**
1. Study E: External memory
2. Study F: Alternative architectures (if available)

---

## Estimated Costs & Time

**Study A:** 600 prompts × $0.05 = ~$30, ~2 hours runtime
**Study B:** Reanalysis of existing data, ~1 week analysis
**Study C:** 300 prompts × $0.05 = ~$15, ~1 hour runtime
**Study D:** 240 prompts × $0.05 = ~$12, ~1 hour runtime
**Study E:** 300 prompts × $0.05 + development time = ~$15 + 1 week dev
**Study F:** Depends on model availability

**Total for Phase 1+2:** ~$70, ~1-2 weeks work

---

## Expected Publications

**Paper 2:** "What Large Language Models DO Show: Context, Structure, and Grounding"
- Studies A, B, C, D
- Positive characterization after null findings
- Computational Linguistics or Cognitive Science journal

**Paper 3:** "Architectural Requirements for Temporal Dynamics in Language Models"
- Studies E, F
- Tests augmentation and alternatives
- NeurIPS or ICLR

**Paper 4 (optional):** "Conversational Grounding in LLMs: Comparing Artificial and Human Dialogue"
- Deep dive into Study B
- Comparison to human data
- Psycholinguistics journal

---

## Git Repository Structure for Follow-Ups

```
llm-habituation-patterns/
├── src/
│   ├── studies/
│   │   ├── [existing studies 1-4...]
│   │   ├── study_context_position.py
│   │   ├── study_prompt_structure.py
│   │   ├── study_temperature_effects.py
│   │   └── study_external_memory.py
│   └── [...]
├── analysis/
│   ├── grounding_analysis.py
│   └── [...]
└── docs/
    ├── paper_response_invariance.md (Paper 1)
    ├── paper_positive_findings.md (Paper 2)
    └── paper_architectural_requirements.md (Paper 3)
```

---

## Ready to Implement

All studies designed with:
- Clear hypotheses
- Peer-reviewed basis
- Specific predictions
- Implementation details
- Analysis plans
- Power/cost estimates

**Next step:** Choose which to implement first based on:
1. Paper 1 reviewer feedback (may request specific follow-ups)
2. Scientific priorities (grounding most interesting?)
3. Resource availability (time, API credits)

---

END OF FOLLOW-UP STUDIES PLAN